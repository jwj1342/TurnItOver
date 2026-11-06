"""Shared base/adapter inference without browser or simulator dependencies."""
from __future__ import annotations

from contextlib import nullcontext
import json
from turnitover.models.local_qwen import LocalQwen
from turnitover.models.client import ModelResult, ModelError


class SharedModel:
    """One base model with independently selectable action and diagnosis adapters."""
    def __init__(self, model_path, adapter, action_tokens, final_tokens, diagnosis_adapter=None):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        self.local = LocalQwen(model_path)
        self.local.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.has_adapter = adapter is not None
        self.has_diagnosis_adapter = diagnosis_adapter is not None
        if adapter is not None:
            model = PeftModel.from_pretrained(base, adapter, adapter_name="action", local_files_only=True)
        elif diagnosis_adapter is not None:
            model = PeftModel.from_pretrained(base, diagnosis_adapter, adapter_name="diagnosis",
                                              local_files_only=True)
        else:
            model = base
        if adapter is not None and diagnosis_adapter is not None:
            model.load_adapter(diagnosis_adapter, adapter_name="diagnosis", local_files_only=True)
        self.local.model = model.eval()
        self.action_tokens, self.final_tokens = action_tokens, final_tokens

    def call(self, enabled, final=False):
        def generate(prompt, images):
            if enabled and not self.has_adapter:
                raise ValueError("Action SFT requested without an adapter")
            self.local.max_tokens = self.final_tokens if final else self.action_tokens
            if final and self.has_diagnosis_adapter:
                self.local.model.set_adapter("diagnosis")
                context = nullcontext()
            elif enabled:
                self.local.model.set_adapter("action")
                context = nullcontext()
            elif self.has_adapter or self.has_diagnosis_adapter:
                context = self.local.model.disable_adapter()
            else:
                context = nullcontext()
            with context:
                return self.local(prompt, images)
        return generate


class RemoteModel:
    """Send only the public model input through an SSH-forwarded endpoint."""
    def __init__(self, endpoint):
        from urllib.parse import urlsplit
        import urllib.request
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.username or parsed.password:
            raise ValueError("Model endpoint must be an unauthenticated HTTP loopback SSH tunnel")
        self.endpoint = endpoint
        # Environment proxies must never receive the public image transport for
        # this local, unauthenticated SSH-forwarded service.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, enabled, final=False):
        def generate(prompt, images):
            import base64
            import urllib.request
            payload = dict(prompt=prompt, images=[base64.b64encode(p.read_bytes()).decode() for p in images],
                           enabled=enabled, final=final)
            request = urllib.request.Request(self.endpoint, json.dumps(payload).encode(),
                                             {"Content-Type": "application/json"})
            try:
                with self.opener.open(request, timeout=600) as response:
                    return ModelResult(**json.load(response))
            except (OSError, ValueError) as exc:
                raise ModelError(f"Remote inference failed: {exc}") from exc
        return generate

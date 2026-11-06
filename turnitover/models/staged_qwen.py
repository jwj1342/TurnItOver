"""Local Qwen inference with separately selectable action and diagnosis LoRAs."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path

from turnitover.models.local_qwen import LocalQwen


def adapter_provenance(path: Path | None) -> dict | None:
    if path is None:
        return None
    config = path / "adapter_config.json"
    weights = sorted(path.glob("adapter_model.*"))
    if not config.is_file() or len(weights) != 1:
        raise ValueError(f"Expected one adapter config and weight under {path}")
    return {"path": str(path), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "weight": weights[0].name, "weight_sha256": hashlib.sha256(weights[0].read_bytes()).hexdigest()}


class StagedQwen:
    """Load one Qwen base and route action/final calls to distinct named adapters."""
    def __init__(self, model_path: Path, action_adapter: Path, diagnosis_adapter: Path | None,
                 action_tokens: int = 128, final_tokens: int = 1024):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        model = PeftModel.from_pretrained(base, action_adapter, adapter_name="action", local_files_only=True)
        if diagnosis_adapter is not None:
            model.load_adapter(diagnosis_adapter, adapter_name="diagnosis", local_files_only=True)
        self.local = LocalQwen(model_path)
        self.local.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.local.model = model.eval()
        self.has_diagnosis_adapter = diagnosis_adapter is not None
        self.action_tokens, self.final_tokens = action_tokens, final_tokens
        self.provenance = {"base": str(model_path), "action_adapter": adapter_provenance(action_adapter),
                           "diagnosis_adapter": adapter_provenance(diagnosis_adapter)}

    def action_call(self, prompt, images):
        self.local.model.set_adapter("action")
        self.local.max_tokens = self.action_tokens
        return self.local(prompt, images)

    def final_call(self, prompt, images):
        self.local.max_tokens = self.final_tokens
        if self.has_diagnosis_adapter:
            self.local.model.set_adapter("diagnosis")
            context = nullcontext()
        else:
            context = self.local.model.disable_adapter()
        with context:
            return self.local(prompt, images)

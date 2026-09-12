"""Lazy local Qwen3-VL inference. Weights must already exist; never download at inference time."""
from __future__ import annotations

import time
from pathlib import Path

from turnitover.models.client import ModelError, ModelResult


class LocalQwen:
    def __init__(self, path: Path, max_tokens: int = 1536):
        self.path = path.resolve()
        self.max_tokens = max_tokens
        self.model = self.processor = None

    def __call__(self, prompt: str, images: list[Path]) -> ModelResult:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise ModelError("Local Qwen requires an allocated CUDA GPU; submit the Qwen Slurm job")
        if self.model is None:
            if not (self.path / "model.safetensors").is_file():
                raise ModelError("Qwen weights missing; run scripts/download_qwen.py first")
            self.processor = AutoProcessor.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.path, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map="auto").eval()
        start = time.perf_counter()
        content = [{"type": "image", "image": str(p.resolve())} for p in images]
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        new_tokens = output[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finished = new_tokens[0, -1].item() in eos
        return ModelResult(text, self.path.name,
                           {"input_tokens": inputs.input_ids.shape[1], "output_tokens": new_tokens.shape[1]},
                           "completed" if finished else "incomplete", (time.perf_counter() - start) * 1000)

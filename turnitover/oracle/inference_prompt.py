"""Model-visible inference prefix, independent of training target processing."""
from __future__ import annotations


def tokenize_inference_prompt(processor, record, images, *, model_type=None):
    from turnitover.oracle.paired_observations import user_content
    if len(images) != len(record["images"]):
        raise ValueError("Image roster mismatch")
    prompt = [{"role": "user", "content": user_content(record, images)}]
    options = {"enable_thinking": False} if model_type == "qwen3_5" else {}
    return prompt, processor.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", **options)

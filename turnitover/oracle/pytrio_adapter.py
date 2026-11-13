"""Encode the shared SFT input for PyTRIO without connecting to the service.

Training callers pass the returned Datum to cross_entropy, then perform their
optimizer step. Rank and alpha are separate from this encoding: the PyTRIO
create_lora_training_client API used by the pilot exposes rank, not alpha.
"""
from __future__ import annotations

import io

import numpy as np

from turnitover.oracle.paired_observations import user_content


IMAGE_PAD = "<|image_pad|>"


def model_input(record, images, tokenizer, image_processor, *, completion=None, sdk=None):
    """Return a sampling ModelInput, or an assistant-only training Datum.

    ``sdk`` is an optional PyTRIO-compatible module for offline verification. The
    production dependency is imported lazily; importing this module makes no API
    requests. Input ordering comes from the same helper used by local inference.
    """
    if sdk is None:
        import pytrio as sdk

    content = user_content(record, images)
    public_content, ordered_images = [], []
    for item in content:
        if item["type"] == "image":
            ordered_images.append(item["image"])
            public_content.append({"type": "image", "image": f"image_{len(ordered_images) - 1}"})
        else:
            if IMAGE_PAD in item["text"]:
                raise ValueError("Reserved image marker in public text")
            public_content.append(item)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": public_content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    parts = rendered.split(IMAGE_PAD)
    if len(parts) != len(ordered_images) + 1:
        raise ValueError("Chat template image markers differ from image roster")

    chunks = []
    for index, image in enumerate(ordered_images):
        chunks.append(sdk.types.EncodedTextChunk(
            tokens=tokenizer.encode(parts[index], add_special_tokens=False)))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        patches = image_processor.get_number_of_image_patches(
            image.height, image.width, images_kwargs={})
        image_tokens = patches // image_processor.merge_size**2
        if image_tokens < 1:
            raise ValueError("Image processor returned an empty image token sequence")
        chunks.append(sdk.ImageChunk(data=buffer.getvalue(), format="png",
                                     expected_tokens=image_tokens))
    chunks.append(sdk.types.EncodedTextChunk(
        tokens=tokenizer.encode(parts[-1], add_special_tokens=False)))
    prompt = sdk.ModelInput(chunks=chunks)
    if completion is None:
        return prompt
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("Assistant completion must be nonempty text")
    target = tokenizer.encode(completion + "<|im_end|>", add_special_tokens=False)
    if len(target) < 2:
        raise ValueError("Assistant completion requires at least two tokens")
    training_input = sdk.ModelInput(chunks=[
        *chunks, sdk.types.EncodedTextChunk(tokens=target[:-1]),
    ])
    if len(prompt) < 1 or len(training_input) != len(prompt) + len(target) - 1:
        raise ValueError("Causal target token alignment failed")
    targets = np.zeros(len(training_input), dtype=np.int64)
    weights = np.zeros(len(training_input), dtype=np.float32)
    start = len(prompt) - 1
    targets[start:] = target
    weights[start:] = 1.0
    return sdk.Datum(model_input=training_input, loss_fn_inputs={
        "target_tokens": targets, "weights": weights,
    })

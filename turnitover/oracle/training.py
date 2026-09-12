"""Processor adapter with assistant-only labels for exported multimodal examples."""
from __future__ import annotations


def tokenize_example(processor, record, images):
    """One example; prefix includes all image tokens and must remain unsupervised."""
    import torch
    if len(images) != len(record["images"]):
        raise ValueError("Image roster mismatch")
    user, assistant = record["messages"]
    if user["role"] != "user" or assistant["role"] != "assistant":
        raise ValueError("Expected one user/assistant pair")
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": user["content"]})
    prompt = [{"role": "user", "content": content}]
    prefix = processor.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt")
    full = processor.apply_chat_template(prompt + [{"role": "assistant", "content": [{"type": "text", "text": assistant["content"]}]}],
                                        tokenize=True, add_generation_prompt=False,
                                        return_dict=True, return_tensors="pt")
    length = prefix["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :length], prefix["input_ids"]):
        raise ValueError("Chat template prefix differs: cannot safely mask supervision")
    labels = full["input_ids"].clone()
    labels[:, :length] = -100
    labels[full["attention_mask"] == 0] = -100
    if not torch.any(labels != -100):
        raise ValueError("Empty assistant target")
    full["labels"] = labels
    return full


def weighted_assistant_loss(logits, labels, weights):
    """Token-average each assistant target, then weight targets within packed groups.

    Pack every variant of each observable group together so its loss weights sum
    to one; mixing isolated variants in arbitrary batches changes normalization.
    """
    import torch
    import torch.nn.functional as F
    weights = torch.as_tensor(weights, device=logits.device, dtype=torch.float32)
    if weights.shape != (logits.shape[0],) or not torch.all(torch.isfinite(weights) & (weights > 0)):
        raise ValueError("One positive finite loss weight per example required")
    target = labels[:, 1:]
    valid = target != -100
    if not torch.all(valid.sum(dim=1) > 0):
        raise ValueError("Every target needs supervised assistant tokens")
    losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                             target.reshape(-1), ignore_index=-100, reduction="none")
    means = losses.reshape_as(target).sum(dim=1) / valid.sum(dim=1)
    return (means * weights).sum() / weights.sum()

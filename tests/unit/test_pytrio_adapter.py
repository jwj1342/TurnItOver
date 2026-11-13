"""Exercise multimodal ordering and causal loss masks without a PyTRIO service."""
from dataclasses import dataclass
from io import BytesIO
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from turnitover.oracle.pytrio_adapter import IMAGE_PAD, model_input


@dataclass
class Text:
    tokens: list


@dataclass
class Picture:
    data: bytes
    format: str
    expected_tokens: int


@dataclass
class Input:
    chunks: list

    def __len__(self):
        return sum(len(c.tokens) if isinstance(c, Text) else c.expected_tokens for c in self.chunks)


@dataclass
class Datum:
    model_input: Input
    loss_fn_inputs: dict


SDK = SimpleNamespace(types=SimpleNamespace(EncodedTextChunk=Text), ImageChunk=Picture,
                      ModelInput=Input, Datum=Datum)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "USER:" + "".join(IMAGE_PAD if c['type'] == 'image' else c['text']
                                 for c in messages[0]['content']) + "ASSISTANT:"

    def encode(self, text, **kwargs):
        return list(text.encode())


PROCESSOR = SimpleNamespace(merge_size=2,
                            get_number_of_image_patches=lambda h, w, images_kwargs: 16)


def example():
    paths = [f'{role}_{state}.png' for state in ('zero', 'mid', 'limit')
             for role in ('reference', 'candidate')]
    payload = {"observation_format": "reference_candidate_pairs_v1", "context": {"views": ["front"]},
               "history": [{"step": index, "state": state,
                            "action": {"type": "request_view", "view_id": "front"},
                            "reference_image_ref": paths[2 * index], "image_ref": paths[2 * index + 1]}
                           for index, state in enumerate(('zero', 'mid', 'limit'))],
               "image_order": paths}
    record = {"images": paths, "messages": [
        {"role": "user", "content": 'Protocol\n' + json.dumps(payload)},
        {"role": "assistant", "content": '{}'}]}
    images = [Image.new('RGB', (4, 4), (i * 30, 0, 0)) for i in range(6)]
    return record, images


def test_pytrio_preserves_shared_pair_labels_and_image_bytes():
    record, images = example()
    prompt = model_input(record, images, Tokenizer(), PROCESSOR, sdk=SDK)
    pictures = [c for c in prompt.chunks if isinstance(c, Picture)]
    assert len(pictures) == 6
    assert [Image.open(BytesIO(c.data)).getpixel((0, 0)) for c in pictures] == [im.getpixel((0, 0)) for im in images]
    assert all(c.expected_tokens == 4 for c in pictures)
    texts = [bytes(c.tokens).decode() for c in prompt.chunks if isinstance(c, Text)]
    assert 'Observation 0, state=zero, view=front: reference' in texts[0]
    assert 'Observation 0: candidate' in texts[1]
    assert 'Observation 1, state=mid, view=front: reference' in texts[2]
    assert 'Observation 2: candidate' in texts[5]
    assert texts[-1] == 'ASSISTANT:'


def test_pytrio_datum_predicts_all_assistant_tokens_and_no_prompt_tokens():
    record, images = example()
    tokenizer = Tokenizer()
    prompt = model_input(record, images, tokenizer, PROCESSOR, sdk=SDK)
    completion = '{"verdict":{}}'
    datum = model_input(record, images, tokenizer, PROCESSOR, completion=completion, sdk=SDK)
    target = tokenizer.encode(completion + '<|im_end|>')
    start = len(prompt) - 1
    weights = datum.loss_fn_inputs['weights']
    assert np.all(weights[:start] == 0)
    assert weights[start:].tolist() == [1] * len(target)
    assert datum.loss_fn_inputs['target_tokens'][start:].tolist() == target
    assert datum.model_input.chunks[-1].tokens == target[:-1]
    assert len(datum.model_input) == len(prompt) + len(target) - 1


def test_pytrio_rejects_bad_roster_and_empty_completion():
    record, images = example()
    with pytest.raises(ValueError, match='roster'):
        model_input(record, images[:-1], Tokenizer(), PROCESSOR, sdk=SDK)
    with pytest.raises(ValueError, match='nonempty'):
        model_input(record, images, Tokenizer(), PROCESSOR, sdk=SDK, completion='')
    record['messages'][0]['content'] = record['messages'][0]['content'].replace('Protocol', IMAGE_PAD)
    with pytest.raises(ValueError, match='Reserved image marker'):
        model_input(record, images, Tokenizer(), PROCESSOR, sdk=SDK)

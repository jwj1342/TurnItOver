from contextlib import contextmanager
import sys
from types import SimpleNamespace

from turnitover.models import action_qwen


class FakeBase:
    def cuda(self):
        return self

    def eval(self):
        return self


class FakePeft(FakeBase):
    def __init__(self):
        self.loaded = []
        self.active = None
        self.disabled = False

    def load_adapter(self, path, *, adapter_name, local_files_only):
        self.loaded.append((path, adapter_name, local_files_only))

    def set_adapter(self, name):
        self.active = name

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


class FakeLocalQwen:
    def __init__(self, path):
        self.path = path
        self.model = None
        self.processor = None
        self.max_tokens = None

    def __call__(self, prompt, images):
        return (self.model.active if hasattr(self.model, "active") else "base",
                self.model.disabled if hasattr(self.model, "disabled") else False,
                self.max_tokens)


def test_shared_model_separates_action_diagnosis_and_base(monkeypatch):
    peft_model = FakePeft()
    monkeypatch.setattr(action_qwen, "LocalQwen", FakeLocalQwen)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: "processor"),
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeBase()),
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: peft_model)))

    model = action_qwen.SharedModel("base", "action", 17, 31, "diagnosis")

    assert model.call(True)("action", []) == ("action", False, 17)
    assert model.call(False, final=True)("diagnosis", []) == ("diagnosis", False, 31)
    assert model.call(False)("base", []) == ("diagnosis", True, 17)
    assert peft_model.loaded == [("diagnosis", "diagnosis", True)]

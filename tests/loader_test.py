from __future__ import annotations

import torch

from toto_interp import load_toto_with_fallback
from toto_interp.loader import resolve_device
from toto.model.toto import Toto

from .test_helpers import make_tiny_toto


def test_load_toto_with_fallback_disables_memory_efficient_attention(monkeypatch):
    sentinel = object()
    calls: list[dict[str, object]] = []

    def fake_from_pretrained(model_id, **kwargs):
        calls.append({"model_id": model_id, **kwargs})
        if len(calls) == 1:
            raise AssertionError("use_memory_efficient_attention kernel unavailable")
        return sentinel

    monkeypatch.setattr(Toto, "from_pretrained", fake_from_pretrained)

    loaded = load_toto_with_fallback("fake-model-id", map_location="cpu", strict=False)

    assert loaded is sentinel
    assert len(calls) == 2
    assert calls[1]["use_memory_efficient_attention"] is False


def test_resolve_device_respects_explicit_choice():
    assert resolve_device("cpu") == "cpu"


def test_load_toto_with_fallback_supports_random_init(monkeypatch):
    model = make_tiny_toto()
    original = next(model.model.parameters()).detach().clone()

    monkeypatch.setattr(Toto, "from_pretrained", lambda *args, **kwargs: model)

    loaded = load_toto_with_fallback("fake-model-id", weight_source="random_init", map_location="cpu")

    randomized = next(loaded.model.parameters()).detach()
    assert not torch.allclose(original, randomized)


def test_load_toto_with_fallback_supports_checkpoint_loading(monkeypatch, tmp_path):
    source = make_tiny_toto()
    target = make_tiny_toto()
    for param in source.model.parameters():
        torch.nn.init.constant_(param, 0.25)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(source.state_dict(), checkpoint_path)

    monkeypatch.setattr(Toto, "from_pretrained", lambda *args, **kwargs: target)

    loaded = load_toto_with_fallback(
        "fake-model-id",
        weight_source="checkpoint",
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )

    source_param = next(source.parameters()).detach()
    loaded_param = next(loaded.parameters()).detach()
    assert torch.allclose(source_param, loaded_param)

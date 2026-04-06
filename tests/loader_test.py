from __future__ import annotations

from toto_interp import load_toto_with_fallback
from toto.model.toto import Toto


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

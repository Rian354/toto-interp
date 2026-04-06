from __future__ import annotations

from typing import Any

import torch

from toto.model.toto import Toto


def resolve_device(device: str | None = None) -> str:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def load_toto_with_fallback(
    model_id: str = "Datadog/Toto-Open-Base-1.0",
    *,
    map_location: str = "cpu",
    device: str | None = None,
    strict: bool = False,
    **model_overrides: Any,
) -> Toto:
    """
    Load Toto with an explicit no-xFormers fallback.

    Toto's checkpoint loader already disables memory-efficient attention when
    xFormers is unavailable, but this helper keeps the fallback behavior
    centralized for the interpretability scripts and tests.
    """

    resolved_device = resolve_device(device)
    load_location = map_location
    if resolved_device != "cpu" and map_location == "cpu":
        load_location = "cpu"

    try:
        model = Toto.from_pretrained(model_id, map_location=load_location, strict=strict, **model_overrides)
    except AssertionError as exc:
        if "use_memory_efficient_attention" not in str(exc):
            raise
        safe_overrides = dict(model_overrides)
        safe_overrides["use_memory_efficient_attention"] = False
        model = Toto.from_pretrained(model_id, map_location=load_location, strict=strict, **safe_overrides)

    if resolved_device != "cpu" and hasattr(model, "to"):
        model.to(resolved_device)
    return model

from __future__ import annotations

from typing import Any

from toto.model.toto import Toto


def load_toto_with_fallback(
    model_id: str = "Datadog/Toto-Open-Base-1.0",
    *,
    map_location: str = "cpu",
    strict: bool = False,
    **model_overrides: Any,
) -> Toto:
    """
    Load Toto with an explicit no-xFormers fallback.

    Toto's checkpoint loader already disables memory-efficient attention when
    xFormers is unavailable, but this helper keeps the fallback behavior
    centralized for the interpretability scripts and tests.
    """

    try:
        return Toto.from_pretrained(model_id, map_location=map_location, strict=strict, **model_overrides)
    except AssertionError as exc:
        if "use_memory_efficient_attention" not in str(exc):
            raise
        safe_overrides = dict(model_overrides)
        safe_overrides["use_memory_efficient_attention"] = False
        return Toto.from_pretrained(model_id, map_location=map_location, strict=strict, **safe_overrides)

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from toto.model.backbone import TotoBackbone
from toto.model.toto import Toto
from .types import InterventionConfig


def _resolve_backbone(model: Toto | TotoBackbone) -> TotoBackbone:
    if isinstance(model, TotoBackbone):
        return model
    if isinstance(model, Toto):
        return model.model
    raise TypeError(f"Unsupported model type for interpretability intervention: {type(model)!r}")


def _token_selector(
    seq_len: int,
    *,
    token_position: str,
    decode_step: int,
    decode_steps: tuple[int, ...] | None,
) -> slice | list[int] | None:
    if token_position == "all_context":
        return slice(None) if seq_len > 1 else None
    if token_position == "final_context":
        return [seq_len - 1] if seq_len > 1 else None
    if token_position == "first_decode":
        if seq_len != 1:
            return None
        if decode_steps is None or decode_step in decode_steps:
            return [0]
        return None
    raise ValueError(f"Unsupported token position: {token_position}")


def _ablate_direction(
    activations: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    denom = vector.pow(2).sum().clamp_min(1e-12)
    projection = (activations * vector).sum(dim=-1, keepdim=True) / denom
    return activations - projection * vector


def _steer_direction(
    activations: torch.Tensor,
    vector: torch.Tensor,
    *,
    strength: float,
    normalize_by_residual: bool,
) -> torch.Tensor:
    if normalize_by_residual:
        unit_vector = vector / vector.norm().clamp_min(1e-12)
        residual_norm = activations.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta = strength * residual_norm * unit_vector
    else:
        delta = strength * vector
    return activations + delta


@contextmanager
def apply_intervention(
    model: Toto | TotoBackbone,
    intervention_config: InterventionConfig,
) -> Iterator[None]:
    """
    Apply a residual-stream intervention to Toto transformer layer outputs.

    The context manager is intentionally narrow: it only edits the requested
    transformer-layer outputs and requested token positions, leaving Toto's
    default forecasting behavior unchanged outside the scope of the context.
    """

    backbone = _resolve_backbone(model)
    transformer_layers = backbone.transformer.layers
    if not transformer_layers:
        yield
        return

    state = {
        "decode_step": 0,
        "seq_len": None,
    }
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def pre_hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        hidden_states = inputs[1]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("Expected transformer layer hidden states as the second positional input.")
        seq_len = int(hidden_states.shape[2])
        state["seq_len"] = seq_len
        if seq_len == 1:
            state["decode_step"] += 1

    handles.append(transformer_layers[0].register_forward_pre_hook(pre_hook))

    def make_layer_hook(layer_idx: int):
        def layer_hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            if layer_idx not in intervention_config.layer_indices:
                return output

            seq_len = state["seq_len"]
            if seq_len is None:
                return output

            selected_tokens = _token_selector(
                seq_len,
                token_position=intervention_config.token_position,
                decode_step=int(state["decode_step"]),
                decode_steps=intervention_config.decode_steps,
            )
            if selected_tokens is None:
                return output

            modified = output.clone()
            vector = intervention_config.vector.to(device=output.device, dtype=output.dtype).view(1, 1, 1, -1)

            if isinstance(selected_tokens, slice):
                selected = modified[:, :, selected_tokens, :]
            else:
                selected = modified[:, :, selected_tokens, :]

            if intervention_config.mode == "ablate":
                updated = _ablate_direction(selected, vector)
            elif intervention_config.mode == "steer":
                updated = _steer_direction(
                    selected,
                    vector,
                    strength=intervention_config.strength,
                    normalize_by_residual=intervention_config.normalize_by_residual,
                )
            else:
                raise ValueError(f"Unsupported intervention mode: {intervention_config.mode}")

            if isinstance(selected_tokens, slice):
                modified[:, :, selected_tokens, :] = updated
            else:
                modified[:, :, selected_tokens, :] = updated
            return modified

        return layer_hook

    for layer_idx, layer in enumerate(transformer_layers):
        if layer_idx in intervention_config.layer_indices:
            handles.append(layer.register_forward_hook(make_layer_hook(layer_idx)))

    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()

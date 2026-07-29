"""Policy-score extraction in a small, model-library-independent core.

The score of a prompt/response sequence is the gradient of the response
log-likelihood with respect to the policy tangent parameters.  This module
deliberately knows nothing about Transformers or PEFT; the only PEFT-specific
contract is the fail-closed selection of trainable ``lora_B`` parameters.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

NamedParameter: TypeAlias = tuple[str, torch.Tensor]


def sequence_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return each sequence's response-only causal log-likelihood.

    ``logits[:, t]`` scores ``input_ids[:, t + 1]``.  Consequently the first
    mask position has no associated conditional probability and is ignored.
    Half-precision logits are normalized in float32 to avoid unnecessary
    underflow; float32 and float64 inputs retain their dtype.
    """

    if not all(isinstance(value, torch.Tensor) for value in (logits, input_ids, response_mask)):
        raise TypeError("logits, input_ids, and response_mask must be torch.Tensor objects")
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence_length, vocabulary_size)")
    if input_ids.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("input_ids and response_mask must have shape (batch, sequence_length)")

    batch_size, sequence_length, vocabulary_size = logits.shape
    if batch_size < 1 or sequence_length < 1 or vocabulary_size < 1:
        raise ValueError("all logits dimensions must be positive")
    expected_shape = (batch_size, sequence_length)
    if input_ids.shape != expected_shape or response_mask.shape != expected_shape:
        raise ValueError("input_ids and response_mask must match the first two logits dimensions")
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    if input_ids.is_floating_point() or input_ids.is_complex() or input_ids.dtype == torch.bool:
        raise TypeError("input_ids must have an integer dtype")
    if response_mask.is_complex():
        raise TypeError("response_mask must be boolean, integer, or floating point")
    if logits.device != input_ids.device or logits.device != response_mask.device:
        raise ValueError("logits, input_ids, and response_mask must be on the same device")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if response_mask.is_floating_point() and not bool(torch.isfinite(response_mask).all()):
        raise ValueError("response_mask must be finite")
    if not bool(((response_mask == 0) | (response_mask == 1)).all()):
        raise ValueError("response_mask must be binary (zero or one)")
    if bool((input_ids < 0).any()) or bool((input_ids >= vocabulary_size).any()):
        raise ValueError("input_ids entries must be valid vocabulary indices")

    shifted_logits = logits[:, :-1, :]
    if logits.dtype in (torch.float16, torch.bfloat16):
        shifted_logits = shifted_logits.float()
    shifted_ids = input_ids[:, 1:].to(dtype=torch.long)
    token_log_probs = (
        shifted_logits.log_softmax(dim=-1)
        .gather(
            dim=-1,
            index=shifted_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    shifted_response_mask = response_mask[:, 1:].to(dtype=torch.bool)
    return token_log_probs.masked_fill(~shifted_response_mask, 0.0).sum(dim=-1)


def select_named_tangent_parameters(model: torch.nn.Module) -> tuple[NamedParameter, ...]:
    """Select trainable ``lora_B`` parameters, failing closed on misconfiguration.

    The returned tuple preserves the original order of ``model.named_parameters()``.
    Any other trainable parameter is an error rather than something silently
    omitted from the Fisher geometry.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    selected: list[NamedParameter] = []
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_B" not in name:
            unexpected.append(name)
        else:
            selected.append((name, parameter))

    if unexpected:
        joined = ", ".join(unexpected)
        raise ValueError(f"all trainable parameters must be lora_B parameters; found: {joined}")
    if not selected:
        raise ValueError("no trainable lora_B parameters were found")
    return tuple(selected)


@dataclass(frozen=True)
class ParameterLayoutEntry:
    """Serializable location of one parameter in a flattened tangent vector."""

    name: str
    shape: tuple[int, ...]
    offset: int
    numel: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a parameter layout name must be a non-empty string")
        if not isinstance(self.shape, tuple) or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in self.shape
        ):
            raise ValueError("a parameter layout shape must be a tuple of non-negative integers")
        if not isinstance(self.offset, int) or isinstance(self.offset, bool) or self.offset < 0:
            raise ValueError("a parameter layout offset must be a non-negative integer")
        if not isinstance(self.numel, int) or isinstance(self.numel, bool) or self.numel < 1:
            raise ValueError("a parameter layout numel must be a positive integer")
        if math.prod(self.shape) != self.numel:
            raise ValueError("parameter layout numel does not match shape")

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable metadata for this entry."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "offset": self.offset,
            "numel": self.numel,
        }


@dataclass(frozen=True)
class ParameterLayout:
    """Ordered mapping from named parameter gradients to a flat score vector."""

    entries: tuple[ParameterLayoutEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple):
            object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries:
            raise ValueError("a parameter layout must contain at least one entry")

        expected_offset = 0
        seen_names: set[str] = set()
        for entry in self.entries:
            if not isinstance(entry, ParameterLayoutEntry):
                raise TypeError("parameter layout entries must be ParameterLayoutEntry objects")
            if entry.name in seen_names:
                raise ValueError(f"duplicate parameter layout name: {entry.name}")
            if entry.offset != expected_offset:
                raise ValueError("parameter layout offsets must be contiguous and ordered")
            expected_offset += entry.numel
            seen_names.add(entry.name)

    @classmethod
    def from_named_parameters(cls, named_parameters: Iterable[NamedParameter]) -> ParameterLayout:
        """Build a layout in precisely the provided iteration order."""

        entries: list[ParameterLayoutEntry] = []
        seen_names: set[str] = set()
        seen_parameters: set[int] = set()
        offset = 0
        for item in named_parameters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("named_parameters must contain (name, parameter) tuples")
            name, parameter = item
            if not isinstance(name, str) or not name:
                raise TypeError("parameter names must be non-empty strings")
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(f"parameter {name!r} must be a torch.Tensor")
            if name in seen_names:
                raise ValueError(f"duplicate parameter name: {name}")
            if id(parameter) in seen_parameters:
                raise ValueError(f"parameter {name!r} occurs more than once")
            numel = parameter.numel()
            if numel < 1:
                raise ValueError(f"parameter {name!r} must be non-empty")
            entry = ParameterLayoutEntry(name, tuple(parameter.shape), offset, numel)
            entries.append(entry)
            offset += numel
            seen_names.add(name)
            seen_parameters.add(id(parameter))
        return cls(tuple(entries))

    @classmethod
    def from_metadata(cls, metadata: Iterable[Mapping[str, Any]]) -> ParameterLayout:
        """Reconstruct a layout from :meth:`to_metadata` output."""

        entries: list[ParameterLayoutEntry] = []
        for item in metadata:
            if not isinstance(item, Mapping):
                raise TypeError("each metadata entry must be a mapping")
            try:
                name = item["name"]
                raw_shape = item["shape"]
                offset = item["offset"]
                numel = item["numel"]
            except KeyError as error:
                raise ValueError(f"parameter metadata is missing {error.args[0]!r}") from error
            if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
                raise TypeError("parameter metadata shape must be a sequence")
            entries.append(
                ParameterLayoutEntry(
                    name=name,
                    shape=tuple(raw_shape),
                    offset=offset,
                    numel=numel,
                )
            )
        return cls(tuple(entries))

    @property
    def dimension(self) -> int:
        """Total number of scalar tangent parameters."""

        return self.entries[-1].offset + self.entries[-1].numel

    @property
    def total_numel(self) -> int:
        """Alias for :attr:`dimension`."""

        return self.dimension

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(entry.shape for entry in self.entries)

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(entry.offset for entry in self.entries)

    @property
    def numels(self) -> tuple[int, ...]:
        return tuple(entry.numel for entry in self.entries)

    def to_metadata(self) -> list[dict[str, Any]]:
        """Return JSON-serializable ordered layout metadata."""

        return [entry.to_metadata() for entry in self.entries]

    def validate_named_parameters(self, named_parameters: Iterable[NamedParameter]) -> None:
        """Require names and shapes to match this layout in exact order."""

        values = tuple(named_parameters)
        if len(values) != len(self.entries):
            raise ValueError("named parameter count does not match the parameter layout")
        for entry, item in zip(self.entries, values, strict=True):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("named_parameters must contain (name, parameter) tuples")
            name, parameter = item
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(f"parameter {name!r} must be a torch.Tensor")
            if name != entry.name or tuple(parameter.shape) != entry.shape:
                raise ValueError("named parameter order or shape does not match the layout")

    def flatten(
        self,
        gradients: Sequence[torch.Tensor],
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Flatten one sample's or a batch's gradients in layout order.

        A one-sample input contains tensors with each parameter's exact shape
        and returns ``[dimension]``.  A batched input contains tensors shaped
        ``[batch, *parameter_shape]`` and returns ``[batch, dimension]``.
        """

        if not isinstance(gradients, Sequence) or isinstance(gradients, (str, bytes)):
            raise TypeError("gradients must be a sequence of tensors")
        if len(gradients) != len(self.entries):
            raise ValueError("gradient count does not match the parameter layout")
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("flattened gradients must have a floating-point dtype")

        flattened: list[torch.Tensor] = []
        batch_size: int | None = None
        batched: bool | None = None
        device: torch.device | None = None
        inferred_dtype: torch.dtype | None = None
        for entry, gradient in zip(self.entries, gradients, strict=True):
            if not isinstance(gradient, torch.Tensor):
                raise TypeError(f"gradient for {entry.name!r} must be a torch.Tensor")
            if not gradient.is_floating_point():
                raise TypeError(f"gradient for {entry.name!r} must be floating point")
            is_single = tuple(gradient.shape) == entry.shape
            is_batched = (
                gradient.ndim == len(entry.shape) + 1 and tuple(gradient.shape[1:]) == entry.shape
            )
            if not is_single and not is_batched:
                raise ValueError(f"gradient for {entry.name!r} has the wrong shape")
            current_batched = not is_single
            if batched is None:
                batched = current_batched
                if current_batched:
                    batch_size = gradient.shape[0]
                    if batch_size < 1:
                        raise ValueError("the per-sample gradient batch must be non-empty")
            elif current_batched != batched:
                raise ValueError("cannot mix single-sample and batched gradients")
            elif current_batched and gradient.shape[0] != batch_size:
                raise ValueError("all per-sample gradients must have the same batch size")

            if device is None:
                device = gradient.device
            elif gradient.device != device:
                raise ValueError("all gradients must be on the same device")
            if dtype is None:
                if inferred_dtype is None:
                    inferred_dtype = gradient.dtype
                elif gradient.dtype != inferred_dtype:
                    raise ValueError("gradient dtypes differ; pass dtype to normalize them")

            output_dtype = dtype if dtype is not None else gradient.dtype
            if current_batched:
                flattened.append(gradient.reshape(gradient.shape[0], entry.numel).to(output_dtype))
            else:
                flattened.append(gradient.reshape(entry.numel).to(output_dtype))

        return torch.cat(flattened, dim=1 if batched else 0)

    def flatten_per_sample_gradients(
        self,
        gradients: Sequence[torch.Tensor],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Flatten batched parameter gradients, requiring a ``[batch, d]`` result."""

        result = self.flatten(gradients, dtype=dtype)
        if result.ndim != 2:
            raise ValueError("expected gradients with a leading per-sample batch dimension")
        return result


def _validate_named_tangent_parameters(
    named_parameters: Iterable[NamedParameter],
) -> tuple[NamedParameter, ...]:
    values = tuple(named_parameters)
    if not values:
        raise ValueError("no trainable lora_B parameters were provided")
    seen_names: set[str] = set()
    seen_parameters: set[int] = set()
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("named_parameters must contain (name, parameter) tuples")
        name, parameter = item
        if not isinstance(name, str) or not name:
            raise TypeError("parameter names must be non-empty strings")
        if not isinstance(parameter, torch.Tensor):
            raise TypeError(f"parameter {name!r} must be a torch.Tensor")
        if not parameter.requires_grad:
            raise ValueError(f"tangent parameter {name!r} does not require gradients")
        if "lora_B" not in name:
            raise ValueError(f"non-lora_B tangent parameter was provided: {name}")
        if not parameter.is_floating_point():
            raise TypeError(f"tangent parameter {name!r} must be floating point")
        if name in seen_names or id(parameter) in seen_parameters:
            raise ValueError(f"duplicate tangent parameter: {name}")
        seen_names.add(name)
        seen_parameters.add(id(parameter))
    return values


def per_sample_scores(
    sample_log_probs: torch.Tensor,
    named_parameters: Iterable[NamedParameter] | torch.nn.Module,
    *,
    layout: ParameterLayout | None = None,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Differentiate every sample log-likelihood into a float32 score matrix.

    A straightforward sample loop is intentional: it is robust across PyTorch
    versions and does not depend on ``is_grads_batched`` behavior.  Intermediate
    samples retain the shared forward graph; ``retain_graph`` controls whether
    the graph remains after the final sample.  Returned scores never retain an
    autograd graph.
    """

    if not isinstance(sample_log_probs, torch.Tensor):
        raise TypeError("sample_log_probs must be a torch.Tensor")
    if sample_log_probs.ndim != 1 or sample_log_probs.numel() < 1:
        raise ValueError("sample_log_probs must have non-empty shape (batch,)")
    if not sample_log_probs.is_floating_point():
        raise TypeError("sample_log_probs must have a floating-point dtype")
    if not bool(torch.isfinite(sample_log_probs).all()):
        raise ValueError("sample_log_probs must be finite")
    if not sample_log_probs.requires_grad:
        raise ValueError("sample_log_probs must require gradients")
    if not isinstance(retain_graph, bool):
        raise TypeError("retain_graph must be a bool")

    if isinstance(named_parameters, torch.nn.Module):
        values = select_named_tangent_parameters(named_parameters)
    else:
        values = _validate_named_tangent_parameters(named_parameters)
    # Apply the same checks to the model-selected route without changing order.
    values = _validate_named_tangent_parameters(values)
    parameters = tuple(parameter for _, parameter in values)

    if layout is None:
        layout = ParameterLayout.from_named_parameters(values)
    elif not isinstance(layout, ParameterLayout):
        raise TypeError("layout must be a ParameterLayout")
    else:
        layout.validate_named_parameters(values)

    rows: list[torch.Tensor] = []
    batch_size = sample_log_probs.shape[0]
    for sample_index in range(batch_size):
        gradients = torch.autograd.grad(
            sample_log_probs[sample_index],
            parameters,
            create_graph=False,
            retain_graph=retain_graph or sample_index + 1 < batch_size,
            allow_unused=False,
        )
        rows.append(layout.flatten(gradients, dtype=torch.float32))

    scores = torch.stack(rows, dim=0).detach()
    if scores.shape != (batch_size, layout.dimension):
        raise RuntimeError("internal error while flattening per-sample scores")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("per-sample policy scores must be finite")
    return scores


__all__ = [
    "NamedParameter",
    "ParameterLayout",
    "ParameterLayoutEntry",
    "per_sample_scores",
    "select_named_tangent_parameters",
    "sequence_log_probs",
]

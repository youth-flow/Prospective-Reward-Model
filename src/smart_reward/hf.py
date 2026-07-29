"""Hugging Face integration for the exact-delta ProRM experiment.

The numerical core does not depend on Transformers or PEFT.  This module keeps
that property: model and tokenizer objects are consumed through their public
PyTorch interfaces, and PEFT is imported only inside
:func:`configure_fixed_a_lora`.

Three contracts are intentionally strict here:

* policy candidates are sampled from the unwarped temperature-one categorical
  distribution (with EOS as the only data-dependent stopping event);
* the exact generated token IDs, including the first EOS, are reused for policy
  scoring; and
* the policy tangent contains only initially-zero LoRA-B matrices, while the
  once-initialized LoRA-A matrices are frozen and fingerprinted.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

import torch

from .scores import ParameterLayout, sequence_log_probs

_CORE_SAMPLING_DEFAULTS: dict[str, object] = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_new_tokens": 0,
    "repetition_penalty": 1.0,
}

# These explicit neutral values override potentially non-neutral values inherited
# from ``model.generation_config``.  They are part of the distribution contract,
# not optional tuning knobs.
_NEUTRAL_GENERATION_DEFAULTS: dict[str, object] = {
    "min_length": 0,
    "max_time": None,
    "stop_strings": None,
    "early_stopping": False,
    "num_beams": 1,
    "num_beam_groups": 1,
    "penalty_alpha": None,
    "dola_layers": None,
    "min_p": None,
    "typical_p": 1.0,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
    "diversity_penalty": 0.0,
    "encoder_repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "bad_words_ids": None,
    "force_words_ids": None,
    "constraints": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "renormalize_logits": False,
    "remove_invalid_values": False,
    "exponential_decay_length_penalty": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "forced_decoder_ids": None,
    "sequence_bias": None,
    "token_healing": False,
    "guidance_scale": None,
    "watermarking_config": None,
}

EXACT_SAMPLING_DEFAULTS: Mapping[str, object] = MappingProxyType(
    {**_CORE_SAMPLING_DEFAULTS, **_NEUTRAL_GENERATION_DEFAULTS}
)

_FORBIDDEN_GENERATION_KEYS = frozenset(
    {
        "generation_config",
        "logits_processor",
        "logits_warper",
        "prefix_allowed_tokens_fn",
        "stopping_criteria",
        "assistant_model",
        "negative_prompt_ids",
        "negative_prompt_attention_mask",
    }
)
_OPERATIONAL_GENERATION_KEYS = frozenset(
    {
        "max_new_tokens",
        "max_length",
        "num_return_sequences",
        "eos_token_id",
        "pad_token_id",
        "bos_token_id",
        "use_cache",
        "return_dict_in_generate",
        "output_scores",
        "output_logits",
        "synced_gpus",
        "low_memory",
        "cache_implementation",
        "cache_config",
    }
)


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _require_exact_value(name: str, value: object, expected: object) -> None:
    if expected is None:
        if value is not None:
            raise ValueError(f"{name} must be None for exact policy sampling")
        return
    if isinstance(expected, bool):
        valid = isinstance(value, bool) and value is expected
    elif isinstance(expected, int):
        valid = (
            isinstance(value, Integral) and not isinstance(value, bool) and int(value) == expected
        )
    elif isinstance(expected, float):
        valid = _is_real(value) and math.isfinite(float(value)) and float(value) == expected
    else:  # pragma: no cover - all current defaults are covered above
        valid = value == expected
    if not valid:
        raise ValueError(f"{name} must equal {expected!r} for exact policy sampling")


def validate_exact_generation_kwargs(
    generation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a canonical, explicit configuration for exact policy sampling.

    The result overrides all common logits warpers/processors with their neutral
    values, so a non-default ``GenerationConfig`` cannot silently change the
    candidate distribution.  Unknown keys fail closed because a future or
    model-specific generation option may alter that distribution.

    A finite response cap is still required by :func:`generate_exact_candidates`;
    it is deliberately not supplied here because it belongs to the experiment
    configuration rather than the token-level categorical distribution.
    """

    if generation_kwargs is None:
        supplied: dict[str, object] = {}
    elif not isinstance(generation_kwargs, Mapping):
        raise TypeError("generation_kwargs must be a mapping or None")
    else:
        supplied = dict(generation_kwargs)

    forbidden = sorted(_FORBIDDEN_GENERATION_KEYS.intersection(supplied))
    if forbidden:
        raise ValueError(
            "custom generation control is forbidden for exact policy sampling: "
            + ", ".join(forbidden)
        )

    allowed = set(EXACT_SAMPLING_DEFAULTS).union(_OPERATIONAL_GENERATION_KEYS)
    unknown = sorted(set(supplied).difference(allowed))
    if unknown:
        raise ValueError("unrecognized generation kwargs fail closed: " + ", ".join(unknown))

    canonical = dict(EXACT_SAMPLING_DEFAULTS)
    canonical.update(supplied)
    for name, expected in EXACT_SAMPLING_DEFAULTS.items():
        _require_exact_value(name, canonical[name], expected)

    for name in ("max_new_tokens", "max_length", "num_return_sequences"):
        if name in canonical:
            value = canonical[name]
            if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            canonical[name] = int(value)
    if "max_new_tokens" in supplied and "max_length" in supplied:
        raise ValueError("set max_new_tokens or max_length, not both")

    for name in (
        "do_sample",
        "use_cache",
        "return_dict_in_generate",
        "output_scores",
        "output_logits",
        "synced_gpus",
        "low_memory",
    ):
        if name in supplied and not isinstance(supplied[name], bool):
            raise TypeError(f"{name} must be a bool")
    if "eos_token_id" in supplied:
        _normalize_eos_ids(supplied["eos_token_id"])
    for name in ("pad_token_id", "bos_token_id"):
        if name in supplied and supplied[name] is not None:
            value = supplied[name]
            if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer or None")

    return canonical


def _normalize_eos_ids(eos_token_id: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(eos_token_id, Integral) and not isinstance(eos_token_id, bool):
        values = (int(eos_token_id),)
    elif isinstance(eos_token_id, Sequence) and not isinstance(
        eos_token_id, (str, bytes, bytearray)
    ):
        values = tuple(eos_token_id)
    else:
        raise TypeError("eos_token_id must be an integer or a non-empty sequence of integers")
    if not values:
        raise ValueError("eos_token_id must not be empty")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0
        for value in values
    ):
        raise ValueError("eos_token_id values must be non-negative integers")
    normalized = tuple(int(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("eos_token_id values must be unique")
    return normalized


def _validate_token_matrix(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must have non-empty shape (batch, sequence_length)")
    if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
        raise TypeError(f"{name} must have an integer dtype")
    if bool((value < 0).any()):
        raise ValueError(f"{name} must contain non-negative token IDs")


@dataclass(frozen=True, slots=True)
class ExactTokenCandidates:
    """Generated candidates with the exact tensors used for policy scoring."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    terminated_by_eos: torch.Tensor
    reached_max_length: torch.Tensor
    prompt_width: int
    source_model_id: int | None = None
    source_trainable_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_token_matrix(self.input_ids, name="input_ids")
        expected_shape = self.input_ids.shape
        for name, value in (
            ("attention_mask", self.attention_mask),
            ("response_mask", self.response_mask),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != expected_shape:
                raise ValueError(f"{name} must match input_ids shape")
            if value.device != self.input_ids.device:
                raise ValueError(f"{name} must be on the input_ids device")
            if not bool(((value == 0) | (value == 1)).all()):
                raise ValueError(f"{name} must be binary")
        for name, value in (
            ("terminated_by_eos", self.terminated_by_eos),
            ("reached_max_length", self.reached_max_length),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != (expected_shape[0],):
                raise ValueError(f"{name} must have shape (batch,)")
            if value.dtype != torch.bool or value.device != self.input_ids.device:
                raise ValueError(f"{name} must be boolean and on the input_ids device")
        if bool((self.terminated_by_eos & self.reached_max_length).any()):
            raise ValueError("a candidate cannot terminate by EOS and reach the length limit")
        if not isinstance(self.prompt_width, int) or isinstance(self.prompt_width, bool):
            raise TypeError("prompt_width must be an integer")
        if not 0 < self.prompt_width < expected_shape[1]:
            raise ValueError("prompt_width must leave at least one generated token")
        if bool(self.response_mask[:, : self.prompt_width].any()):
            raise ValueError("response_mask must exclude every prompt/padding position")
        if bool((self.response_mask.sum(dim=1) < 1).any()):
            raise ValueError("every candidate must contain at least one response token")
        if self.source_model_id is not None and (
            not isinstance(self.source_model_id, int) or isinstance(self.source_model_id, bool)
        ):
            raise TypeError("source_model_id must be an integer or None")
        if self.source_trainable_sha256 is not None and (
            not isinstance(self.source_trainable_sha256, str)
            or len(self.source_trainable_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.source_trainable_sha256
            )
        ):
            raise ValueError("source_trainable_sha256 must be a lowercase SHA256 digest or None")


# A descriptive alias for callers that think in batches rather than candidates.
CandidateTokenBatch = ExactTokenCandidates


def build_exact_token_candidates(
    prompt_input_ids: torch.Tensor,
    generated: torch.Tensor | object,
    *,
    eos_token_id: int | Sequence[int],
    prompt_attention_mask: torch.Tensor | None = None,
    pad_token_id: int | None = None,
    max_new_tokens: int | None = None,
    max_length: int | None = None,
) -> ExactTokenCandidates:
    """Build response masks directly over generated IDs without retokenization.

    ``generated`` may be a tensor or a Transformers generation output exposing
    ``.sequences``.  Its prompt prefix must exactly equal ``prompt_input_ids``
    (repeated contiguously for ``num_return_sequences``).  The first EOS is
    included in the response mask; any following batch padding is excluded.
    """

    _validate_token_matrix(prompt_input_ids, name="prompt_input_ids")
    sequences = (
        generated if isinstance(generated, torch.Tensor) else getattr(generated, "sequences", None)
    )
    _validate_token_matrix(sequences, name="generated sequences")
    if sequences.device != prompt_input_ids.device:
        raise ValueError("generated sequences and prompt_input_ids must be on the same device")
    if max_new_tokens is not None and max_length is not None:
        raise ValueError("set max_new_tokens or max_length, not both")
    for name, value in (("max_new_tokens", max_new_tokens), ("max_length", max_length)):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None")
    if pad_token_id is not None and (
        not isinstance(pad_token_id, Integral)
        or isinstance(pad_token_id, bool)
        or int(pad_token_id) < 0
    ):
        raise ValueError("pad_token_id must be a non-negative integer or None")

    batch_size, prompt_width = prompt_input_ids.shape
    candidate_count, sequence_width = sequences.shape
    if sequence_width <= prompt_width:
        raise ValueError("generated sequences must contain at least one response token")
    if candidate_count % batch_size:
        raise ValueError("generated candidate count must be a multiple of prompt batch size")
    repeats = candidate_count // batch_size
    expected_prefix = prompt_input_ids.repeat_interleave(repeats, dim=0)
    if not torch.equal(sequences[:, :prompt_width], expected_prefix):
        raise ValueError("generated sequences do not preserve the exact prompt token prefix")
    response_capacity = sequence_width - prompt_width
    if max_new_tokens is not None and response_capacity > max_new_tokens:
        raise ValueError("generated sequences exceed max_new_tokens")
    if max_length is not None:
        if prompt_width >= max_length:
            raise ValueError("max_length must exceed the padded prompt width")
        if sequence_width > max_length:
            raise ValueError("generated sequences exceed max_length")

    if prompt_attention_mask is None:
        repeated_prompt_attention = torch.ones_like(expected_prefix, dtype=torch.bool)
    else:
        if not isinstance(prompt_attention_mask, torch.Tensor):
            raise TypeError("prompt_attention_mask must be a torch.Tensor or None")
        if prompt_attention_mask.shape != prompt_input_ids.shape:
            raise ValueError("prompt_attention_mask must match prompt_input_ids shape")
        if prompt_attention_mask.device != prompt_input_ids.device:
            raise ValueError("prompt_attention_mask must be on the prompt_input_ids device")
        if not bool(((prompt_attention_mask == 0) | (prompt_attention_mask == 1)).all()):
            raise ValueError("prompt_attention_mask must be binary")
        repeated_prompt_attention = prompt_attention_mask.to(torch.bool).repeat_interleave(
            repeats, dim=0
        )

    eos_ids = _normalize_eos_ids(eos_token_id)
    response_ids = sequences[:, prompt_width:]
    eos_hits = torch.zeros_like(response_ids, dtype=torch.bool)
    for eos_id in eos_ids:
        eos_hits |= response_ids == eos_id

    positions = torch.arange(response_capacity, device=sequences.device).expand_as(response_ids)
    eos_sentinel = torch.full_like(positions, response_capacity)
    first_eos = torch.where(eos_hits, positions, eos_sentinel).amin(dim=1)
    terminated = first_eos < response_capacity
    response_mask = positions <= first_eos.unsqueeze(1)

    # Once EOS appears, every later position must be generation padding.  This
    # guard detects malformed/reconstructed token sequences before score use.
    after_eos = positions > first_eos.unsqueeze(1)
    if bool(after_eos.any()):
        if pad_token_id is None:
            raise ValueError(
                "pad_token_id is required when generated rows contain post-EOS padding"
            )
        if bool((response_ids[after_eos] != int(pad_token_id)).any()):
            raise ValueError("non-padding token found after the first EOS")

    limit = max_new_tokens
    if limit is None and max_length is not None:
        limit = max_length - prompt_width
    if limit is None:
        reached_limit = torch.zeros_like(terminated)
    else:
        reached_limit = (~terminated) & (torch.full_like(first_eos, response_capacity) >= limit)

    attention_mask = torch.cat((repeated_prompt_attention, response_mask), dim=1)
    return ExactTokenCandidates(
        input_ids=sequences,
        attention_mask=attention_mask,
        response_mask=torch.cat(
            (torch.zeros_like(expected_prefix, dtype=torch.bool), response_mask), dim=1
        ),
        terminated_by_eos=terminated,
        reached_max_length=reached_limit,
        prompt_width=prompt_width,
    )


def generate_exact_candidates(
    model: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    *,
    prompt_attention_mask: torch.Tensor | None = None,
    generation_kwargs: Mapping[str, object] | None = None,
) -> ExactTokenCandidates:
    """Sample and package candidates under the exact-distribution contract."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if model.training:
        raise ValueError("model must be in eval mode for candidate sampling and scoring")
    canonical = validate_exact_generation_kwargs(generation_kwargs)
    if "max_new_tokens" not in canonical and "max_length" not in canonical:
        raise ValueError("exact generation requires a finite max_new_tokens or max_length")

    generate_inputs: dict[str, object] = {"input_ids": prompt_input_ids, **canonical}
    if prompt_attention_mask is not None:
        generate_inputs["attention_mask"] = prompt_attention_mask
    with torch.inference_mode():
        generated = model.generate(**generate_inputs)
    generated_sequences = (
        generated if isinstance(generated, torch.Tensor) else getattr(generated, "sequences", None)
    )
    if not isinstance(generated_sequences, torch.Tensor):
        raise TypeError("model.generate must return a tensor or expose a sequences tensor")
    # Transformers creates the sampled token IDs inside inference_mode.  Such
    # tensors cannot later be saved by autograd operations (for example,
    # gather in exact log-probability scoring).  A clone made after leaving
    # inference_mode keeps every sampled ID unchanged while converting the
    # immutable generation result into a normal tensor that can safely index
    # a differentiable policy forward pass.
    if generated_sequences.is_inference():
        generated_sequences = generated_sequences.clone()

    eos_token_id = canonical.get("eos_token_id")
    if eos_token_id is None:
        generation_config = getattr(model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("eos_token_id must be supplied or defined on model.generation_config")
    pad_token_id = canonical.get("pad_token_id")
    if pad_token_id is None:
        generation_config = getattr(model, "generation_config", None)
        pad_token_id = getattr(generation_config, "pad_token_id", None)

    candidates = build_exact_token_candidates(
        prompt_input_ids,
        generated_sequences,
        eos_token_id=eos_token_id,
        prompt_attention_mask=prompt_attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=canonical.get("max_new_tokens"),
        max_length=canonical.get("max_length"),
    )
    trainable_state = tuple(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    return replace(
        candidates,
        source_model_id=id(model),
        source_trainable_sha256=_fingerprint_named_tensors(trainable_state),
    )


def _extract_model_logits(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output must expose a logits tensor")
    return logits


def score_exact_candidates(
    model: torch.nn.Module,
    candidates: ExactTokenCandidates,
) -> torch.Tensor:
    """Compute response log probabilities from the exact generated token IDs.

    No ``no_grad`` context is used because materialization differentiates these log
    probabilities with respect to LoRA-B.  Candidates produced by
    :func:`generate_exact_candidates` are tied to the same in-memory model
    instance, preventing accidental score extraction under another policy.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(candidates, ExactTokenCandidates):
        raise TypeError("candidates must be ExactTokenCandidates")
    if candidates.source_model_id is not None and candidates.source_model_id != id(model):
        raise ValueError(
            "candidates must be scored by the exact policy instance that generated them"
        )
    if candidates.source_trainable_sha256 is not None:
        trainable_state = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        current_digest = _fingerprint_named_tensors(trainable_state)
        if current_digest != candidates.source_trainable_sha256:
            raise ValueError("policy tangent parameters changed between generation and scoring")
    if model.training:
        raise ValueError("model must remain in eval mode while extracting policy scores")
    output = model(
        input_ids=candidates.input_ids,
        attention_mask=candidates.attention_mask,
        use_cache=False,
    )
    logits = _extract_model_logits(output)
    return sequence_log_probs(logits, candidates.input_ids, candidates.response_mask)


def assert_noop_logits(
    reference_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> float:
    """Assert that attaching the zero-B adapter leaves reference logits unchanged.

    Returns the maximum absolute difference for run logging.
    """

    for name, value in (("reference_logits", reference_logits), ("adapted_logits", adapted_logits)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite floating-point tensor")
    if reference_logits.shape != adapted_logits.shape:
        raise ValueError("reference_logits and adapted_logits must have identical shapes")
    if reference_logits.device != adapted_logits.device:
        raise ValueError("reference_logits and adapted_logits must be on the same device")
    for name, value in (("atol", atol), ("rtol", rtol)):
        if not _is_real(value) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be a finite non-negative real scalar")
    difference = (reference_logits.detach().float() - adapted_logits.detach().float()).abs()
    max_absolute_error = difference.max().item() if difference.numel() else 0.0
    if not torch.allclose(
        reference_logits.detach().float(),
        adapted_logits.detach().float(),
        atol=float(atol),
        rtol=float(rtol),
    ):
        raise AssertionError(
            "zero-B LoRA adapter is not a logits no-op: "
            f"max_abs_error={max_absolute_error:.8g}, atol={float(atol):.8g}, "
            f"rtol={float(rtol):.8g}"
        )
    return max_absolute_error


def _lora_kind(name: str) -> str | None:
    components = name.split(".")
    if "lora_A" in components:
        return "A"
    if "lora_B" in components:
        return "B"
    # Small fake modules and some PEFT versions use a terminal attribute rather
    # than the standard ``.lora_A.default.weight`` nesting.
    if "lora_A" in name:
        return "A"
    if "lora_B" in name:
        return "B"
    return None


def _lora_pair_key(name: str) -> str:
    if "lora_A" in name:
        return name.replace("lora_A", "lora_*", 1)
    if "lora_B" in name:
        return name.replace("lora_B", "lora_*", 1)
    raise ValueError(f"not a standard LoRA-A/B parameter name: {name}")


def _fingerprint_named_tensors(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        stable = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(stable.dtype).encode("ascii")
        digest.update(struct.pack(">I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack(">I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack(">I", stable.ndim))
        for dimension in stable.shape:
            digest.update(struct.pack(">Q", dimension))
        raw_bytes = bytes(stable.reshape(-1).view(torch.uint8).tolist())
        digest.update(struct.pack(">Q", len(raw_bytes)))
        digest.update(raw_bytes)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FixedALoRASetup:
    """Configured policy tangent plus its stable coordinate metadata."""

    model: torch.nn.Module
    layout: ParameterLayout
    a_state_sha256: str
    trainable_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("layout must be a ParameterLayout")
        if self.trainable_names != self.layout.names:
            raise ValueError("trainable_names must exactly match the stable layout")
        if (
            not isinstance(self.a_state_sha256, str)
            or len(self.a_state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.a_state_sha256)
        ):
            raise ValueError("a_state_sha256 must be a lowercase SHA256 digest")

    def named_tangent_parameters(self) -> tuple[tuple[str, torch.Tensor], ...]:
        """Resolve trainable tensors again and verify layout/name stability."""

        by_name = dict(self.model.named_parameters())
        try:
            values = tuple((name, by_name[name]) for name in self.trainable_names)
        except KeyError as error:
            raise RuntimeError(
                f"configured tangent parameter disappeared: {error.args[0]}"
            ) from error
        self.layout.validate_named_parameters(values)
        for name, parameter in values:
            if not parameter.requires_grad or _lora_kind(name) != "B":
                raise RuntimeError("fixed-A LoRA trainability changed after configuration")
        return values


def _require_peft() -> Any:
    try:
        module = importlib.import_module("peft")
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "configure_fixed_a_lora requires the optional dependency 'peft'; "
            "install prospective-reward-model[llm]"
        ) from error
    if not callable(getattr(module, "get_peft_model", None)):
        raise ImportError("installed peft does not expose callable get_peft_model")
    return module


def configure_fixed_a_lora(model: torch.nn.Module, config: object) -> FixedALoRASetup:
    """Attach PEFT LoRA and expose only initially-zero LoRA-B parameters.

    LoRA-A is initialized once by PEFT, frozen, and SHA256 fingerprinted.  All
    base-model and adapter parameters are frozen before the lexicographically
    stable LoRA-B coordinate layout is opened for gradient extraction.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    peft = _require_peft()
    adapted_model = peft.get_peft_model(model, config)
    if not isinstance(adapted_model, torch.nn.Module):
        raise TypeError("peft.get_peft_model must return a torch.nn.Module")

    named_parameters = tuple(adapted_model.named_parameters())
    a_parameters = tuple(
        (name, value) for name, value in named_parameters if _lora_kind(name) == "A"
    )
    b_parameters = tuple(
        (name, value) for name, value in named_parameters if _lora_kind(name) == "B"
    )
    if not a_parameters or not b_parameters:
        raise ValueError("PEFT adapter must expose both lora_A and lora_B parameters")
    a_keys = {_lora_pair_key(name) for name, _ in a_parameters}
    b_keys = {_lora_pair_key(name) for name, _ in b_parameters}
    if a_keys != b_keys:
        missing_a = sorted(b_keys - a_keys)
        missing_b = sorted(a_keys - b_keys)
        raise ValueError(
            "every LoRA target must expose one A/B parameter pair; "
            f"missing_A={missing_a!r}, missing_B={missing_b!r}"
        )
    if any(not parameter.is_floating_point() for _, parameter in (*a_parameters, *b_parameters)):
        raise TypeError("LoRA-A and LoRA-B parameters must be floating point")
    if any(not bool(torch.isfinite(parameter.detach()).all()) for _, parameter in a_parameters):
        raise ValueError("LoRA-A parameters must be finite")
    zero_a = [
        name
        for name, parameter in a_parameters
        if not bool(torch.count_nonzero(parameter.detach()))
    ]
    if zero_a:
        raise ValueError(
            "LoRA-A must be initialized to a non-zero tangent basis: " + ", ".join(zero_a)
        )
    nonzero_b = [
        name for name, parameter in b_parameters if bool(torch.count_nonzero(parameter.detach()))
    ]
    if nonzero_b:
        raise ValueError("LoRA-B must be initialized exactly to zero: " + ", ".join(nonzero_b))

    adapted_model.requires_grad_(False)
    stable_b_parameters = tuple(sorted(b_parameters, key=lambda item: item[0]))
    for _, parameter in stable_b_parameters:
        parameter.requires_grad_(True)

    # Fail closed if PEFT ever introduces an unclassified trainable tensor.
    trainable = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in adapted_model.named_parameters()
                if parameter.requires_grad
            ),
            key=lambda item: item[0],
        )
    )
    if tuple(name for name, _ in trainable) != tuple(name for name, _ in stable_b_parameters):
        raise RuntimeError("only the intended LoRA-B parameters may remain trainable")
    if any(parameter.requires_grad for _, parameter in a_parameters):
        raise RuntimeError("LoRA-A parameters must remain frozen")

    layout = ParameterLayout.from_named_parameters(trainable)
    return FixedALoRASetup(
        model=adapted_model,
        layout=layout,
        a_state_sha256=_fingerprint_named_tensors(a_parameters),
        trainable_names=layout.names,
    )


def pool_final_response_hidden_state(
    hidden_states: torch.Tensor | Sequence[torch.Tensor],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool the final-layer hidden state at each final response token.

    Because the exact response mask includes EOS, normally terminated samples
    pool EOS; length-limited samples pool their final generated token.  Prompt
    and post-EOS padding positions can never be selected.
    """

    if isinstance(hidden_states, Sequence) and not isinstance(hidden_states, torch.Tensor):
        if not hidden_states:
            raise ValueError("hidden_states sequence must not be empty")
        hidden = hidden_states[-1]
    else:
        hidden = hidden_states
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("hidden_states must be a tensor or a non-empty sequence of tensors")
    if hidden.ndim != 3 or min(hidden.shape) < 1:
        raise ValueError("final hidden state must have shape (batch, sequence_length, hidden_size)")
    if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
        raise ValueError("final hidden state must be finite and floating point")
    if not isinstance(response_mask, torch.Tensor) or response_mask.shape != hidden.shape[:2]:
        raise ValueError("response_mask must match the first two hidden-state dimensions")
    if response_mask.device != hidden.device:
        raise ValueError("response_mask and hidden_states must be on the same device")
    if not bool(((response_mask == 0) | (response_mask == 1)).all()):
        raise ValueError("response_mask must be binary")
    mask = response_mask.to(torch.bool)
    if bool((mask.sum(dim=1) < 1).any()):
        raise ValueError("every sample must contain at least one response token")
    starts = mask[:, :1].to(torch.int64).sum(dim=1)
    if mask.shape[1] > 1:
        starts = starts + ((~mask[:, :-1]) & mask[:, 1:]).sum(dim=1)
    if bool((starts != 1).any()):
        raise ValueError("each response_mask row must select one contiguous token span")

    positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[:2])
    final_positions = positions.masked_fill(~mask, -1).amax(dim=1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, final_positions]


def build_oracle_chat(prompt: str, response: str) -> list[dict[str, str]]:
    """Build the Skywork reward-model card's two-message chat (no system role)."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def extract_scalar_oracle_logits(output: object) -> torch.Tensor:
    """Extract one scalar sequence-classification logit per batch element."""

    logits = _extract_model_logits(output)
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("oracle logits must be finite and floating point")
    if logits.ndim == 1:
        if logits.numel() != 1:
            raise ValueError("one-dimensional oracle logits are only valid for a single sample")
        return logits
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] != 1:
        raise ValueError("oracle logits must have shape (batch, 1)")
    return logits[:, 0]


def score_oracle_chats(
    model: torch.nn.Module,
    tokenizer: object,
    prompts: Sequence[str],
    responses: Sequence[str],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Tokenize model-card chats without a system message and return raw logits."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a sequence of strings")
    if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence):
        raise TypeError("responses must be a sequence of strings")
    if len(prompts) < 1 or len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same non-zero length")
    chats = [
        build_oracle_chat(prompt, response)
        for prompt, response in zip(prompts, responses, strict=True)
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("tokenizer must expose callable apply_chat_template")
    encoded = apply_chat_template(
        chats,
        tokenize=True,
        add_generation_prompt=False,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    )
    if isinstance(encoded, torch.Tensor):
        model_inputs: dict[str, torch.Tensor] = {"input_ids": encoded}
    elif isinstance(encoded, Mapping):
        model_inputs = {
            name: value for name, value in encoded.items() if isinstance(value, torch.Tensor)
        }
        if "input_ids" not in model_inputs:
            raise ValueError("tokenized oracle chat must contain input_ids")
    else:
        raise TypeError("apply_chat_template must return a tensor or a tensor mapping")
    target_device = device
    if target_device is None:
        declared_device = getattr(model, "device", None)
        if declared_device is not None:
            target_device = declared_device
        else:
            first_parameter = next(model.parameters(), None)
            if first_parameter is not None:
                target_device = first_parameter.device
    if target_device is not None:
        model_inputs = {name: value.to(target_device) for name, value in model_inputs.items()}

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model(**model_inputs)
    finally:
        model.train(was_training)
    scores = extract_scalar_oracle_logits(output)
    if scores.shape != (len(chats),):
        raise ValueError("oracle returned a different batch size than the input chats")
    return scores.detach()


__all__ = [
    "CandidateTokenBatch",
    "EXACT_SAMPLING_DEFAULTS",
    "ExactTokenCandidates",
    "FixedALoRASetup",
    "assert_noop_logits",
    "build_exact_token_candidates",
    "build_oracle_chat",
    "configure_fixed_a_lora",
    "extract_scalar_oracle_logits",
    "generate_exact_candidates",
    "pool_final_response_hidden_state",
    "score_exact_candidates",
    "score_oracle_chats",
    "validate_exact_generation_kwargs",
]

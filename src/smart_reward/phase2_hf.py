"""Pinned Hugging Face runtime for the Phase-2 common-beta state machine.

The pure orchestration in :mod:`smart_reward.phase2_rollout` owns the
train/test information boundary.  This module supplies the concrete
Qwen/Skywork sessions while preserving two additional runtime invariants:

* the policy and operational oracle are never resident at the same time; and
* every policy arm is written directly from the zero-LoRA-B coordinate origin.

For an updated arm, trajectories are first sampled from that exact policy.
Both the updated and reference policies are then evaluated on those same token
histories, yielding the required on-policy
``KL(pi_updated || pi_reference)`` estimate without retokenization.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from . import hf as _hf
from . import phase1 as _phase1
from .config import validate_config
from .hf import FixedALoRASetup
from .oracle import RobustOracleTransform
from .phase1_rollout import (
    _load_oracle_runtime,
    _load_policy_runtime,
    _template_sha256,
    _zero_b_,
)
from .phase2_rollout import (
    PHASE2_ARM_ORDER,
    Phase2ArmDeployment,
    Phase2OracleSession,
    Phase2PolicyRollout,
    Phase2PolicySession,
    Phase2RuntimeBackend,
    Phase2Trajectory,
)
from .policy_update import (
    select_causal_response_logits,
    selected_causal_updated_to_reference_kl_per_sequence,
    set_tangent_update_,
)
from .prompts import PromptRecord
from .scores import ParameterLayout
from .seeding import derive_seed


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_seed(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be an integer in [0, 2**63 - 1]")
    return value


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _release_cuda(device: torch.device) -> None:
    """Release Python/model references before returning the next session."""

    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _current_a_sha256(model: torch.nn.Module) -> str:
    named_a = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if _hf._lora_kind(name) == "A"
    )
    if not named_a:
        raise RuntimeError("the policy no longer exposes LoRA-A parameters")
    return _hf._fingerprint_named_tensors(named_a)


def _validate_prompt_batch(
    prompts: Sequence[str],
    responses: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(prompts, (str, bytes, bytearray)) or not isinstance(
        prompts,
        Sequence,
    ):
        raise TypeError("prompts must be a sequence of strings")
    if isinstance(responses, (str, bytes, bytearray)) or not isinstance(
        responses,
        Sequence,
    ):
        raise TypeError("responses must be a sequence of strings")
    prompt_values = tuple(prompts)
    response_values = tuple(responses)
    if not prompt_values or len(prompt_values) != len(response_values):
        raise ValueError("prompts and responses must be non-empty and equal length")
    if any(not isinstance(prompt, str) or not prompt for prompt in prompt_values):
        raise ValueError("every prompt must be a non-empty string")
    if any(not isinstance(response, str) for response in response_values):
        raise TypeError("every response must be a string")
    return prompt_values, response_values


@dataclass(slots=True)
class _HuggingFaceOracleSession:
    tokenizer: object
    model: torch.nn.Module
    device: torch.device
    closed: bool = False

    def score_transformed(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        transform: RobustOracleTransform,
        batch_size: int,
    ) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("oracle session is closed")
        if not isinstance(transform, RobustOracleTransform):
            raise TypeError("transform must be RobustOracleTransform")
        size = _positive_integer(batch_size, name="batch_size")
        prompt_values, response_values = _validate_prompt_batch(prompts, responses)
        pieces: list[torch.Tensor] = []
        for start in range(0, len(prompt_values), size):
            raw = _hf.score_oracle_chats(
                self.model,
                self.tokenizer,
                prompt_values[start : start + size],
                response_values[start : start + size],
                device=self.device,
            )
            if (
                not isinstance(raw, torch.Tensor)
                or raw.shape != (min(size, len(prompt_values) - start),)
                or not raw.is_floating_point()
                or not bool(torch.isfinite(raw).all())
            ):
                raise RuntimeError("oracle returned malformed raw scores")
            transformed = transform(raw).detach().to(device="cpu", dtype=torch.float32)
            pieces.append(transformed)
            del raw, transformed
        result = torch.cat(pieces)
        if result.shape != (len(prompt_values),):
            raise RuntimeError("oracle score count changed during batching")
        return result

    def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _HuggingFacePolicySession:
    tokenizer: object
    setup: FixedALoRASetup
    config: Mapping[str, object]
    device: torch.device
    expected_a_sha256: str
    closed: bool = False

    def _assert_fixed_a(self) -> None:
        if self.setup.a_state_sha256 != self.expected_a_sha256:
            raise RuntimeError("loaded LoRA-A identity does not match the artifact")
        if _current_a_sha256(self.setup.model) != self.expected_a_sha256:
            raise RuntimeError("LoRA-A changed during Phase-2 policy evaluation")

    def _set_deployment(self, deployment: Phase2ArmDeployment) -> None:
        named_tangent = self.setup.named_tangent_parameters()
        parameter_device = named_tangent[0][1].device
        if any(parameter.device != parameter_device for _, parameter in named_tangent):
            raise RuntimeError("LoRA-B parameters span multiple devices")
        displacement = deployment.displacement.detach().to(
            device=parameter_device,
        )
        if displacement.numel() != self.setup.layout.dimension:
            raise ValueError("deployment displacement has the wrong tangent dimension")
        set_tangent_update_(
            named_tangent,
            self.setup.layout,
            displacement,
            step_size=1.0,
        )
        self._assert_fixed_a()

    def _selected_logits(self, candidates: _hf.ExactTokenCandidates):
        with torch.no_grad():
            output = self.setup.model(
                input_ids=candidates.input_ids,
                attention_mask=candidates.attention_mask,
                use_cache=False,
            )
            full_logits = _hf._extract_model_logits(output)
            selected = select_causal_response_logits(
                full_logits,
                candidates.response_mask,
            )
            del full_logits, output
        return selected

    def rollout(
        self,
        deployment: Phase2ArmDeployment,
        test_prompts: Sequence[PromptRecord],
        *,
        candidates_per_prompt: int,
        max_response_tokens: int,
        rollout_seed: int,
        kl_token_chunk_size: int,
    ) -> Phase2PolicyRollout:
        if self.closed:
            raise RuntimeError("policy session is closed")
        if not isinstance(deployment, Phase2ArmDeployment):
            raise TypeError("deployment must be Phase2ArmDeployment")
        if deployment.arm_name not in PHASE2_ARM_ORDER:
            raise ValueError("unknown common-beta policy arm")
        candidate_count = _positive_integer(
            candidates_per_prompt,
            name="candidates_per_prompt",
        )
        max_tokens = _positive_integer(
            max_response_tokens,
            name="max_response_tokens",
        )
        chunk_size = _positive_integer(
            kl_token_chunk_size,
            name="kl_token_chunk_size",
        )
        base_seed = _validate_seed(rollout_seed, name="rollout_seed")
        if isinstance(test_prompts, (str, bytes, bytearray)) or not isinstance(
            test_prompts,
            Sequence,
        ):
            raise TypeError("test_prompts must be a PromptRecord sequence")
        prompts = tuple(test_prompts)
        if not prompts or any(
            not isinstance(prompt, PromptRecord) or prompt.split != "test" for prompt in prompts
        ):
            raise ValueError("test_prompts must contain only test PromptRecord objects")

        policy = self.config["policy"]
        if not isinstance(policy, Mapping):
            raise TypeError("validated policy config must be a mapping")
        if max_tokens != int(policy["max_response_tokens"]):
            raise ValueError("runtime response horizon differs from the bound source config")
        sampling = policy["sampling"]
        if not isinstance(sampling, Mapping):
            raise TypeError("validated policy sampling config must be a mapping")
        generation_kwargs: dict[str, object] = {
            **dict(sampling),
            "num_return_sequences": candidate_count,
            "max_new_tokens": max_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }

        trajectories: list[Phase2Trajectory] = []
        kl_pieces: list[torch.Tensor] = []
        try:
            for prompt in prompts:
                # The arm name is intentionally absent from this namespace:
                # every arm uses common random numbers at a given prompt.
                prompt_seed = derive_seed(
                    base_seed,
                    f"phase2-test-prompt:{prompt.prompt_id}",
                )
                self._set_deployment(deployment)
                prompt_inputs, prompt_semantics = _phase1._encode_full_policy_prompt(
                    self.tokenizer,
                    prompt,
                    max_prompt_tokens=int(policy["max_prompt_tokens"]),
                    device=self.device,
                )
                with _phase1._fork_torch_seed(prompt_seed, self.device):
                    candidates = _hf.generate_exact_candidates(
                        self.setup.model,
                        prompt_inputs["input_ids"],
                        prompt_attention_mask=prompt_inputs["attention_mask"],
                        generation_kwargs=generation_kwargs,
                    )
                if candidates.input_ids.shape[0] != candidate_count:
                    raise RuntimeError("policy returned an unexpected candidate count")

                if deployment.arm_name == "zero_b":
                    if bool(torch.count_nonzero(deployment.displacement)):
                        raise RuntimeError("zero-B arm received a nonzero displacement")
                    per_sequence_kl = torch.zeros(
                        candidate_count,
                        dtype=torch.float32,
                        device="cpu",
                    )
                else:
                    updated_logits = self._selected_logits(candidates)
                    _zero_b_(self.setup)
                    self._assert_fixed_a()
                    reference_logits = self._selected_logits(candidates)
                    per_sequence_kl = (
                        selected_causal_updated_to_reference_kl_per_sequence(
                            updated_logits,
                            reference_logits,
                            token_chunk_size=chunk_size,
                        )
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                    )
                    del updated_logits, reference_logits

                prompt_text = _phase1._prompt_text(prompt)
                for candidate_index in range(candidate_count):
                    active = candidates.response_mask[candidate_index].bool()
                    response_ids = candidates.input_ids[candidate_index][active]
                    trajectories.append(
                        Phase2Trajectory(
                            arm_name=deployment.arm_name,
                            prompt_id=str(prompt.prompt_id),
                            candidate_index=candidate_index,
                            prompt=prompt_text,
                            raw_prompt_sha256=str(prompt_semantics["raw_prompt_sha256"]),
                            policy_chat_token_count=int(
                                prompt_semantics["policy_chat_token_count"]
                            ),
                            policy_prompt_token_ids_sha256=str(
                                prompt_semantics["policy_prompt_token_ids_sha256"]
                            ),
                            max_prompt_tokens=int(prompt_semantics["max_prompt_tokens"]),
                            prompt_truncated=False,
                            raw_prompt_preserved=True,
                            response=_phase1._decode_response(
                                self.tokenizer,
                                response_ids,
                            ),
                            token_ids=tuple(
                                int(value)
                                for value in candidates.input_ids[candidate_index].tolist()
                            ),
                            response_mask=tuple(
                                int(value)
                                for value in candidates.response_mask[candidate_index].tolist()
                            ),
                            terminated_by_eos=bool(
                                candidates.terminated_by_eos[candidate_index].item()
                            ),
                            reached_max_length=bool(
                                candidates.reached_max_length[candidate_index].item()
                            ),
                            prompt_rollout_seed=prompt_seed,
                        )
                    )
                kl_pieces.append(per_sequence_kl)
                del candidates, prompt_inputs, per_sequence_kl
                self._assert_fixed_a()
        finally:
            _zero_b_(self.setup)
            self._assert_fixed_a()

        kl = torch.cat(kl_pieces)
        expected = len(prompts) * candidate_count
        if kl.shape != (expected,) or len(trajectories) != expected:
            raise RuntimeError("policy rollout geometry changed during execution")
        return Phase2PolicyRollout(
            arm_name=deployment.arm_name,
            trajectories=tuple(trajectories),
            per_sequence_kl_updated_to_reference=kl,
        )

    def close(self) -> None:
        if not self.closed:
            _zero_b_(self.setup)
            self._assert_fixed_a()
            self.closed = True


class HuggingFacePhase2Backend(Phase2RuntimeBackend):
    """Concrete, local-snapshot-only backend for the formal Phase-2 run."""

    def __init__(
        self,
        source_config: Mapping[str, object],
        *,
        device: str | torch.device = "cuda",
        local_files_only: bool = True,
    ) -> None:
        self.config = validate_config(source_config)
        self.device = torch.device(device)
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be bool")
        self.local_files_only = local_files_only
        if any(self.config[section]["dtype"] != "float32" for section in ("policy", "oracle")):
            raise ValueError("formal Phase-2 policy and oracle runtimes require float32")
        policy = self.config["policy"]
        if not isinstance(policy, Mapping) or policy.get("fixed_lora_a") is None:
            raise ValueError("formal Phase-2 runtime requires policy.fixed_lora_a")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self._active_session: str | None = None

    def _enter(self, name: str) -> None:
        if self._active_session is not None:
            raise RuntimeError(
                "policy and oracle sessions must never be co-resident; "
                f"active={self._active_session!r}, requested={name!r}"
            )
        self._active_session = name

    def _exit(self, name: str) -> None:
        if self._active_session != name:
            raise RuntimeError("Phase-2 backend session lifecycle is inconsistent")
        self._active_session = None
        _release_cuda(self.device)

    @contextmanager
    def oracle_session(
        self,
        *,
        expected_chat_template_sha256: str,
    ) -> Iterator[Phase2OracleSession]:
        expected = _validate_digest(
            expected_chat_template_sha256,
            name="expected_chat_template_sha256",
        )
        self._enter("oracle")
        runtime = None
        session = None
        try:
            runtime = _load_oracle_runtime(
                self.config,
                device=self.device,
                local_files_only=self.local_files_only,
            )
            if _template_sha256(runtime.tokenizer) != expected:
                raise RuntimeError("loaded oracle chat template differs from the artifact")
            session = _HuggingFaceOracleSession(
                tokenizer=runtime.tokenizer,
                model=runtime.model,
                device=self.device,
            )
            yield session
        finally:
            if session is not None:
                session.close()
            del session, runtime
            self._exit("oracle")

    @contextmanager
    def policy_session(
        self,
        *,
        seed: int,
        expected_a_sha256: str,
        expected_layout: ParameterLayout,
        expected_chat_template_sha256: str,
    ) -> Iterator[Phase2PolicySession]:
        validated_seed = _validate_seed(seed, name="seed")
        expected_a = _validate_digest(
            expected_a_sha256,
            name="expected_a_sha256",
        )
        expected_template = _validate_digest(
            expected_chat_template_sha256,
            name="expected_chat_template_sha256",
        )
        if not isinstance(expected_layout, ParameterLayout):
            raise TypeError("expected_layout must be ParameterLayout")
        self._enter("policy")
        runtime = None
        session = None
        try:
            runtime = _load_policy_runtime(
                self.config,
                seed=validated_seed,
                device=self.device,
                local_files_only=self.local_files_only,
            )
            if runtime.setup.layout != expected_layout:
                raise RuntimeError("loaded LoRA-B layout differs from the artifact")
            if runtime.setup.a_state_sha256 != expected_a:
                raise RuntimeError("loaded LoRA-A fingerprint differs from the artifact")
            if _template_sha256(runtime.tokenizer) != expected_template:
                raise RuntimeError("loaded policy chat template differs from the artifact")
            session = _HuggingFacePolicySession(
                tokenizer=runtime.tokenizer,
                setup=runtime.setup,
                config=self.config,
                device=self.device,
                expected_a_sha256=expected_a,
            )
            session._assert_fixed_a()
            yield session
        finally:
            if session is not None:
                session.close()
            del session, runtime
            self._exit("policy")


def measured_kl_standard_error(values: torch.Tensor) -> float:
    """Return a finite descriptive SE for per-sequence KL diagnostics."""

    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 1
        or values.numel() < 2
        or not values.is_floating_point()
        or values.requires_grad
        or not bool(torch.isfinite(values).all())
        or bool((values < 0.0).any())
    ):
        raise ValueError("values must be a finite detached non-negative KL vector")
    value = values.detach().to(device="cpu", dtype=torch.float64)
    result = float((value.std(unbiased=True) / math.sqrt(value.numel())).item())
    if not math.isfinite(result) or result < 0.0:
        raise FloatingPointError("measured KL standard error is invalid")
    return result


__all__ = [
    "HuggingFacePhase2Backend",
    "measured_kl_standard_error",
]

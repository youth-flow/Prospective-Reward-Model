from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import smart_reward.phase2_hf as phase2_hf
from smart_reward.config import load_config
from smart_reward.hf import FixedALoRASetup, _fingerprint_named_tensors
from smart_reward.oracle import RobustOracleTransform
from smart_reward.phase2_hf import (
    HuggingFacePhase2Backend,
    measured_kl_standard_error,
)
from smart_reward.phase2_rollout import Phase2ArmDeployment
from smart_reward.prompts import ChatMessage, PromptRecord
from smart_reward.scores import ParameterLayout

ROOT = Path(__file__).resolve().parents[1]


class _TinyTokenizer:
    chat_template = "tiny-template"
    eos_token_id = 3
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, *args, **kwargs):
        del args
        self.calls.append(dict(kwargs))
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 2), dtype=torch.int64),
        }

    def decode(self, token_ids, **kwargs):
        del kwargs
        return " ".join(str(int(value)) for value in token_ids)


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Parameter(
            torch.tensor([1.25], dtype=torch.float32),
            requires_grad=False,
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.float32),
            requires_grad=True,
        )
        self.generation_config = SimpleNamespace(eos_token_id=3, pad_token_id=0)

    def generate(self, input_ids, **kwargs):
        count = int(kwargs["num_return_sequences"])
        rows = []
        for index in range(count):
            response = torch.tensor(
                [4 + index, 3],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            rows.append(torch.cat((input_ids[0], response)))
        return torch.stack(rows)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        batch, length = input_ids.shape
        vocab = 8
        logits = torch.zeros(
            (batch, length, vocab),
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[..., 0] = self.lora_B * 2.0
        logits[..., 1] = -self.lora_B
        return SimpleNamespace(logits=logits)


def _tiny_policy_runtime():
    model = _TinyPolicy().eval()
    named_b = (("lora_B", model.lora_B),)
    layout = ParameterLayout.from_named_parameters(named_b)
    a_sha = _fingerprint_named_tensors((("lora_A", model.lora_A),))
    setup = FixedALoRASetup(
        model=model,
        layout=layout,
        a_state_sha256=a_sha,
        trainable_names=layout.names,
    )
    return SimpleNamespace(
        tokenizer=_TinyTokenizer(),
        setup=setup,
    )


def _prompt() -> PromptRecord:
    return PromptRecord(
        prompt_id="test-1",
        messages=(ChatMessage(role="user", content="Explain one thing."),),
        split="test",
    )


def test_real_policy_session_applies_direct_update_and_measures_on_policy_kl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _tiny_policy_runtime()
    monkeypatch.setattr(
        phase2_hf,
        "_load_policy_runtime",
        lambda *args, **kwargs: runtime,
    )
    backend = HuggingFacePhase2Backend(
        load_config(ROOT / "configs" / "common_beta_pilot_base.yaml"),
        device="cpu",
    )
    expected_template = phase2_hf._template_sha256(runtime.tokenizer)
    deployment = Phase2ArmDeployment(
        arm_name="prorm_plus",
        beta_common=7.0,
        displacement=torch.tensor([0.5], dtype=torch.float64),
        direction_evidence={"kind": "test"},
        common_beta_evidence={"kind": "test"},
    )

    with backend.policy_session(
        seed=20260801,
        expected_a_sha256=runtime.setup.a_state_sha256,
        expected_layout=runtime.setup.layout,
        expected_chat_template_sha256=expected_template,
    ) as session:
        rollout = session.rollout(
            deployment,
            (_prompt(),),
            candidates_per_prompt=2,
            max_response_tokens=256,
            rollout_seed=91,
            kl_token_chunk_size=1,
        )

    assert rollout.arm_name == "prorm_plus"
    assert len(rollout.trajectories) == 2
    assert rollout.history_source == "updated_policy"
    assert rollout.kl_orientation == "pi_updated_to_pi0"
    assert bool((rollout.per_sequence_kl_updated_to_reference > 0).all())
    assert all(
        item.prompt_rollout_seed == rollout.trajectories[0].prompt_rollout_seed
        for item in rollout.trajectories
    )
    assert all(item.prompt == "Explain one thing." for item in rollout.trajectories)
    assert all(item.policy_chat_token_count == 2 for item in rollout.trajectories)
    assert all(item.prompt_truncated is False for item in rollout.trajectories)
    assert runtime.tokenizer.calls
    assert runtime.tokenizer.calls[0]["truncation"] is False
    assert "max_length" not in runtime.tokenizer.calls[0]
    assert not bool(torch.count_nonzero(runtime.setup.model.lora_B))


def test_policy_rollout_fails_closed_instead_of_truncating_long_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _tiny_policy_runtime()

    class LongPromptTokenizer(_TinyTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            del args
            self.calls.append(dict(kwargs))
            return {
                "input_ids": torch.arange(1025, dtype=torch.int64).reshape(1, -1),
                "attention_mask": torch.ones((1, 1025), dtype=torch.int64),
            }

    runtime.tokenizer = LongPromptTokenizer()
    monkeypatch.setattr(
        phase2_hf,
        "_load_policy_runtime",
        lambda *args, **kwargs: runtime,
    )
    backend = HuggingFacePhase2Backend(
        load_config(ROOT / "configs" / "common_beta_pilot_base.yaml"),
        device="cpu",
    )
    deployment = Phase2ArmDeployment(
        arm_name="zero_b",
        beta_common=7.0,
        displacement=torch.zeros(1),
        direction_evidence=None,
        common_beta_evidence=None,
    )
    with (
        backend.policy_session(
            seed=20260801,
            expected_a_sha256=runtime.setup.a_state_sha256,
            expected_layout=runtime.setup.layout,
            expected_chat_template_sha256=phase2_hf._template_sha256(runtime.tokenizer),
        ) as session,
        pytest.raises(ValueError, match="truncation is forbidden"),
    ):
        session.rollout(
            deployment,
            (_prompt(),),
            candidates_per_prompt=2,
            max_response_tokens=256,
            rollout_seed=91,
            kl_token_chunk_size=1,
        )
    assert runtime.tokenizer.calls[0]["truncation"] is False
    assert "max_length" not in runtime.tokenizer.calls[0]


def test_trajectory_rejects_prompt_semantics_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _tiny_policy_runtime()
    monkeypatch.setattr(
        phase2_hf,
        "_load_policy_runtime",
        lambda *args, **kwargs: runtime,
    )
    backend = HuggingFacePhase2Backend(
        load_config(ROOT / "configs" / "common_beta_pilot_base.yaml"),
        device="cpu",
    )
    deployment = Phase2ArmDeployment(
        arm_name="zero_b",
        beta_common=7.0,
        displacement=torch.zeros(1),
        direction_evidence=None,
        common_beta_evidence=None,
    )
    with backend.policy_session(
        seed=20260801,
        expected_a_sha256=runtime.setup.a_state_sha256,
        expected_layout=runtime.setup.layout,
        expected_chat_template_sha256=phase2_hf._template_sha256(runtime.tokenizer),
    ) as session:
        rollout = session.rollout(
            deployment,
            (_prompt(),),
            candidates_per_prompt=2,
            max_response_tokens=256,
            rollout_seed=91,
            kl_token_chunk_size=1,
        )
    with pytest.raises(ValueError, match="raw prompt SHA256"):
        replace(rollout.trajectories[0], raw_prompt_sha256="0" * 64)


def test_zero_b_session_skips_divergence_and_stays_exactly_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _tiny_policy_runtime()
    monkeypatch.setattr(
        phase2_hf,
        "_load_policy_runtime",
        lambda *args, **kwargs: runtime,
    )
    backend = HuggingFacePhase2Backend(
        load_config(ROOT / "configs" / "common_beta_pilot_base.yaml"),
        device="cpu",
    )
    deployment = Phase2ArmDeployment(
        arm_name="zero_b",
        beta_common=7.0,
        displacement=torch.zeros(1),
        direction_evidence=None,
        common_beta_evidence=None,
    )
    with backend.policy_session(
        seed=20260801,
        expected_a_sha256=runtime.setup.a_state_sha256,
        expected_layout=runtime.setup.layout,
        expected_chat_template_sha256=phase2_hf._template_sha256(runtime.tokenizer),
    ) as session:
        rollout = session.rollout(
            deployment,
            (_prompt(),),
            candidates_per_prompt=2,
            max_response_tokens=256,
            rollout_seed=91,
            kl_token_chunk_size=1,
        )
    assert not bool(torch.count_nonzero(rollout.per_sequence_kl_updated_to_reference))
    assert not bool(torch.count_nonzero(runtime.setup.model.lora_B))


def test_oracle_session_applies_only_frozen_transform_and_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _TinyTokenizer()
    model = torch.nn.Linear(1, 1).eval()
    monkeypatch.setattr(
        phase2_hf,
        "_load_oracle_runtime",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=tokenizer, model=model),
    )
    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def fake_score(model_arg, tokenizer_arg, prompts, responses, *, device):
        assert model_arg is model
        assert tokenizer_arg is tokenizer
        assert device == torch.device("cpu")
        calls.append((tuple(prompts), tuple(responses)))
        return torch.arange(len(prompts), dtype=torch.float32) + 1.0

    monkeypatch.setattr(phase2_hf._hf, "score_oracle_chats", fake_score)
    backend = HuggingFacePhase2Backend(
        load_config(ROOT / "configs" / "common_beta_pilot_base.yaml"),
        device="cpu",
    )
    transform = RobustOracleTransform(b=0.0, tau=1.0)
    with backend.oracle_session(
        expected_chat_template_sha256=phase2_hf._template_sha256(tokenizer)
    ) as session:
        values = session.score_transformed(
            ("p0", "p1", "p2"),
            ("r0", "r1", "r2"),
            transform=transform,
            batch_size=2,
        )
    expected = torch.cat(
        (
            transform(torch.tensor([1.0, 2.0])),
            transform(torch.tensor([1.0])),
        )
    )
    torch.testing.assert_close(values, expected)
    assert calls == [
        (("p0", "p1"), ("r0", "r1")),
        (("p2",), ("r2",)),
    ]


def test_backend_rejects_policy_oracle_co_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _TinyTokenizer()
    model = torch.nn.Linear(1, 1).eval()
    monkeypatch.setattr(
        phase2_hf,
        "_load_oracle_runtime",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=tokenizer, model=model),
    )
    backend = HuggingFacePhase2Backend(
        load_config(ROOT / "configs" / "common_beta_pilot_base.yaml"),
        device="cpu",
    )
    with (
        backend.oracle_session(expected_chat_template_sha256=phase2_hf._template_sha256(tokenizer)),
        pytest.raises(RuntimeError, match="never be co-resident"),
        backend.oracle_session(expected_chat_template_sha256=phase2_hf._template_sha256(tokenizer)),
    ):
        pass
    assert backend._active_session is None


def test_phase2_backend_rejects_a_source_config_without_global_fixed_a() -> None:
    with pytest.raises(ValueError, match="fixed_lora_a"):
        HuggingFacePhase2Backend(
            load_config(ROOT / "configs" / "main.yaml"),
            device="cpu",
        )


def test_measured_kl_standard_error_is_descriptive_and_strict() -> None:
    values = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    assert measured_kl_standard_error(values) == pytest.approx(
        values.std(unbiased=True).item() / 3**0.5
    )
    with pytest.raises(ValueError, match="non-negative"):
        measured_kl_standard_error(torch.tensor([0.0, -0.1]))

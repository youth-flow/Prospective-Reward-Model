from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import smart_reward.phase1_rollout as stage
from smart_reward.config import config_hash, load_config
from smart_reward.data import CandidateNode
from smart_reward.experiment import (
    ControlledFeatureExperiment,
    EvaluationTensorData,
    TrainingTensorData,
)
from smart_reward.hf import FixedALoRASetup
from smart_reward.oracle import RobustOracleTransform
from smart_reward.prompts import ChatMessage, PromptRecord
from smart_reward.scores import ParameterLayout
from smart_reward.seeding import SeedBundle


def _head_sha(weight: list[float]) -> str:
    tensor = torch.tensor(weight, dtype=torch.float32)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(bytes(tensor.view(torch.uint8).tolist()))
    return digest.hexdigest()


def _comparison(
    digest: str,
    seed: int,
    artifact_metadata_sha256: str = "d" * 64,
) -> dict[str, object]:
    def learner(name: str, weight: list[float]) -> dict[str, object]:
        return {
            "method": name,
            "head_weight": weight,
            "head_sha256": _head_sha(weight),
        }

    return {
        "schema_version": "controlled-comparison/v1",
        "config_hash": digest,
        "seed": seed,
        "artifact_dir": "phase1",
        "artifact_metadata_sha256": artifact_metadata_sha256,
        "run_manifest": "run-manifest.json",
        "run_manifest_sha256": "c" * 64,
        "environment_identity": {
            "formal": True,
            "git_commit": "a" * 40,
            "image_sha256": "b" * 64,
            "hf_inventory_sha256": "c" * 64,
            "account": "sigroup",
            "partition": "gpu-l20",
            "gpu_models": ["test-gpu"],
        },
        "damping_runs": [
            {
                "damping_multiplier": 0.1,
                "result": {
                    "bt_mle": learner("bt_mle", [8.0]),
                    "srm_plus": learner("srm_plus", [9.0]),
                },
            },
            {
                "damping_multiplier": 1.0,
                "result": {
                    "bt_mle": learner("bt_mle", [0.8]),
                    "srm_plus": learner("srm_plus", [1.2]),
                },
            },
        ],
    }


def _candidate(
    prompt_id: str,
    index: int,
    *,
    prompt_tokens: tuple[int, ...],
    response_tokens: tuple[int, ...],
) -> CandidateNode:
    return CandidateNode(
        prompt_id=prompt_id,
        candidate_id=f"{prompt_id}::{index}",
        prompt=f"prompt {prompt_id}",
        response="response",
        token_ids=(*prompt_tokens, *response_tokens, 0),
        response_mask=(
            *(0 for _ in prompt_tokens),
            *(1 for _ in response_tokens),
            0,
        ),
        terminated_by_eos=False,
        reached_max_length=False,
    )


def test_policy_runtime_uses_global_lora_a_seed_for_every_phase2_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "common_beta_pilot_base.yaml"
    )
    expected_sha = config["policy"]["fixed_lora_a"]["expected_sha256"]
    expected_seed = config["policy"]["fixed_lora_a"]["initialization_seed"]
    tokenizer_factory = object()
    model_factory = object()
    tokenizer = SimpleNamespace(
        chat_template="test-template",
        pad_token_id=0,
        eos_token_id=1,
        truncation_side=None,
    )

    class Model:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    model = Model()
    transformers = SimpleNamespace(
        AutoTokenizer=tokenizer_factory,
        AutoModelForCausalLM=model_factory,
    )
    peft = SimpleNamespace(LoraConfig=lambda **kwargs: kwargs)
    monkeypatch.setattr(
        stage._phase1,
        "_require_module",
        lambda name: transformers if name == "transformers" else peft,
    )
    monkeypatch.setattr(
        stage._phase1,
        "_load_pretrained",
        lambda factory, *args, **kwargs: tokenizer if factory is tokenizer_factory else model,
    )
    observed_seeds: list[int] = []

    @contextmanager
    def fake_fork(seed, device):
        del device
        observed_seeds.append(seed)
        yield

    monkeypatch.setattr(stage._phase1, "_fork_torch_seed", fake_fork)
    setup = SimpleNamespace(model=model, a_state_sha256=expected_sha)
    monkeypatch.setattr(stage._hf, "configure_fixed_a_lora", lambda *args, **kwargs: setup)

    first = stage._load_policy_runtime(
        config,
        seed=20260801,
        device=torch.device("cpu"),
        local_files_only=True,
    )
    second = stage._load_policy_runtime(
        config,
        seed=20260802,
        device=torch.device("cpu"),
        local_files_only=True,
    )

    assert first.setup.a_state_sha256 == second.setup.a_state_sha256 == expected_sha
    assert observed_seeds == [expected_seed, expected_seed, expected_seed, expected_seed]

    setup.a_state_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="fingerprint"):
        stage._load_policy_runtime(
            config,
            seed=20260803,
            device=torch.device("cpu"),
            local_files_only=True,
        )


def test_comparison_parser_selects_only_unique_main_run_and_checks_identity() -> None:
    digest = "a" * 64
    value = _comparison(digest, 17)
    heads = stage.parse_comparison_heads(
        value,
        expected_config_hash=digest,
        expected_seed=17,
        expected_artifact_metadata_sha256="d" * 64,
        expected_dimension=1,
    )
    assert heads == {"bt_mle": (0.8,), "prorm_plus": (1.2,)}


def test_comparison_parser_accepts_canonical_v2_learner_keys() -> None:
    digest = "a" * 64
    value = _comparison(digest, 17)
    value["schema_version"] = "controlled-comparison/v2"
    for run in value["damping_runs"]:
        result = run["result"]
        legacy = result.pop("srm_plus")
        legacy["method"] = "prorm_plus"
        result["prorm_plus"] = legacy

    heads = stage.parse_comparison_heads(
        value,
        expected_config_hash=digest,
        expected_seed=17,
        expected_artifact_metadata_sha256="d" * 64,
        expected_dimension=1,
    )

    assert heads == {"bt_mle": (0.8,), "prorm_plus": (1.2,)}

    duplicate = json.loads(json.dumps(value))
    duplicate["damping_runs"].append(duplicate["damping_runs"][1])
    with pytest.raises(ValueError, match="exactly one"):
        stage.parse_comparison_heads(
            duplicate,
            expected_config_hash=digest,
            expected_seed=17,
            expected_artifact_metadata_sha256="d" * 64,
        )
    tampered = json.loads(json.dumps(value))
    tampered["damping_runs"][1]["result"]["bt_mle"]["head_weight"] = [0.9]
    with pytest.raises(ValueError, match="head_sha256"):
        stage.parse_comparison_heads(
            tampered,
            expected_config_hash=digest,
            expected_seed=17,
            expected_artifact_metadata_sha256="d" * 64,
        )


def test_probe_selection_is_order_invariant_and_padding_preserves_token_roles() -> None:
    values = [
        _candidate("t0", 0, prompt_tokens=(1, 2), response_tokens=(3,)),
        _candidate("t0", 1, prompt_tokens=(1, 2), response_tokens=(4, 5)),
        _candidate("t1", 0, prompt_tokens=(6,), response_tokens=(7, 8)),
        _candidate("x0", 0, prompt_tokens=(9,), response_tokens=(10,)),
    ]
    first = stage.select_kl_probe_nodes(values, ("t0", "t1"), count=3, seed=11)
    second = stage.select_kl_probe_nodes(list(reversed(values)), ("t0", "t1"), count=3, seed=11)
    assert [node.candidate_id for node in first] == [node.candidate_id for node in second]
    assert all(node.prompt_id != "x0" for node in first)

    ordered = (values[0], values[2])
    batch = stage.pad_reference_candidates(
        ordered,
        pad_token_id=0,
        source_model_id=123,
        source_trainable_sha256="b" * 64,
    )
    assert batch.prompt_width == 1
    assert batch.input_ids.tolist() == [[1, 2, 3], [6, 7, 8]]
    assert batch.attention_mask.tolist() == [
        [True, True, True],
        [True, True, True],
    ]
    assert batch.response_mask.tolist() == [
        [False, False, True],
        [False, True, True],
    ]
    # The original active prefix and every response index remain unchanged.
    for row, node in enumerate(ordered):
        final_active = max(index for index, bit in enumerate(node.response_mask) if bit)
        assert batch.input_ids[row, : final_active + 1].tolist() == list(
            node.token_ids[: final_active + 1]
        )
        assert batch.response_mask[row, : final_active + 1].tolist() == [
            bool(value) for value in node.response_mask[: final_active + 1]
        ]


def _direction_evidence() -> dict[str, object]:
    return {"schema_version": "policy-direction/v1", "direction": [1.0]}


def _update_evidence() -> dict[str, object]:
    return {
        "schema_version": "measured-kl-update/v1",
        "converged": True,
        "applied": True,
    }


def test_rollout_result_uses_one_shared_reference_mean(tmp_path: Path) -> None:
    reference_base = (tmp_path / "run").resolve()
    result = stage.assemble_rollout_result(
        config_sha256="c" * 64,
        seed=3,
        artifact_dir=(tmp_path / "artifact").resolve(),
        comparison_json=(reference_base / "comparison.json").resolve(),
        updated_rollouts_jsonl=(reference_base / "updated_rollouts.jsonl").resolve(),
        reference_base=reference_base,
        artifact_metadata_sha256="d" * 64,
        comparison_sha256="e" * 64,
        updated_rollouts_sha256="f" * 64,
        run_manifest_sha256="a" * 64,
        environment_identity={
            "formal": True,
            "git_commit": "b" * 40,
            "image_sha256": "c" * 64,
            "hf_inventory_sha256": "d" * 64,
            "account": "sigroup",
            "partition": "gpu-l20",
            "gpu_models": ["NVIDIA L20"],
        },
        kl_probe_candidate_ids=("a", "b"),
        reference_rollout_rewards=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        artifact_test_rewards=torch.tensor([-0.5, 0.5, -0.5, 0.5]),
        learner_direction_evidence={
            "bt_mle": _direction_evidence(),
            "prorm_plus": _direction_evidence(),
        },
        learner_update_evidence={
            "bt_mle": _update_evidence(),
            "prorm_plus": _update_evidence(),
        },
        learner_transformed_rewards={
            "bt_mle": [0.0, 2.0, 2.0, 4.0],
            "prorm_plus": [1.0, 1.0, 3.0, 3.0],
        },
        rollout_seed=7,
        num_test_prompts=2,
        candidates_per_prompt=2,
    )
    assert result["artifact_dir"] == "../artifact"
    assert result["comparison_json"] == "comparison.json"
    assert result["updated_rollouts_jsonl"] == "updated_rollouts.jsonl"
    assert result["test_reference"]["source"] == "zero_b_common_random_number_rollout"
    assert result["test_reference"]["transformed_oracle_mean"] == pytest.approx(0.0)
    assert result["artifact_test_descriptive_sanity"]["paired_with_updated_rollouts"] is False
    assert result["learners"]["bt_mle"]["paired_improvement_over_zero_b_reference"][
        "mean_difference"
    ] == pytest.approx(2.0)
    assert result["learners"]["bt_mle"]["paired_improvement_over_zero_b_reference"][
        "sample_standard_error"
    ] == pytest.approx(1.0)
    assert result["learners"]["prorm_plus"]["paired_improvement_over_zero_b_reference"][
        "mean_difference"
    ] == pytest.approx(2.0)
    assert result["learners"]["prorm_plus"]["paired_improvement_over_zero_b_reference"][
        "sample_standard_error"
    ] == pytest.approx(1.0)
    assert result["artifact_metadata_sha256"] == "d" * 64
    assert result["comparison_sha256"] == "e" * 64
    assert result["updated_rollouts_sha256"] == "f" * 64
    assert result["run_manifest_sha256"] == "a" * 64
    assert result["environment_identity"]["account"] == "sigroup"
    assert result["raw_oracle_values_serialized"] is False


def test_two_file_publish_rolls_back_first_file_if_second_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rollouts = tmp_path / "updated_rollouts.jsonl"
    result = tmp_path / "result.json"
    real_replace = stage.os.replace
    calls = 0

    def fail_second_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(stage.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected"):
        stage._publish_output_pair(
            rollouts,
            ({"policy": "reference"},),
            result,
            {"schema_version": "test"},
        )
    assert not rollouts.exists()
    assert not result.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_formal_environment_must_match_artifact_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SRM_GIT_COMMIT", raising=False)
    monkeypatch.delenv("SRM_IMAGE_SHA256", raising=False)
    monkeypatch.delenv("SRM_HF_INVENTORY_SHA256", raising=False)
    producer = {
        "git_commit": "a" * 40,
        "image_sha256": "b" * 64,
        "hf_inventory_sha256": "c" * 64,
    }
    assert stage._validate_producer_identity(producer) == producer

    monkeypatch.setenv("SRM_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("SRM_IMAGE_SHA256", "b" * 64)
    monkeypatch.setenv("SRM_HF_INVENTORY_SHA256", "c" * 64)
    assert stage._validate_producer_identity(producer) == producer
    monkeypatch.setenv("SRM_IMAGE_SHA256", "c" * 64)
    with pytest.raises(ValueError, match="producer identity"):
        stage._validate_producer_identity(producer)


def test_legacy_producer_without_inventory_is_nonformal_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = {"git_commit": "a" * 40, "image_sha256": "b" * 64}
    for name in (
        "SLURM_JOB_ID",
        "PRORM_GIT_COMMIT",
        "SRM_GIT_COMMIT",
        "PRORM_IMAGE_SHA256",
        "SRM_IMAGE_SHA256",
        "PRORM_HF_INVENTORY_SHA256",
        "SRM_HF_INVENTORY_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    assert stage._validate_producer_identity(legacy) == legacy

    monkeypatch.setenv("PRORM_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("PRORM_IMAGE_SHA256", "b" * 64)
    monkeypatch.setenv("PRORM_HF_INVENTORY_SHA256", "c" * 64)
    with pytest.raises(ValueError, match="producer identity"):
        stage._validate_producer_identity(legacy)


def _experiment() -> ControlledFeatureExperiment:
    scores = torch.tensor([[[-1.0], [0.0], [1.0], [2.0]], [[-0.5], [0.5], [1.5], [2.5]]])
    features = scores.clone()
    train = TrainingTensorData(
        prompt_ids=("t0", "t1"),
        policy_scores=scores,
        reward_features=features,
        h=torch.tensor([0.2, -0.1]),
        left_wins=torch.tensor([1, 0]),
        num_annotations=torch.tensor([1, 1]),
    )

    def heldout(prompt_ids: tuple[str, ...], reward_offset: float) -> EvaluationTensorData:
        count = len(prompt_ids)
        return EvaluationTensorData(
            prompt_ids=prompt_ids,
            policy_scores=torch.tensor([[[-1.0], [0.0], [1.0], [2.0]]]).repeat(count, 1, 1),
            reward_features=torch.tensor([[[-1.0], [0.0], [1.0], [2.0]]]).repeat(count, 1, 1),
            true_rewards=torch.full((count, 4), reward_offset),
        )

    return ControlledFeatureExperiment(
        train=train,
        validation=heldout(("v0",), 0.0),
        test=heldout(("x0", "x1"), 0.0),
    )


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.lora_B = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.generation_config = SimpleNamespace(eos_token_id=4, pad_token_id=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 5), dtype=torch.float32)
        logits[..., 1] = 4.0 * self.lora_A * self.lora_B
        logits[..., 2] = -4.0 * self.lora_A * self.lora_B
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        repeats = int(kwargs["num_return_sequences"])
        prefix = input_ids.repeat_interleave(repeats, dim=0)
        logits = torch.stack(
            (
                4.0 * self.lora_B.expand(repeats),
                -4.0 * self.lora_B.expand(repeats),
            ),
            dim=1,
        )
        sampled = torch.multinomial(logits.softmax(dim=1), 1) + 1
        return torch.cat((prefix, sampled), dim=1)


class _TinyTokenizer:
    def __init__(self, template: str) -> None:
        self.chat_template = template
        self.eos_token_id = 4
        self.pad_token_id = 0
        self.truncation_side = "left"

    def apply_chat_template(self, chats: object, **_: object) -> dict[str, torch.Tensor]:
        # Policy receives one chat (a list of message dicts); oracle receives a
        # batch (a list of lists of message dicts).
        batch = len(chats) if chats and isinstance(chats[0], list) else 1
        ids = torch.tensor([[3, 3]] * batch, dtype=torch.int64)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids: list[int], **_: object) -> str:
        return " ".join(str(value) for value in token_ids)


class _TinyOracle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        # Finite deterministic score; no raw value is persisted by the stage.
        return SimpleNamespace(logits=input_ids.float().mean(dim=1, keepdim=True) / 10.0)


def _prompt_records() -> list[PromptRecord]:
    return [
        PromptRecord(
            prompt_id=prompt_id,
            messages=(ChatMessage(role="user", content=f"prompt {prompt_id}"),),
            split=split,
        )
        for prompt_id, split in (
            ("t0", "train"),
            ("t1", "train"),
            ("v0", "validation"),
            ("x0", "test"),
            ("x1", "test"),
        )
    ]


def _all_candidates() -> list[CandidateNode]:
    result: list[CandidateNode] = []
    for prompt_index, prompt_id in enumerate(("t0", "t1", "v0", "x0", "x1")):
        for candidate_index in range(4):
            result.append(
                _candidate(
                    prompt_id,
                    candidate_index,
                    prompt_tokens=(1, 2) if prompt_index % 2 else (2,),
                    response_tokens=(1 + candidate_index % 2,),
                )
            )
    return result


def test_fake_loader_end_to_end_writes_no_raw_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path("configs/smoke.yaml"))
    # The tiny fixture has only eight train candidates; production smoke keeps
    # the main experiment's 16-candidate KL geometry.
    config["evaluation"]["kl_probe_candidates"] = 8
    seed = int(config["run"]["seed"])
    digest = config_hash(config)
    artifact_path = tmp_path / "fake-artifact"
    artifact_path.mkdir()
    metadata_path = artifact_path / "metadata.json"
    metadata_path.write_text("{}\n", encoding="utf-8")
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(_comparison(digest, seed, metadata_sha)), encoding="utf-8"
    )
    experiment = _experiment()

    policy = _TinyPolicy().eval()
    layout = ParameterLayout.from_named_parameters((("lora_B", policy.lora_B),))
    a_sha = stage._hf._fingerprint_named_tensors((("lora_A", policy.lora_A),))
    setup = FixedALoRASetup(
        model=policy,
        layout=layout,
        a_state_sha256=a_sha,
        trainable_names=layout.names,
    )
    policy_tokenizer = _TinyTokenizer("policy-template")
    oracle_tokenizer = _TinyTokenizer("oracle-template")
    contract = stage._ArtifactContract(
        a_state_sha256=a_sha,
        layout=layout,
        policy_chat_template_sha256=hashlib.sha256(b"policy-template").hexdigest(),
        oracle_chat_template_sha256=hashlib.sha256(b"oracle-template").hexdigest(),
        oracle_transform=RobustOracleTransform(b=0.0, tau=1.0),
        jsonl_sha256={},
    )

    monkeypatch.setattr(stage, "load_controlled_feature_artifact", lambda *a, **k: experiment)
    monkeypatch.setattr(stage, "_artifact_contract", lambda *a, **k: contract)
    monkeypatch.setattr(stage, "load_prompt_jsonl", lambda *a, **k: _prompt_records())
    monkeypatch.setattr(stage, "load_jsonl", lambda *a, **k: _all_candidates())
    monkeypatch.setattr(
        stage,
        "_load_policy_runtime",
        lambda *a, **k: stage._PolicyRuntime(policy_tokenizer, setup),
    )
    monkeypatch.setattr(
        stage,
        "_load_oracle_runtime",
        lambda *a, **k: stage._OracleRuntime(oracle_tokenizer, _TinyOracle().eval()),
    )
    original_match = stage.match_fixed_a_measured_kl
    matched_geometry_dtypes: list[tuple[torch.dtype, torch.dtype]] = []

    def checked_match(*args, **kwargs):
        direction = args[3]
        train_node_scores = kwargs["train_node_scores"]
        matched_geometry_dtypes.append((direction.dtype, train_node_scores.dtype))
        assert direction.device == train_node_scores.device
        return original_match(*args, **kwargs)

    monkeypatch.setattr(stage, "match_fixed_a_measured_kl", checked_match)

    output = tmp_path / "matched.json"
    payload = stage.evaluate_matched_kl_rollouts(
        config,
        seed=seed,
        artifact_dir=artifact_path,
        comparison_json=comparison_path,
        output_json=output,
        device="cpu",
    )
    assert payload["schema_version"] == "matched-kl-rollout/v2"
    assert payload["artifact_dir"] == "fake-artifact"
    assert payload["comparison_json"] == "comparison.json"
    assert payload["updated_rollouts_jsonl"] == "updated_rollouts.jsonl"
    assert output.exists()
    rollout_path = tmp_path / "updated_rollouts.jsonl"
    rows = [json.loads(line) for line in rollout_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 24
    assert {row["schema_version"] for row in rows} == {"updated-rollout/v2"}
    assert {row["policy"] for row in rows} == {"reference", "bt_mle", "prorm_plus"}
    assert {row["policy_source"] for row in rows} == {
        "zero_b_reference",
        "matched_kl_update",
    }
    assert all("raw" not in key for row in rows for key in row)
    assert all("transformed_oracle_reward" in row for row in rows)
    assert payload["common_random_numbers"]["seed"] == SeedBundle.from_base_seed(seed).rollout
    assert (
        payload["learners"]["bt_mle"]["paired_improvement_over_zero_b_reference"]["num_pairs"] == 2
    )
    assert (
        payload["updated_rollouts_sha256"] == hashlib.sha256(rollout_path.read_bytes()).hexdigest()
    )
    assert matched_geometry_dtypes == [
        (torch.float64, torch.float64),
        (torch.float64, torch.float64),
    ]
    assert torch.count_nonzero(policy.lora_B).item() == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        stage.evaluate_matched_kl_rollouts(
            config,
            seed=seed,
            artifact_dir=artifact_path,
            comparison_json=comparison_path,
            output_json=output,
            device="cpu",
        )

    def failed_match(*_: object, **__: object) -> SimpleNamespace:
        with torch.no_grad():
            policy.lora_B.fill_(9.0)
        return SimpleNamespace(converged=False, applied=False)

    monkeypatch.setattr(stage, "match_fixed_a_measured_kl", failed_match)
    failed_directory = tmp_path / "failed-stage"
    failed_output = failed_directory / "matched.json"
    with pytest.raises(RuntimeError, match="restored to zero-B"):
        stage.evaluate_matched_kl_rollouts(
            config,
            seed=seed,
            artifact_dir=artifact_path,
            comparison_json=comparison_path,
            output_json=failed_output,
            device="cpu",
        )
    assert torch.count_nonzero(policy.lora_B).item() == 0
    assert not failed_output.exists()
    assert not (failed_directory / "updated_rollouts.jsonl").exists()

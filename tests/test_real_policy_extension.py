from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import smart_reward.real_policy_extension as extension_module
from smart_reward.config import load_config
from smart_reward.real_policy_extension import (
    ADDITIONAL_RESPONSES,
    BASE_RESPONSES,
    TOTAL_RESPONSES,
    extend_real_policy_rollout,
    load_real_policy_extension_config,
    resolve_source_config,
)


def _row(prompt_id: str, response_index: int, policy: str) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "response_index": response_index,
        "prompt": f"prompt-{prompt_id}",
        "response": f"response-{prompt_id}-{response_index}",
        "oracle_reward": float(response_index),
        "forward_kl": 0.0,
        "policy_instance": policy,
        "reward_source": "pi0",
        "beta": 0.2,
    }


def test_m6_extension_config_freezes_four_plus_two() -> None:
    path = Path("configs/real_policy_m6_extension.yaml")
    extension = load_real_policy_extension_config(path)
    source = resolve_source_config(path, extension)

    assert BASE_RESPONSES == 4
    assert ADDITIONAL_RESPONSES == 2
    assert TOTAL_RESPONSES == 6
    assert source["data"]["num_candidates"] == 6
    assert source["evaluation"]["rollout"]["responses_per_prompt"] == 4
    assert extension["evaluation"]["responses_per_prompt"] == 6


def test_incremental_rollout_retains_four_and_generates_only_two(
    tmp_path: Path, monkeypatch
) -> None:
    extension_path = Path("configs/real_policy_m6_extension.yaml")
    extension = load_real_policy_extension_config(extension_path)
    config = load_config(Path("configs/fisher_trpo_main.yaml"))
    config["execution"]["rollout_checkpoint_prompts"] = 2
    config["execution"]["rollout_prompt_batch_size"] = 1
    seed = config["run"]["seeds"][0]
    policy = "pi0"
    prompts = [SimpleNamespace(prompt_id="p0"), SimpleNamespace(prompt_id="p1")]
    source_rows = [
        _row(prompt.prompt_id, index, policy)
        for prompt in prompts
        for index in range(BASE_RESPONSES)
    ]
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(
        json.dumps({"evidence": {"oracle_transform": {"b": 0.0, "tau": 1.0}}}) + "\n",
        encoding="utf-8",
    )
    reward = tmp_path / "reward.json"
    reward.write_text("{}\n", encoding="utf-8")
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    (adapters / "metadata.json").write_text("{}\n", encoding="utf-8")
    source_rollout = tmp_path / "source-rollout"
    source_rollout.mkdir()
    output = tmp_path / "extended-rollout"
    source_identities = {
        "source_rollout_metadata": "1" * 64,
        "source_rollout_receipt": "2" * 64,
        "source_rollouts": "3" * 64,
    }
    producer = {"git_commit": "4" * 40}
    generated_response_counts: list[int] = []

    class _Model:
        def set_adapter(self, name: str) -> None:
            raise AssertionError(f"pi0 must not set an adapter: {name}")

    monkeypatch.setattr(
        extension_module, "load_real_policy_extension_config", lambda path: extension
    )
    monkeypatch.setattr(extension_module, "resolve_source_config", lambda *args: config)
    monkeypatch.setattr(
        extension_module,
        "_validate_source",
        lambda *args, **kwargs: (config, "5" * 64, {}),
    )
    monkeypatch.setattr(
        extension_module,
        "_validate_source_rollout",
        lambda *args, **kwargs: ({"producer": producer}, source_rows, source_identities),
    )
    monkeypatch.setattr(
        extension_module,
        "_source_adapter_metadata",
        lambda *args, **kwargs: {"producer": producer, "adapters": {"a": {}}},
    )
    monkeypatch.setattr(extension_module, "_test_prompts", lambda *args: prompts)
    monkeypatch.setattr(
        extension_module,
        "_load_models",
        lambda *args, **kwargs: (_Model(), object(), object(), object()),
    )

    def _generate(*args, responses: int, **kwargs):
        generated_response_counts.append(responses)
        batch_prompts = args[4]
        return [
            {
                "prompt_id": prompt.prompt_id,
                "response_index": index,
                "prompt": f"prompt-{prompt.prompt_id}",
                "response": f"new-{prompt.prompt_id}-{index}",
                "oracle_reward": 10.0 + index,
                "forward_kl": 0.0,
            }
            for prompt in batch_prompts
            for index in range(responses)
        ]

    monkeypatch.setattr(extension_module, "_generate_policy_batch", _generate)

    def _write_receipt(path, *args, **kwargs):
        path.write_text("{}\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(extension_module, "write_stage_receipt", _write_receipt)
    monkeypatch.setattr(
        extension_module,
        "validate_extended_real_policy_rollout",
        lambda *args, **kwargs: ({}, []),
    )
    monkeypatch.setattr(extension_module, "_producer", lambda: producer)

    metadata = extend_real_policy_rollout(
        extension_path,
        artifact,
        reward,
        adapters,
        source_rollout,
        output,
        policy_name=policy,
        seed=seed,
        device="cpu",
    )

    rows = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
    assert metadata["base_responses_per_prompt"] == 4
    assert metadata["additional_responses_per_prompt"] == 2
    assert metadata["responses_per_prompt"] == 6
    assert generated_response_counts == [2, 2]
    assert len(rows) == len(prompts) * TOTAL_RESPONSES
    assert [row["response_index"] for row in rows] == [0, 1, 2, 3, 4, 5] * len(prompts)
    assert [row for row in rows if row["response_index"] < BASE_RESPONSES] == source_rows
    assert all(
        str(row["response"]).startswith("new-")
        for row in rows
        if row["response_index"] >= BASE_RESPONSES
    )

#!/usr/bin/env python3
"""Measure full-gradient signal versus prompt-level DPO gradient noise at pi0."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from smart_reward.artifacts import load_exact_delta_artifact
from smart_reward.config import config_hash, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("artifact_dir")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--chunk-prompts", type=int, default=8)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    experiment = load_exact_delta_artifact(
        arguments.artifact_dir,
        expected_config_hash=config_hash(config),
        expected_seed=arguments.seed,
    )
    scores = experiment.train.policy_scores
    rewards = experiment.train.true_rewards
    if scores.ndim != 3 or rewards.shape != scores.shape[:2]:
        raise ValueError("unexpected train score/reward shapes")
    prompts, candidates, dimension = scores.shape
    first, second = torch.triu_indices(candidates, candidates, offset=1)
    pair_count = first.numel()
    gradient_sum = torch.zeros(dimension, dtype=torch.float64)
    prompt_gradient_squared_sum = 0.0
    target_deviation_squared_sum = 0.0

    for start in range(0, prompts, arguments.chunk_prompts):
        stop = min(start + arguments.chunk_prompts, prompts)
        score = scores[start:stop].to(torch.float64)
        reward = rewards[start:stop].to(torch.float64)
        target = torch.sigmoid(reward[:, first] - reward[:, second])
        weight = 0.5 - target
        coefficient = torch.zeros((stop - start, candidates), dtype=torch.float64)
        coefficient.scatter_add_(1, first.expand(stop - start, -1), weight)
        coefficient.scatter_add_(1, second.expand(stop - start, -1), -weight)
        prompt_gradient = (
            arguments.beta * torch.einsum("bm,bmd->bd", coefficient, score) / float(pair_count)
        )
        gradient_sum += prompt_gradient.sum(dim=0)
        prompt_gradient_squared_sum += float(prompt_gradient.square().sum().item())
        target_deviation_squared_sum += float(weight.square().sum().item())

    gradient = gradient_sum / float(prompts)
    signal_squared = float(gradient.square().sum().item())
    prompt_second_moment = prompt_gradient_squared_sum / float(prompts)
    prompt_noise_trace = max(prompt_second_moment - signal_squared, 0.0)
    batch_signal_to_noise = {}
    for batch_size in (2, 8, 16, 32, 64, 128, 512, prompts):
        effective = min(batch_size, prompts)
        noise_norm = math.sqrt(prompt_noise_trace / float(effective))
        batch_signal_to_noise[str(effective)] = (
            math.inf if noise_norm == 0.0 else math.sqrt(signal_squared) / noise_norm
        )

    output = {
        "schema": "prorm-dpo-gradient-signal-diagnostic/v1",
        "seed": arguments.seed,
        "beta": arguments.beta,
        "split": "train_only",
        "prompts": prompts,
        "candidates": candidates,
        "pairs_per_prompt": pair_count,
        "policy_dimension": dimension,
        "full_gradient_l2": math.sqrt(signal_squared),
        "prompt_gradient_rms_l2": math.sqrt(prompt_second_moment),
        "prompt_noise_trace": prompt_noise_trace,
        "batch_signal_to_noise": batch_signal_to_noise,
        "mean_squared_target_deviation_from_half": (
            target_deviation_squared_sum / float(prompts * pair_count)
        ),
        "interpretation": (
            "At pi0, the exact full-batch negative gradient is a descent direction. "
            "Batch SNR quantifies whether prompt-batch stochastic gradients obscure it."
        ),
    }
    target = Path(arguments.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()

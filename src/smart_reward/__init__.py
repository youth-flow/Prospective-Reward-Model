"""Prospective Reward Modeling experiment package."""

from .config import PROTOCOL, config_hash, load_config, validate_config
from .evaluation import (
    GeometrySettings,
    evaluate_local_policy,
    evaluate_reference_policy,
    solve_natural_direction,
    summarize_rollouts,
)
from .exact import (
    ExactDeltaExperiment,
    ExactSplitData,
    MLETrainingConfig,
    ProTrainingConfig,
    evaluate_reward_head,
    fit_mle_reward,
    fit_pro_reward,
    pair_indices,
    pairwise_differences,
    policy_reward_moment,
)
from .exact_phase import assemble_exact_delta_experiment, materialize_exact_delta
from .exact_policy import export_exact_ngd_adapters
from .exact_run import run_exact_reward_comparison
from .rollout import evaluate_policy_rollouts

__all__ = [
    "PROTOCOL",
    "ExactDeltaExperiment",
    "ExactSplitData",
    "GeometrySettings",
    "MLETrainingConfig",
    "ProTrainingConfig",
    "assemble_exact_delta_experiment",
    "config_hash",
    "evaluate_local_policy",
    "evaluate_policy_rollouts",
    "evaluate_reference_policy",
    "evaluate_reward_head",
    "export_exact_ngd_adapters",
    "fit_mle_reward",
    "fit_pro_reward",
    "load_config",
    "materialize_exact_delta",
    "pair_indices",
    "pairwise_differences",
    "policy_reward_moment",
    "run_exact_reward_comparison",
    "solve_natural_direction",
    "summarize_rollouts",
    "validate_config",
]

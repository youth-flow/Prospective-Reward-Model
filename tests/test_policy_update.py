from __future__ import annotations

import pytest
import torch

from smart_reward.policy_update import set_tangent_update_, unflatten_tangent_vector
from smart_reward.scores import ParameterLayout


def test_common_beta_updates_are_exact_inverse_scalings() -> None:
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(3))
    named = (("a", first), ("b", second))
    layout = ParameterLayout.from_named_parameters(named)
    direction = torch.arange(1, 8, dtype=torch.float32)
    set_tangent_update_(named, layout, direction, step_size=1.0)
    beta_one = torch.cat((first.flatten(), second.flatten())).clone()
    set_tangent_update_(named, layout, direction, step_size=0.25)
    beta_four = torch.cat((first.flatten(), second.flatten())).clone()
    assert torch.allclose(beta_four, beta_one / 4.0)


def test_unflatten_rejects_wrong_dimension() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    layout = ParameterLayout.from_named_parameters((("p", parameter),))
    with pytest.raises(ValueError, match="length"):
        unflatten_tangent_vector(torch.zeros(3), layout)

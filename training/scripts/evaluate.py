"""Shared deterministic policy dispatch for Classic v4 evaluation."""
import numpy as np
import torch
from policy import actions, OnnxPolicy


def run_policy(policy, obs, active=None):
    if isinstance(policy, OnnxPolicy) and active is not None:
        result = np.ones(len(obs), dtype=int)
        result[active] = policy(obs[active])
        return result
    return actions(policy, obs) if isinstance(policy, torch.nn.Module) else policy(obs)

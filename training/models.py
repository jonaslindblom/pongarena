"""Ordinary dense PyTorch policies; no feature engineering at inference time."""
import torch
from torch import nn


def network(widths, activation='Tanh'):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layer = nn.Linear(a, b)
        nn.init.orthogonal_(layer.weight, gain=2 ** .5)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i < len(widths) - 2:
            layers.append({'Tanh': nn.Tanh, 'Relu': nn.ReLU}[activation]())
    nn.init.orthogonal_(layers[-1].weight, gain=.01)
    return nn.Sequential(*layers)


def decide(policy, observation):
    if isinstance(policy, nn.Module):
        with torch.inference_mode():
            return policy(torch.from_numpy(observation)).argmax(-1).numpy()
    return policy(observation)

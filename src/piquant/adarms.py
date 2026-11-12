"""Rewrite the action expert's adaRMS modulation layers, whose output depends only on the denoising step.

Each adaRMS norm maps the timestep embedding to a scale, shift and gate through an FP32 linear layer. The
timesteps are the same on every call, so a table of each step's output replaces the layer, and a step reads
one row instead of the layer's weights.
"""

from typing import Literal, NamedTuple, get_args

import torch
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from torch import nn

AdarmsMode = Literal["fp32", "bf16", "precomputed"]
ADARMS_MODES: tuple[AdarmsMode, ...] = get_args(AdarmsMode)


class ActionDiff(NamedTuple):
    """Max absolute action difference on one fixed-noise call."""

    vs_fp32: float  # after the rewrite, against the model as stored
    run_to_run: float  # between two calls of the model as stored: the noise floor for vs_fp32


class _StepTable(nn.Module):
    def __init__(self, rows: torch.Tensor):
        super().__init__()
        self.register_buffer("rows", rows)

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        return self.rows[step.squeeze(-1)]


class _CastInput(nn.Module):
    """Runs a linear layer stored in another dtype on this dtype's input."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.to(self.linear.weight.dtype)).to(x.dtype)


def apply_adarms(policy: PI05Policy, batch: dict[str, torch.Tensor], mode: AdarmsMode) -> ActionDiff | None:
    """Rewrite the adaRMS layers for mode, and measure how far that moves the actions. None for fp32."""
    if mode == "fp32":
        return None
    config = policy.config
    noise = policy.model.sample_noise((1, config.chunk_size, config.max_action_dim), config.device)

    def actions() -> torch.Tensor:
        # clone: a compiled call's output can live in CUDA-graph memory the next call overwrites.
        return policy.predict_action_chunk(batch, noise=noise).clone()

    reference = actions()
    run_to_run = (actions() - reference).abs().max().item()
    if mode == "bf16":
        cast_adarms(policy.model, torch.bfloat16)
    else:
        precompute_adarms(policy.model, config.num_inference_steps)
    return ActionDiff((actions() - reference).abs().max().item(), run_to_run)


def cast_adarms(model: PI05Pytorch, dtype: torch.dtype) -> None:
    for norm in _adaptive_norms(model):
        norm.dense = _CastInput(norm.dense.to(dtype))


@torch.no_grad()
def precompute_adarms(model: PI05Pytorch, num_steps: int) -> None:
    """Replace each adaRMS dense layer with its output at each step. The model then samples correctly only with
    num_steps steps; profile.py checks its actions against the original's."""
    conds = []

    def record_cond(x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        conds.append(model.embed_suffix(x_t, timestep)[3])
        return torch.zeros_like(x_t)

    proj = model.action_in_proj
    noise = torch.zeros(1, model.config.chunk_size, proj.in_features, device=proj.weight.device, dtype=proj.weight.dtype)
    euler_integrate(record_cond, noise, num_steps)

    for norm in _adaptive_norms(model):
        norm.dense = _StepTable(torch.cat([norm.dense(cond) for cond in conds]))
        norm.cond_dim = 1

    embed_suffix = model.embed_suffix

    def embed_suffix_with_step(noisy_actions: torch.Tensor, timestep: torch.Tensor):
        *suffix, _ = embed_suffix(noisy_actions, timestep)
        return *suffix, torch.round((1.0 - timestep) * num_steps).long()[:, None]

    model.embed_suffix = embed_suffix_with_step


def _adaptive_norms(model: PI05Pytorch) -> list[PiGemmaRMSNorm]:
    expert = model.paligemma_with_expert.gemma_expert
    return [m for m in expert.modules() if isinstance(m, PiGemmaRMSNorm) and m.dense is not None]

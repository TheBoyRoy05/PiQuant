"""Simulate signed INT8 attention and MLP linears in the Pi0.5 action expert."""

import logging
from functools import wraps

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.scripts import lerobot_eval
from torch import nn

LOGGER = logging.getLogger(__name__)
INT8_MAX = 127


def fake_int8(values: torch.Tensor) -> torch.Tensor:
    """Symmetric round-trip quantization along the last axis, preserving the original dtype."""
    fp32 = values.float()
    scale = fp32.abs().amax(dim=-1, keepdim=True) / INT8_MAX
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (fp32 / scale).round().clamp(-INT8_MAX, INT8_MAX)
    return (quantized * scale).to(values.dtype)


@torch.no_grad()
def simulate_action_expert_w8a8(policy: PI05Policy) -> tuple[str, ...]:
    """Round-trip weights per output channel and activations per input token.

    Biases, adaRMS, the two outer action projections, the VLM and the vision encoder
    keep their checkpoint precision. The linears still use floating-point kernels.
    """
    expert = policy.model.paligemma_with_expert.gemma_expert.model
    linears = [
        (name, layer)
        for name, layer in expert.named_modules()
        if isinstance(layer, nn.Linear) and (".self_attn." in name or ".mlp." in name)
    ]
    if not linears:
        raise ValueError("No attention or MLP linears found in the Pi0.5 action expert.")

    for _, layer in linears:
        layer.weight.copy_(fake_int8(layer.weight))
        layer.register_forward_pre_hook(_quantize_activation)
    return tuple(name for name, _ in linears)


def _quantize_activation(_layer: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return (fake_int8(args[0]), *args[1:])


def main() -> None:
    """Run the normal LeRobot eval with simulated action-expert W8A8."""
    make_policy = lerobot_eval.make_policy

    @wraps(make_policy)
    def make_quantized_policy(*args, **kwargs):
        policy = make_policy(*args, **kwargs)
        if not isinstance(policy, PI05Policy) or policy.config.compile_model:
            raise ValueError("The W8A8 accuracy experiment requires eager Pi0.5.")
        selected = simulate_action_expert_w8a8(policy)
        LOGGER.info("Simulating action-expert W8A8 on %d attention and MLP linears: %s", len(selected), selected)
        return policy

    lerobot_eval.make_policy = make_quantized_policy
    try:
        lerobot_eval.main()
    finally:
        lerobot_eval.make_policy = make_policy


if __name__ == "__main__":
    main()

from types import SimpleNamespace

import torch
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from torch import nn

from piquant.adarms import _CastInput, _StepTable, _adaptive_norms, cast_adarms, precompute_adarms

WIDTH, ACTION_DIM, CHUNK, STEPS = 16, 4, 3, 10


class ToyExpert(nn.Module):
    """The pieces of PI05Pytorch the rewrite touches: the timestep path and a few adaRMS norms."""

    embed_suffix = PI05Pytorch.embed_suffix
    _apply_checkpoint = PI05Pytorch._apply_checkpoint

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(chunk_size=CHUNK, min_period=4e-3, max_period=4.0)
        self.gradient_checkpointing_enabled = False
        self.action_in_proj = nn.Linear(ACTION_DIM, WIDTH)
        self.time_mlp_in = nn.Linear(WIDTH, WIDTH)
        self.time_mlp_out = nn.Linear(WIDTH, WIDTH)
        norms = nn.ModuleList(
            [PiGemmaRMSNorm(WIDTH, cond_dim=WIDTH), PiGemmaRMSNorm(WIDTH, cond_dim=WIDTH), PiGemmaRMSNorm(WIDTH)]
        )
        for norm in norms[:2]:
            nn.init.normal_(norm.dense.weight)  # zero-initialised by default, which would hide a wrong table
        self.paligemma_with_expert = SimpleNamespace(gemma_expert=norms)

    @torch.no_grad()
    def normed_over_schedule(self, x):
        """Each adaRMS norm's output and gate at every timestep, visited by LeRobot's own sampling loop."""
        outputs = []

        def denoise_fn(x_t, timestep):
            cond = self.embed_suffix(x_t, timestep)[3]
            outputs.append([norm(x, cond=cond) for norm in _adaptive_norms(self)])
            return torch.zeros_like(x_t)

        euler_integrate(denoise_fn, torch.zeros(1, CHUNK, ACTION_DIM), STEPS)
        return outputs


def test_adaptive_norms_skips_plain_norms():
    assert len(_adaptive_norms(ToyExpert())) == 2


def test_precomputed_norms_match_original_at_every_sampling_step():
    torch.manual_seed(0)
    model = ToyExpert()
    x = torch.randn(1, CHUNK, WIDTH)
    before = model.normed_over_schedule(x)

    precompute_adarms(model, STEPS)
    after = model.normed_over_schedule(x)

    assert all(isinstance(norm.dense, _StepTable) for norm in _adaptive_norms(model))
    assert len(after) == STEPS
    for step_before, step_after in zip(before, after, strict=True):
        for (out_b, gate_b), (out_a, gate_a) in zip(step_before, step_after, strict=True):
            assert torch.equal(out_a, out_b)
            assert torch.equal(gate_a, gate_b)


def test_precomputed_steps_differ_from_each_other():
    model = ToyExpert()
    precompute_adarms(model, STEPS)

    rows = _adaptive_norms(model)[0].dense.rows
    assert rows.shape == (STEPS, 3 * WIDTH)
    assert len(torch.unique(rows, dim=0)) == STEPS


def test_cast_adarms_stores_dense_in_bf16_and_keeps_the_output_dtype():
    torch.manual_seed(0)
    model = ToyExpert()
    x = torch.randn(1, CHUNK, WIDTH)
    before = model.normed_over_schedule(x)

    cast_adarms(model, torch.bfloat16)
    after = model.normed_over_schedule(x)

    assert all(isinstance(norm.dense, _CastInput) for norm in _adaptive_norms(model))
    assert all(norm.dense.linear.weight.dtype == torch.bfloat16 for norm in _adaptive_norms(model))
    for (out_b, _), (out_a, _) in zip(before[5], after[5], strict=True):
        assert out_a.dtype == out_b.dtype
        torch.testing.assert_close(out_a, out_b, rtol=0.05, atol=0.05)

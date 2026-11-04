from dataclasses import replace

import pytest

from piquant.roofline import (
    FORMATS,
    GPUS,
    PI05_ACTION_EXPERT,
    PI05_LIBERO,
    Gpu,
    Workload,
    latency_floor,
    stage_costs,
)

# Ma et al., "Running VLAs at Real-time Speed", Table 2, Roofline column: https://arxiv.org/html/2510.26742v1#S4.T2
# Expected values are its Vision Encoder, LLM and Action Expert subtotals. The paper never states its peak FLOP/s;
# 182.8 TFLOPS is back-solved from its GEMM rows (e.g. 512x2048x32768 at 6.391 ms), above the 4090 datasheet 165.2.
# int8_ops_per_s is unused here: a 4 x BF16 placeholder.
PAPER_4090 = Gpu(bf16_flops_per_s=182.8e12, int8_ops_per_s=731.2e12, bandwidth_bytes_per_s=1008e9)
PAPER_PI0 = Workload(cameras=2, text_tokens=0, action_tokens=64)  # Table 2 assumes two views and no prompt
PI0_ACTION_EXPERT = replace(PI05_ACTION_EXPERT, adaptive_norm=False)
PAPER_DENOISING_STEPS = 10
BF16 = FORMATS["BF16"]


def test_reproduces_published_pi0_roofline_on_4090_within_5_percent():
    stages = stage_costs(PAPER_PI0, PI0_ACTION_EXPERT)
    vision_encoder_ms = latency_floor(stages.vision_encoder, PAPER_4090, BF16).ms
    llm_ms = latency_floor(stages.llm, PAPER_4090, BF16).ms
    action_expert_ms = PAPER_DENOISING_STEPS * latency_floor(stages.action_expert, PAPER_4090, BF16).ms

    assert vision_encoder_ms == pytest.approx(2.485, rel=0.05)
    assert llm_ms == pytest.approx(10.727, rel=0.05)
    assert action_expert_ms == pytest.approx(6.486, rel=0.05)


def test_pi05_libero_on_3090_ti_bf16_matches_lerobot_as_run():
    stages = stage_costs(PI05_LIBERO, PI05_ACTION_EXPERT)
    gpu = GPUS["RTX 3090 Ti"]
    vision_encoder = latency_floor(stages.vision_encoder, gpu, BF16)
    llm = latency_floor(stages.llm, gpu, BF16)
    action_expert = latency_floor(stages.action_expert, gpu, BF16)

    assert vision_encoder.ms == pytest.approx(8.20, abs=0.01)
    assert llm.ms == pytest.approx(46.95, abs=0.01)
    assert action_expert.ms == pytest.approx(0.86, abs=0.01)
    assert vision_encoder.compute_bound
    assert llm.compute_bound
    assert not action_expert.compute_bound

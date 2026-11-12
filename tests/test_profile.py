import argparse

import pytest
import torch
from torch import nn

from piquant.profile import _floors, _positive_int, _roofline_gpu, _row_floors, _weight_bytes
from piquant.timing import Prefix


def test_floors_use_tf32_for_vision_and_bf16_elsewhere():
    floor = _floors(Prefix(image_slots=3, blank_image_slots=1, tokens_per_image=256, text_tokens=200), 50, "RTX 3090 Ti")

    assert floor["vision encoder"] == pytest.approx(16.40, abs=0.01)
    assert floor["llm prefix"] == pytest.approx(46.95, abs=0.01)
    assert floor["action expert"] == pytest.approx(0.86, abs=0.01)


def test_row_floors_count_the_expert_once_per_step():
    rows = _row_floors({"vision encoder": 16.0, "llm prefix": 47.0, "action expert": 0.5}, steps=10)

    assert rows["step"] == 0.5
    assert rows["denoising"] == rows["action expert"] == 5.0
    assert rows["total"] == 16.0 + 47.0 + 5.0


def test_positive_int_rejects_zero():
    assert _positive_int("1") == 1
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")


def test_weight_bytes_groups_by_dtype_and_filters_by_name():
    model = nn.ModuleDict({"proj": nn.Linear(4, 4, bias=False), "norm": nn.ModuleDict({"dense": nn.Linear(4, 12)})})
    model["proj"].to(torch.bfloat16)

    assert _weight_bytes(model) == {torch.bfloat16: 16 * 2, torch.float32: (48 + 12) * 4}
    assert _weight_bytes(model, select=lambda name: "norm.dense" in name) == {torch.float32: (48 + 12) * 4}


def test_roofline_gpu_matches_device_names_exactly():
    assert _roofline_gpu("NVIDIA GeForce RTX 5080") == "RTX 5080"
    assert _roofline_gpu("NVIDIA GeForce RTX 3090 Ti") == "RTX 3090 Ti"
    assert _roofline_gpu("NVIDIA GeForce RTX 5080 Laptop GPU") is None
    assert _roofline_gpu("NVIDIA GeForce RTX 3090") is None
    assert _roofline_gpu("NVIDIA H100 80GB HBM3") is None


def test_floor_without_adarms_weights_is_lower_for_the_action_expert_only():
    prefix = Prefix(image_slots=3, blank_image_slots=1, tokens_per_image=256, text_tokens=200)

    with_adarms = _floors(prefix, 50, "RTX 3090 Ti")
    without = _floors(prefix, 50, "RTX 3090 Ti", adarms_weights=False)

    assert without["action expert"] < with_adarms["action expert"]
    assert without["vision encoder"] == with_adarms["vision encoder"]
    assert without["llm prefix"] == with_adarms["llm prefix"]

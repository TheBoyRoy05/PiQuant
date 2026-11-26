from types import SimpleNamespace

import torch
from torch import nn

from piquant.w8a8 import fake_int8, simulate_w8a8


def test_fake_int8_scales_each_row_and_preserves_zero():
    values = torch.tensor([[127.0, -1.0, 0.0], [254.0, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.bfloat16)

    expected = torch.tensor([[127.0, -1.0, 0.0], [254.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=values.dtype)
    torch.testing.assert_close(fake_int8(values), expected, rtol=0, atol=0)


def test_prefix_quantization_keeps_the_expert_in_checkpoint_precision():
    def backbone():
        projection = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.tensor([[254.0, 1.0]]))
        return nn.Sequential(nn.ModuleDict({"mlp": nn.ModuleDict({"up_proj": projection})}))

    prefix, expert = backbone(), backbone()
    model = SimpleNamespace(
        gemma_expert=SimpleNamespace(model=expert),
        paligemma=SimpleNamespace(model=SimpleNamespace(language_model=prefix)),
    )
    policy = SimpleNamespace(model=SimpleNamespace(paligemma_with_expert=model))

    assert simulate_w8a8(policy, "prefix") == ("0.mlp.up_proj",)
    inputs = torch.tensor([[0.01, 10.0]])
    torch.testing.assert_close(prefix[0]["mlp"]["up_proj"](inputs), torch.tensor([[0.0]]))
    torch.testing.assert_close(expert[0]["mlp"]["up_proj"](inputs), torch.tensor([[12.54]]))

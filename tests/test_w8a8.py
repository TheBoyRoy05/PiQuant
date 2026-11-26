import torch

from piquant.w8a8 import fake_int8


def test_fake_int8_scales_each_row_and_preserves_zero():
    values = torch.tensor([[127.0, -1.0, 0.0], [254.0, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.bfloat16)

    expected = torch.tensor([[127.0, -1.0, 0.0], [254.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=values.dtype)
    torch.testing.assert_close(fake_int8(values), expected, rtol=0, atol=0)

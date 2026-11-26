import torch

from src.models.baseline_3d_conv.model import SimpleBaselineB


def test_baseline_output_shape():
    model = SimpleBaselineB(in_channels=5, time_steps=6)
    output = model(torch.randn(2, 5, 6, 32, 32))
    assert output.shape == (2, 1, 1, 32, 32)

import torch

from src.models.tiny_3d_unet.model import Tiny3DUNet


def test_tiny_3d_unet_output_shape():
    model = Tiny3DUNet()
    output = model(torch.randn(2, 5, 6, 32, 32))
    assert output.shape == (2, 1, 1, 32, 32)

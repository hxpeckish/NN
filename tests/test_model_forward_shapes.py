import torch

from ndf.models.ft_jnf import FTJNF


def test_model_forward_shapes():
    model = FTJNF(num_mics=4)
    x = torch.randn(2, 5, 8, 8)
    out = model(x)
    assert out.shape == (2, 5, 8, 2)
    assert torch.isfinite(out).all()

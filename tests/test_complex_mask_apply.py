import torch

from ndf.models.complex_mask import apply_complex_mask


def test_complex_mask_apply():
    mask = torch.tensor([[[[1.0, 0.0]]]])
    ref = torch.tensor([[[[2.0, 3.0]]]])
    out = apply_complex_mask(mask, ref)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref)

    mask = torch.tensor([[[[0.0, 1.0]]]])
    ref = torch.tensor([[[[2.0, 3.0]]]])
    out = apply_complex_mask(mask, ref)
    expected = torch.tensor([[[[-3.0, 2.0]]]])
    assert torch.allclose(out, expected)

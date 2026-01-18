import torch

from ndf.losses.l1_norm_batch import BatchNormalizedL1


def test_loss_finite():
    loss_fn = BatchNormalizedL1()
    pred = torch.randn(2, 5, 8, 2)
    target = torch.randn(2, 5, 8, 2)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)

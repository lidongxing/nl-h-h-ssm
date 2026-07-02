"""PH-Scan reference vs finite / shape checks (Triton parity later)."""

import torch

from csrc.ph_scan_kernel import ph_scan_reference


def test_ph_scan_shape():
    T, D = 5, 3
    x = torch.randn(1, T, D) * 0.1
    a = torch.sigmoid(torch.randn(1, T, D))
    b = torch.sigmoid(torch.randn(1, T, D))
    c = torch.tensor(0.5)
    y = ph_scan_reference(x, a, b, None, c, dim=1)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

"""Möbius / Poincaré alignment smoke tests."""

import torch

from nlh_ssm.ops import hyperbolic as H


def test_mobius_add_small_c():
    x = torch.randn(2, 4) * 0.05
    y = torch.randn(2, 4) * 0.05
    c = torch.tensor(1e-12)
    out = H.mobius_add(x, y, c)
    assert torch.allclose(out, x + y, atol=1e-5)


def test_exp_log_roundtrip():
    c = torch.tensor(1.0)
    v = torch.randn(3, 5) * 0.2
    y = H.expmap0(v, c)
    v2 = H.logmap0(y, c)
    assert torch.allclose(v, v2, atol=1e-4)

import torch

from nlh_ssm.metrics import (
    acd,
    acd_geometry_rmse,
    crps_gaussian,
    mat_score,
    rmsse,
    smape,
    tw_mse,
)


def test_smape_zero_when_equal():
    y = torch.tensor([[1.0, 2.0, 3.0]])
    assert float(smape(y, y)) == 0.0


def test_rmsse_zero_when_equal():
    y_pred = torch.tensor([[1.0, 2.0, 3.0]])
    y_true = torch.tensor([[1.0, 2.0, 3.0]])
    y_train = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    assert float(rmsse(y_pred, y_true, y_train)) == 0.0


def test_tw_mse_identity_cov_trace_non_negative():
    y_pred = torch.tensor([[1.0, 2.0], [1.0, 1.0], [2.0, 2.0]])
    y_true = torch.tensor([[1.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
    val = tw_mse(y_pred, y_true)
    assert float(val) >= 0.0


def test_crps_gaussian_near_zero_when_perfect_and_tiny_sigma():
    y_true = torch.tensor([0.0, 1.0, -1.0])
    mu = y_true.clone()
    sigma = torch.full_like(y_true, 1e-4)
    val = crps_gaussian(y_true, mu, sigma)
    assert float(val) < 1e-3


def test_mat_score_basic():
    val = mat_score(delta_metric=2.0, peak_vram_gb=4.0)
    assert torch.isclose(val, torch.tensor(0.5), atol=1e-6)


def test_acd_paper_finite():
    pts = torch.randn(2, 5, 4) * 0.01
    depths = torch.arange(5, dtype=torch.float32)
    exp_dist = (depths.unsqueeze(0) - depths.unsqueeze(1)).abs().unsqueeze(0).repeat(2, 1, 1)
    exp_dist = exp_dist + 0.5  # avoid zero tree distance (paper ACD skips D_T=0)
    val = acd(pts, exp_dist, c=1.0)
    assert torch.isfinite(val)


def test_acd_geometry_rmse_finite():
    pts = torch.randn(1, 4, 3) * 0.01
    depths = torch.arange(4, dtype=torch.float32)
    exp_dist = (depths.unsqueeze(0) - depths.unsqueeze(1)).abs().unsqueeze(0)
    val = acd_geometry_rmse(pts, exp_dist, c=1.0)
    assert torch.isfinite(val)


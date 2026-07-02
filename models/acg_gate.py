"""
Adaptive Curvature Gating (ACG) — thin facade.

The full ACG path lives inside ``nlh_ssm.modules.NLHBlock`` (``curv_proj``, ``c_base``,
``depth_gain_raw``) and is fused with PH-Scan in ``csrc.ph_scan_kernel.ph_scan_fused_acg``.

This module documents the public surface for papers / Table 4; import the block for training.
"""

from __future__ import annotations

import torch.nn as nn

from nlh_ssm.modules.nlh_block import NLHBlock

__all__ = ["NLHBlock"]

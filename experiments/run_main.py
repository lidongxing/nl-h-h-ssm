"""
YAML-driven launcher for ``run_exp.py`` (single dataset / model run).

Example:
  python experiments/run_main.py --config experiments/configs/m5.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install PyYAML: pip install pyyaml") from e

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install torch to validate GPU config.") from e

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.launch_device import parse_device_field


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    model = str(cfg["model"])
    dataset_path = str(cfg["dataset_path"])
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("lr", 1e-3))
    nlh_hparams = cfg.get("nlh_hparams_file")
    nlh_path: Path | None = None
    if nlh_hparams:
        nlh_path = Path(str(nlh_hparams))
        if not nlh_path.is_absolute():
            nlh_path = (_ROOT / nlh_path).resolve()

    run_exp = _ROOT / "experiments" / "run_exp.py"
    launch = parse_device_field(cfg.get("device"))

    if launch.mode == "ddp":
        ids = launch.cuda_indices
        assert ids is not None
        if not torch.cuda.is_available():
            raise SystemExit("YAML lists multiple GPUs but CUDA is not available.")
        for i in ids:
            if i < 0 or i >= torch.cuda.device_count():
                raise SystemExit(
                    f"Invalid GPU index {i} (visible device count={torch.cuda.device_count()}): {list(ids)}"
                )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(ids)}",
            str(run_exp),
            "--model",
            model,
            "--dataset",
            dataset_path,
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--cuda_indices",
            ",".join(str(i) for i in ids),
        ]
    else:
        cmd = [
            sys.executable,
            str(run_exp),
            "--model",
            model,
            "--dataset",
            dataset_path,
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
        ]
        if launch.device_arg:
            cmd.extend(["--device", launch.device_arg])

    if nlh_path is not None:
        cmd.extend(["--nlh_hparams_file", str(nlh_path)])

    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=str(_ROOT))
    except subprocess.CalledProcessError as e:
        print(
            "\n[run_main] run_exp.py failed (exit "
            f"{e.returncode}). Common causes: invalid --device vs "
            "torch.cuda.device_count(), missing data/processed/*.csv, or "
            "nlh_ssm not importable (run: pip install -e . in this venv).\n"
            "Re-run from repo root to see the full Python traceback:\n  cd "
            f"{_ROOT}\n  "
            + " ".join(cmd),
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()

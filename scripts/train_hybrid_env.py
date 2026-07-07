#!/usr/bin/env python3
"""Launch train/train.py with the current-host Motus runtime."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


DEFAULT_WAM_ROOT = Path(
    os.environ.get(
        "MOTUS_WAM_ROOT",
        Path.home() / "storage2/users/hnliu_data/wam_accel",
    )
)
MMLAB07_SITE = Path(
    os.environ.get(
        "MOTUS_RUNTIME_SITE",
        DEFAULT_WAM_ROOT / "envs/motus_env_mmlab07/lib/python3.10/site-packages",
    )
)
COMPAT_SITE = Path(
    os.environ.get(
        "MOTUS_CURRENTHOST_COMPAT_SITE",
        Path.home() / ".local/lib/python3.10/site-packages",
    )
)


def _dedup_prepend(paths: list[str]) -> None:
    wanted = [str(Path(p)) for p in paths]
    seen = set(wanted)
    sys.path[:] = wanted + [p for p in sys.path if p not in seen]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Import CUDA-compatible torch/torchvision first. The current-host runtime
    # still keeps those wheels in the explicit compatibility site.
    _dedup_prepend([str(COMPAT_SITE), str(MMLAB07_SITE), str(repo_root)])
    import torch  # noqa: F401
    import torchvision  # noqa: F401

    _dedup_prepend([str(repo_root), str(COMPAT_SITE), str(MMLAB07_SITE)])
    os.chdir(repo_root)
    sys.argv = [str(repo_root / "train" / "train.py"), *sys.argv[1:]]
    runpy.run_path(str(repo_root / "train" / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()

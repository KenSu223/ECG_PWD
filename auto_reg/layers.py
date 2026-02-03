"""Compatibility shim; forwards legacy imports and script execution to `ecg_pwd.auto_reg.layers`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

if __name__ == "__main__":
    runpy.run_module("ecg_pwd.auto_reg.layers", run_name="__main__")
else:
    from ecg_pwd.auto_reg.layers import *  # noqa: F401,F403

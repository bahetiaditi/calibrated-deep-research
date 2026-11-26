#!/usr/bin/env python3
"""§7.3 entry point: runs benchmark.run_eval for the ablations configuration(s).
Arguments pass through, e.g. `--split calibration --limit 4`."""
import sys

from benchmark.run_eval import main
from src.eval.configs import ABLATIONS, BASELINES, CONFIGS  # noqa: F401

if __name__ == "__main__":
    for name in ABLATIONS:
        code = main(["--config", name, *sys.argv[1:]])
        if code: sys.exit(code)

#!/usr/bin/env python3
"""§7.3 entry point: runs benchmark.run_eval for the baselines configuration(s).
Arguments pass through, e.g. `--split calibration --limit 4`."""
import sys

from benchmark.run_eval import main
from src.eval.configs import ABLATIONS, BASELINES, CONFIGS  # noqa: F401

if __name__ == "__main__":
    for name in BASELINES:
        if not CONFIGS[name].available:
            print(f"skipping {name}: {CONFIGS[name].note}"); continue
        code = main(["--config", name, *sys.argv[1:]])
        if code: sys.exit(code)

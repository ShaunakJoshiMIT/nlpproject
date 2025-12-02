#!/usr/bin/python3 python

"""

"""

import json
import logging
from pathlib import Path

from exp_cla_finetune import experiments as exp_cla
from exp_generation import experiments as exp_gen


if __name__ == "__main__":
    # Number of decimals when printing metrics.
    # For TSE metrics we keep the paper's scaling (×10^-3) but use higher precision.
    NB_DECIMALS = 5
    logger = logging.getLogger("res")
    (fh := logging.FileHandler(Path("runs", "all_metrics.log"))).setLevel(logging.DEBUG)
    (sh := logging.StreamHandler()).setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)

    for exp in exp_cla + exp_gen:
        logger.debug(f"\n{exp.name}")
        metrics = {}
        for baseline in exp.baselines:
            if not (baseline.run_path / "test_results.json").is_file():
                continue
            with open(baseline.run_path / "test_results.json") as file:
                results = json.load(file)
            # round results
            for key, val in results.items():
                if isinstance(val, float):
                    if "tse" in key.split("_"):
                        # Scale TSE by 1e3 (as in the paper) and keep more precision
                        results[key] = round(val * 1000, NB_DECIMALS)
                    else:
                        results[key] = round(val, NB_DECIMALS)

            logger.debug(f"{baseline.name} - {results}")

    # Additionally, print a compact TSE table for generative experiments on Maestro TSD,
    # mimicking the style of Table 2 in the paper (values scaled by 1e-3).
    print("\n=== TSE summary (×10⁻³) for Maestro / TSD ===")
    print("Baseline\tTSEtype\tTSEndup\tTSEtime")
    for exp in exp_gen:
        # Restrict to Maestro + TSD experiments (your current setup)
        if exp.dataset != "Maestro":
            continue
        if not exp.name.endswith("TSD"):
            continue
        for baseline in exp.baselines:
            path = baseline.run_path / "test_results.json"
            if not path.is_file():
                continue
            with open(path) as file:
                res = json.load(file)
            t_type = float(res.get("tse_type", 0.0)) * 1000.0
            t_ndup = float(res.get("tse_ndup", 0.0)) * 1000.0
            t_time = float(res.get("tse_time", 0.0)) * 1000.0
            print(f"{baseline.name}\t{t_type:.3f}\t{t_ndup:.3f}\t{t_time:.3f}")

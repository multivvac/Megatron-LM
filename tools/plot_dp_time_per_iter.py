#!/usr/bin/env python3
"""
GPUs vs Time per Iteration (Data Parallelism)
--------------------------------------------
Plots average iteration time (ms) against number of GPUs for TP1-PP1 runs.

Usage:
    python3 tools/plot_dp_time_per_iter.py
    python3 tools/plot_dp_time_per_iter.py --logs-dir logs/plots --exclude-first-n 5 --output logs/plots/dp_time_per_iter.png
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_filename(filename: str) -> Dict[str, int] | None:
    match = re.match(r"gpt2-N(\d+)-G(\d+)-TP(\d+)-PP(\d+)_times\.csv", filename)
    if not match:
        return None
    return {
        "nodes": int(match.group(1)),
        "gpus_per_node": int(match.group(2)),
        "tp": int(match.group(3)),
        "pp": int(match.group(4)),
    }


def read_elapsed_ms(csv_path: Path) -> List[float]:
    times = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["elapsed_ms"]))
    return times


def main():
    parser = argparse.ArgumentParser(description="Plot GPUs vs time per iteration for TP1-PP1 logs")
    parser.add_argument("--logs-dir", type=str, default="logs/plots", help="Directory with *_times.csv files")
    parser.add_argument("--exclude-first-n", type=int, default=5, help="Exclude first N iterations (warmup)")
    parser.add_argument("--output", type=str, default="logs/plots/data_parallelism_time_per_iter.png", help="Output plot path")
    parser.add_argument("--title", type=str, default="GPUs vs Time per Iteration (TP1-PP1)", help="Plot title")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    csv_files = list(logs_dir.glob("*-TP1-PP1_times.csv"))
    if not csv_files:
        print(f"No TP1-PP1 CSV files found in {logs_dir}")
        return

    results = []
    for csv_path in csv_files:
        cfg = parse_filename(csv_path.name)
        if not cfg:
            continue
        total_gpus = cfg["nodes"] * cfg["gpus_per_node"]
        times = read_elapsed_ms(csv_path)
        if len(times) <= args.exclude_first_n:
            avg_time = np.mean(times)
        else:
            avg_time = np.mean(times[args.exclude_first_n:])
        results.append({"gpus": total_gpus, "avg_time_ms": avg_time, "filename": csv_path.name})
        print(f"{csv_path.name}: {total_gpus} GPUs, avg time = {avg_time:.1f} ms")

    if not results:
        print("No valid results found")
        return

    # sort by GPU count
    results.sort(key=lambda r: r["gpus"]) 
    gpus = [r["gpus"] for r in results]
    times_ms = [r["avg_time_ms"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(gpus, times_ms, marker="o", linewidth=2, markersize=8, color="#1f77b4")
    plt.xlabel("Number of GPUs", fontsize=12)
    plt.ylabel("Time per Iteration (ms)", fontsize=12)
    plt.title(args.title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xticks(gpus)

    # annotate points
    for g, t in zip(gpus, times_ms):
        plt.annotate(f"{t:.0f}", (g, t), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    # Ideal time scaling (1/g) relative to 1 GPU baseline
    if len(gpus) > 1:
        baseline_time = times_ms[0]
        baseline_gpus = gpus[0]
        ideal_times = [baseline_time * (baseline_gpus / g) for g in gpus]
        plt.plot(gpus, ideal_times, "--", alpha=0.6, color="gray", label="Ideal 1/G scaling")
        plt.legend()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output}")

    # Print time scaling efficiency (actual vs ideal)
    if len(gpus) > 1:
        print("\nTime Scaling Efficiency (lower is better):")
        baseline_time = times_ms[0]
        baseline_gpus = gpus[0]
        for g, t in zip(gpus, times_ms):
            ideal_t = baseline_time * (baseline_gpus / g)
            efficiency = (ideal_t / t) * 100.0
            print(f"  {g} GPUs: {efficiency:.1f}% of ideal time")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Data Parallelism: Speedup & Efficiency (TP1-PP1)
------------------------------------------------
Computes speedup(k) = time(1 GPU) / time(k GPUs) and
parallel efficiency(k) = speedup(k) / k for TP1-PP1 runs.
Generates two plots:
 - GPUs vs Speedup (ideal y=x reference)
 - GPUs vs Efficiency (%)

Usage:
    python3 tools/plot_dp_speedup_efficiency.py

Options:
    --logs-dir DIR           Directory containing *_times.csv (default: logs/plots)
    --exclude-first-n N      Exclude first N iterations when averaging time (default: 5)
    --out-speedup PATH       Output path for speedup plot (default: logs/plots/data_parallelism_speedup.png)
    --out-eff PATH           Output path for efficiency plot (default: logs/plots/data_parallelism_efficiency.png)
    --title-speedup TEXT     Title for speedup plot
    --title-eff TEXT         Title for efficiency plot
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_filename(filename: str) -> Dict[str, int] | None:
    m = re.match(r"gpt2-N(\d+)-G(\d+)-TP(\d+)-PP(\d+)_times\.csv", filename)
    if not m:
        return None
    return {
        "nodes": int(m.group(1)),
        "gpus_per_node": int(m.group(2)),
        "tp": int(m.group(3)),
        "pp": int(m.group(4)),
    }


def read_elapsed_ms(csv_path: Path) -> List[float]:
    times: List[float] = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["elapsed_ms"]))
    return times


def compute_avg_time(times: List[float], exclude_first_n: int) -> float:
    if len(times) <= exclude_first_n:
        return float(np.mean(times))
    return float(np.mean(times[exclude_first_n:]))


def main():
    parser = argparse.ArgumentParser(description="Plot speedup and efficiency for TP1-PP1 runs")
    parser.add_argument("--logs-dir", type=str, default="logs/plots")
    parser.add_argument("--exclude-first-n", type=int, default=5)
    parser.add_argument("--out-speedup", type=str, default="logs/plots/data_parallelism_speedup.png")
    parser.add_argument("--out-eff", type=str, default="logs/plots/data_parallelism_efficiency.png")
    parser.add_argument("--title-speedup", type=str, default="Speedup vs GPUs (TP1-PP1)")
    parser.add_argument("--title-eff", type=str, default="Parallel Efficiency vs GPUs (TP1-PP1)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    csv_files = list(logs_dir.glob("*-TP1-PP1_times.csv"))
    if not csv_files:
        print(f"No TP1-PP1 CSV files found in {logs_dir}")
        return

    # Gather avg times per GPU count
    data = []  # (gpus, avg_time_ms, filename)
    for csv_path in csv_files:
        cfg = parse_filename(csv_path.name)
        if not cfg:
            print(f"Skipping {csv_path.name}: cannot parse config")
            continue
        total_gpus = cfg["nodes"] * cfg["gpus_per_node"]
        times = read_elapsed_ms(csv_path)
        avg_ms = compute_avg_time(times, args.exclude_first_n)
        data.append((total_gpus, avg_ms, csv_path.name))
        print(f"{csv_path.name}: {total_gpus} GPUs, avg time = {avg_ms:.1f} ms")

    if not data:
        print("No valid data found")
        return

    # Sort by GPU count and extract baseline
    data.sort(key=lambda x: x[0])
    gpus = [d[0] for d in data]
    avg_times_ms = [d[1] for d in data]

    # Baseline: prefer 1 GPU if present; otherwise smallest GPU count
    baseline_idx = 0
    if 1 in gpus:
        baseline_idx = gpus.index(1)
    baseline_gpus = gpus[baseline_idx]
    baseline_time_ms = avg_times_ms[baseline_idx]

    # Compute speedup and efficiency
    speedups = [baseline_time_ms / t for t in avg_times_ms]
    efficiencies = [(s / g) * 100.0 for s, g in zip(speedups, gpus)]

    # Plot speedup
    plt.figure(figsize=(10, 6))
    plt.plot(gpus, speedups, marker="o", linewidth=2, label="Measured Speedup")
    # Ideal speedup line y = x (linear scaling)
    plt.plot(gpus, gpus, "--", color="gray", alpha=0.6, label="Ideal Speedup (y=x)")
    plt.xlabel("Number of GPUs", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    plt.title(args.title_speedup, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xticks(gpus)
    for g, s in zip(gpus, speedups):
        plt.annotate(f"{s:.2f}", (g, s), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    plt.legend()
    Path(args.out_speedup).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_speedup, dpi=300, bbox_inches="tight")
    print(f"\nSpeedup plot saved to {args.out_speedup}")

    # Plot efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(gpus, efficiencies, marker="o", linewidth=2, color="#2ca02c", label="Parallel Efficiency")
    plt.xlabel("Number of GPUs", fontsize=12)
    plt.ylabel("Efficiency (%)", fontsize=12)
    plt.title(args.title_eff, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xticks(gpus)
    plt.yticks(range(0, 101, 10))
    for g, e in zip(gpus, efficiencies):
        plt.annotate(f"{e:.1f}%", (g, e), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    # Optional: reference line at 100%
    plt.axhline(100, linestyle="--", color="gray", alpha=0.5)
    plt.legend()
    Path(args.out_eff).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_eff, dpi=300, bbox_inches="tight")
    print(f"Efficiency plot saved to {args.out_eff}")

    # Print table of results
    print("\nResults:")
    for g, t, s, e in zip(gpus, avg_times_ms, speedups, efficiencies):
        print(f"  {g:>2} GPUs: avg_time = {t:.1f} ms, speedup = {s:.2f}, efficiency = {e:.1f}%")


if __name__ == "__main__":
    main()

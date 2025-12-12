#!/usr/bin/env python3
"""
Plot Data Parallelism Scalability
----------------------------------
Generates a throughput vs. number of GPUs plot for data parallel training.

Usage:
    python3 tools/plot_data_parallelism.py --batch-size 128 --seq-length 512
    python3 tools/plot_data_parallelism.py --batch-size 128 --seq-length 512 --output plots/dp_scalability.png
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_filename(filename: str) -> Dict[str, int]:
    """Extract nodes, GPUs, TP, PP from filename like 'gpt2-N1-G4-TP1-PP1_times.csv'."""
    match = re.match(r"gpt2-N(\d+)-G(\d+)-TP(\d+)-PP(\d+)_times\.csv", filename)
    if not match:
        return None
    return {
        "nodes": int(match.group(1)),
        "gpus_per_node": int(match.group(2)),
        "tp": int(match.group(3)),
        "pp": int(match.group(4)),
    }


def read_csv_times(csv_path: Path) -> List[float]:
    """Read elapsed_ms column from CSV and return list of times."""
    times = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["elapsed_ms"]))
    return times


def compute_throughput(
    elapsed_ms: float, global_batch_size: int, sequence_length: int
) -> float:
    """Compute throughput in samples per second.
    
    Throughput = (global_batch_size * sequence_length) / (elapsed_ms / 1000)
    """
    elapsed_sec = elapsed_ms / 1000.0
    samples_per_sec = (global_batch_size * sequence_length) / elapsed_sec
    return samples_per_sec


def main():
    parser = argparse.ArgumentParser(
        description="Plot data parallelism scalability (TP1-PP1 configs only)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Global batch size used in training",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        required=True,
        help="Sequence length used in training",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/plots",
        help="Directory containing the CSV files (default: logs/plots)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/plots/data_parallelism_scalability.png",
        help="Output path for the plot (default: logs/plots/data_parallelism_scalability.png)",
    )
    parser.add_argument(
        "--exclude-first-n",
        type=int,
        default=5,
        help="Exclude first N iterations from averaging (warmup, default: 5)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Data Parallelism Scalability",
        help="Plot title",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Error: Directory {logs_dir} does not exist")
        return

    # Find all TP1-PP1 CSV files
    csv_files = list(logs_dir.glob("*-TP1-PP1_times.csv"))
    if not csv_files:
        print(f"No TP1-PP1 CSV files found in {logs_dir}")
        return

    results = []
    for csv_path in csv_files:
        config = parse_filename(csv_path.name)
        if not config:
            print(f"Skipping {csv_path.name} (unable to parse filename)")
            continue

        total_gpus = config["nodes"] * config["gpus_per_node"]
        times = read_csv_times(csv_path)

        if len(times) <= args.exclude_first_n:
            print(
                f"Warning: {csv_path.name} has only {len(times)} iterations, skipping warmup exclusion"
            )
            avg_time = np.mean(times)
        else:
            avg_time = np.mean(times[args.exclude_first_n :])

        throughput = compute_throughput(avg_time, args.batch_size, args.seq_length)

        results.append(
            {
                "gpus": total_gpus,
                "avg_time_ms": avg_time,
                "throughput": throughput,
                "config": config,
                "filename": csv_path.name,
            }
        )
        print(
            f"{csv_path.name}: {total_gpus} GPUs, avg time = {avg_time:.1f} ms, throughput = {throughput:.1f} samples/sec"
        )

    if not results:
        print("No valid data found")
        return

    # Sort by number of GPUs
    results.sort(key=lambda x: x["gpus"])

    # Extract data for plotting
    gpus = [r["gpus"] for r in results]
    throughputs = [r["throughput"] for r in results]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(gpus, throughputs, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Number of GPUs", fontsize=12)
    plt.ylabel("Throughput (samples/second)", fontsize=12)
    plt.title(args.title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xticks(gpus)

    # Add data labels
    for i, (g, t) in enumerate(zip(gpus, throughputs)):
        plt.annotate(
            f"{t:.0f}",
            (g, t),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # Calculate and display ideal scaling
    if len(gpus) > 1:
        baseline_throughput = throughputs[0]
        baseline_gpus = gpus[0]
        ideal_throughputs = [
            baseline_throughput * (g / baseline_gpus) for g in gpus
        ]
        plt.plot(
            gpus,
            ideal_throughputs,
            "--",
            alpha=0.5,
            color="gray",
            label="Ideal Linear Scaling",
        )
        plt.legend()

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")

    # Print scaling efficiency
    if len(gpus) > 1:
        print("\nScaling Efficiency:")
        baseline_throughput = throughputs[0]
        baseline_gpus = gpus[0]
        for i, (g, t) in enumerate(zip(gpus, throughputs)):
            ideal_t = baseline_throughput * (g / baseline_gpus)
            efficiency = (t / ideal_t) * 100
            print(f"  {g} GPUs: {efficiency:.1f}% efficient")


if __name__ == "__main__":
    main()

import re
import os
import csv
from pathlib import Path


ITER_LINE_RE = re.compile(
    r"\[\s*(\d{4}-\d{2}-\d{2} [^\]]+)\] iteration\s+(\d+)\/\s*(\d+) \|[\s\S]*?elapsed time per iteration \(ms\):\s*([0-9.]+)\s*\|[\s\S]*?learning rate:\s*([0-9.E+-]+)\s*\|[\s\S]*?lm loss:\s*([0-9.E+-]+)",
    re.MULTILINE,
)


def parse_log_file(log_path: Path):
    """Parse a Megatron-LM training log and extract per-iteration metrics.

    Returns a list of dicts with keys: iteration, total_iterations, elapsed_ms, lr, loss.
    """
    text = log_path.read_text(errors="ignore")
    rows = []
    for m in ITER_LINE_RE.finditer(text):
        # ts = m.group(1)  # timestamp present but not needed for plot
        iteration = int(m.group(2))
        total = int(m.group(3))
        elapsed_ms = float(m.group(4))
        lr = float(m.group(5))
        loss = float(m.group(6))
        rows.append(
            {
                "iteration": iteration,
                "total_iterations": total,
                "elapsed_ms": elapsed_ms,
                "learning_rate": lr,
                "loss": loss,
            }
        )
    return rows


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "total_iterations",
                "elapsed_ms",
                "learning_rate",
                "loss",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    logs_dir = Path("logs")
    out_dir = logs_dir / "plots"
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return

    log_files = list(logs_dir.glob("*.log"))
    if not log_files:
        print(f"No .log files found in {logs_dir}")
        return

    for log_path in log_files:
        rows = parse_log_file(log_path)
        if not rows:
            print(f"No iteration lines parsed from {log_path}")
            continue
        out_path = out_dir / f"{log_path.stem}_times.csv"
        write_csv(rows, out_path)
        print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

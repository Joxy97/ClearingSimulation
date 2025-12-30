from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_PATTERNS = {
    "total_dl": re.compile(r"^Total DL:\s+([-+]?\d+(?:\.\d+)?)$"),
    "clients_gain": re.compile(r"^Clients' gain:\s+([-+]?\d+(?:\.\d+)?)$"),
    "total_defaulted": re.compile(r"^Total defaulted clients:\s+(\d+)$"),
    "mean_qubo_size": re.compile(r"^Mean QUBO size .*:\s+([-+]?\d+(?:\.\d+)?)$"),
    "mean_qubo_time": re.compile(r"^Mean QUBO time .*:\s+([-+]?\d+(?:\.\d+)?)$"),
    "mean_qubo_energy": re.compile(r"^Mean QUBO energy:\s+([-+]?\d+(?:\.\d+)?)$"),
    "total_sim_time": re.compile(r"^Total simulation time .*:\s+([-+]?\d+(?:\.\d+)?)$"),
}


def _parse_summary(output: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in output.splitlines():
        line = line.strip()
        for key, pattern in SUMMARY_PATTERNS.items():
            if key in values:
                continue
            match = pattern.match(line)
            if match:
                values[key] = float(match.group(1))
    missing = [k for k in SUMMARY_PATTERNS if k not in values]
    if missing:
        raise ValueError(f"Missing summary fields: {', '.join(missing)}")
    return values


def _build_command(seed_market: int, seed_trade: int, M: int, T: int, qubo_solver: str) -> List[str]:
    return [
        sys.executable,
        "-u",
        os.path.join("scripts", "run_simulation.py"),
        "--stress",
        "--stress-level",
        "1.0",
        "--start-state",
        "0",
        "--init-position-scale",
        "150",
        "--init-position-density",
        "0.3",
        "--trade-base-scale",
        "10",
        "--trade-noise-scale",
        "2",
        "--trade-max-abs",
        "100",
        "--stress-day",
        "3",
        "--stress-return-scale",
        "10",
        "--stress-return-shift",
        "-0.05",
        "--M",
        str(M),
        "--T",
        str(T),
        "--qubo-solver",
        qubo_solver,
        "--seed-market",
        str(seed_market),
        "--seed-trade",
        str(seed_trade),
        "--log-path",
        "",
    ]


def _run_once(run_idx: int, seed_market: int, seed_trade: int, M: int, T: int, qubo_solver: str) -> Dict[str, float]:
    cmd = _build_command(seed_market, seed_trade, M, T, qubo_solver)
    env = os.environ.copy()
    if qubo_solver == "hybrid":
        env["CLSIM_RUN"] = str(run_idx)
        env["CLSIM_TIME_LIMIT"] = "10"
    else:
        env.pop("CLSIM_RUN", None)
        env.pop("CLSIM_TIME_LIMIT", None)
    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    output_lines: List[str] = []
    if proc.stdout is None:
        raise RuntimeError("Failed to capture process output.")
    for line in proc.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)
    returncode = proc.wait()
    output = "".join(output_lines)
    if returncode != 0:
        raise RuntimeError(
            f"Run {run_idx} failed with code {returncode}:\n"
            f"output:\n{output}"
        )
    return _parse_summary(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulation benchmark and export histogram dashboard.")
    parser.add_argument("--runs", type=int, default=100, help="Number of simulations to run.")
    parser.add_argument("--output", type=str, default="benchmark_dashboard.png", help="Output PNG path.")
    parser.add_argument("--seed-market", type=int, default=123, help="Base market seed.")
    parser.add_argument("--seed-trade", type=int, default=456, help="Base trade seed.")
    parser.add_argument("--M", type=int, default=1000, help="Number of clients.")
    parser.add_argument("--T", type=int, default=30, help="Number of simulation days.")
    parser.add_argument("--qubo-solver", type=str, default="sa", choices=["sa", "hybrid"], help="QUBO solver.")
    args = parser.parse_args()

    totals_dl: List[float] = []
    gains: List[float] = []
    total_defaulted: List[float] = []
    mean_qubo_sizes: List[float] = []
    mean_qubo_times: List[float] = []
    mean_qubo_energies: List[float] = []
    total_sim_times: List[float] = []

    for i in range(args.runs):
        seed_market = args.seed_market + i
        seed_trade = args.seed_trade + i
        print(f"Running simulation {i + 1}/{args.runs} (seed_market={seed_market}, seed_trade={seed_trade})")
        summary = _run_once(i + 1, seed_market, seed_trade, args.M, args.T, args.qubo_solver)
        totals_dl.append(summary["total_dl"])
        gains.append(summary["clients_gain"])
        total_defaulted.append(summary["total_defaulted"])
        mean_qubo_sizes.append(summary["mean_qubo_size"])
        mean_qubo_times.append(summary["mean_qubo_time"])
        mean_qubo_energies.append(summary["mean_qubo_energy"])
        total_sim_times.append(summary["total_sim_time"])

    metrics = [
        ("Total DL", totals_dl),
        ("Clients' gains", gains),
        ("Total defaulted clients", total_defaulted),
        ("Mean QUBO size", mean_qubo_sizes),
        ("Mean QUBO time (s)", mean_qubo_times),
        ("Mean QUBO energy", mean_qubo_energies),
        ("Total simulation time (s)", total_sim_times),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()
    for idx, (title, values) in enumerate(metrics):
        ax = axes[idx]
        ax.hist(values, bins=20, color="#94a3b8", edgecolor="#334155")
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("count")

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Simulation Benchmark Histograms (n={args.runs})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150)
    print(f"Saved dashboard to {args.output}")


if __name__ == "__main__":
    main()

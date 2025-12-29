from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from clearing.data_prep import generate_dataset, save_dataset_csv, save_quantizer  # noqa: E402
from clearing.market import make_default_params  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic market data and export RBM-ready bits.")
    parser.add_argument("--T", type=int, default=10000, help="Number of days to simulate.")
    parser.add_argument("--N", type=int, default=500, help="Number of instruments.")
    parser.add_argument("--S", type=int, default=3, help="Number of market states.")
    parser.add_argument("--F", type=int, default=2, help="Number of factors.")
    parser.add_argument("--K", type=int, default=4, help="Bits per instrument for quantization.")
    parser.add_argument("--q-low", type=float, default=0.001, help="Lower quantile for range.")
    parser.add_argument("--q-high", type=float, default=0.999, help="Upper quantile for range.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, or cuda.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Tensor dtype.")
    parser.add_argument("--out-csv", type=str, default="data/data.csv", help="CSV output path.")
    parser.add_argument("--out-quantizer", type=str, default="data/quantizer.pt", help="Quantizer output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    params = make_default_params(N=args.N, S=args.S, F=args.F, device=device, dtype=dtype)

    _Z, _R, _D_all, D_bits, q = generate_dataset(
        T=args.T,
        params=params,
        K=args.K,
        q_low=args.q_low,
        q_high=args.q_high,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    save_dataset_csv(D_bits, args.out_csv)
    if args.out_quantizer:
        save_quantizer(q, args.out_quantizer)

    print(f"Saved dataset to {args.out_csv}")
    if args.out_quantizer:
        print(f"Saved quantizer to {args.out_quantizer}")


if __name__ == "__main__":
    main()

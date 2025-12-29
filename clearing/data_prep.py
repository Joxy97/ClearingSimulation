from __future__ import annotations

from typing import Tuple

import torch

from .market import MarketParams, simulate_market
from .quantization import ReturnQuantizer, encode_d, fit_return_quantizer


def generate_dataset(
    T: int,
    params: MarketParams,
    *,
    K: int = 4,
    q_low: float = 0.001,
    q_high: float = 0.999,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ReturnQuantizer]:
    Z, R, D_all = simulate_market(T=T, params=params, device=device, dtype=dtype, seed=seed)
    q = fit_return_quantizer(R, K=K, q_low=q_low, q_high=q_high)
    D_bits_all = encode_d(D_all, S=params.S, q=q)
    return Z, R, D_all, D_bits_all, q


def save_dataset_csv(D_bits: torch.Tensor, out_path: str) -> None:
    import pandas as pd

    D_bits_np = D_bits.detach().cpu().numpy()
    df = pd.DataFrame(D_bits_np)
    df.to_csv(out_path, index=False)


def save_quantizer(q: ReturnQuantizer, out_path: str) -> None:
    torch.save(q, out_path)


def load_quantizer(path: str) -> ReturnQuantizer:
    try:
        q = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        q = torch.load(path, map_location="cpu")
    if not isinstance(q, ReturnQuantizer):
        raise ValueError("Loaded object is not a ReturnQuantizer")
    return q

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TradeParams:
    momentum_weight: float = 1.0
    gamma_exposure: float = 0.5

    base_scale: float = 10.0

    noise_scale: float = 0.5

    p_trade_when_zero: float = 0.10
    zero_trade_std: float = 1.0

    max_abs_trade: Optional[float] = None


@torch.no_grad()
def propose_trades(
    P: torch.Tensor,
    r_t: torch.Tensor,
    params: TradeParams,
    *,
    alive: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Returns:
      DeltaP: [M, N] proposed portfolio deltas (can be fractional).
    """
    if P.ndim != 2:
        raise ValueError("P must be [M, N].")
    if r_t.ndim != 1 or r_t.shape[0] != P.shape[1]:
        raise ValueError("r_t must be [N] matching P.shape[1].")

    device = P.device
    dtype = P.dtype
    M, N = P.shape

    if alive is None:
        alive = torch.ones(M, device=device, dtype=torch.bool)
    else:
        alive = alive.to(device=device, dtype=torch.bool)

    r = r_t.to(device=device, dtype=dtype)[None, :]

    signal = params.momentum_weight * r

    exposure_scale = (P.abs() + 1.0).pow(params.gamma_exposure)

    mu = params.base_scale * signal * exposure_scale

    std = params.noise_scale * (P.abs().sqrt() + 1.0)
    eps = torch.randn((M, N), device=device, dtype=dtype, generator=generator)
    DeltaP = mu + std * eps

    if params.p_trade_when_zero < 1.0:
        mask0 = (P == 0)
        if mask0.any():
            u = torch.rand((M, N), device=device, dtype=dtype, generator=generator)
            keep = u < float(params.p_trade_when_zero)
            DeltaP = torch.where(mask0 & (~keep), torch.zeros((), device=device, dtype=dtype), DeltaP)

            if params.zero_trade_std is not None and params.zero_trade_std > 0:
                eps0 = torch.randn((M, N), device=device, dtype=dtype, generator=generator) * float(params.zero_trade_std)
                DeltaP = torch.where(mask0 & keep, eps0, DeltaP)

    DeltaP = torch.where(alive[:, None], DeltaP, torch.zeros((), device=device, dtype=dtype))

    if params.max_abs_trade is not None:
        DeltaP = torch.clamp(DeltaP, -float(params.max_abs_trade), float(params.max_abs_trade))

    return DeltaP

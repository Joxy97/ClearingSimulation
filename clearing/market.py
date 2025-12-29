from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MarketParams:
    N: int
    S: int
    F: int

    Pz: torch.Tensor
    B: torch.Tensor
    sigma_f: torch.Tensor
    sigma_e: torch.Tensor

    rho: float = 0.0

    jump_p: Optional[torch.Tensor] = None
    jump_sigma: Optional[torch.Tensor] = None
    jump_mu: float = 0.0


def _sample_categorical(probs: torch.Tensor, generator: Optional[torch.Generator]) -> int:
    idx = torch.multinomial(probs, 1, replacement=True, generator=generator)
    return int(idx.item())


@torch.no_grad()
def generate_day(
    z_prev: int,
    r_prev: torch.Tensor,
    params: MarketParams,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Generate one day:
      d_t = [onehot(z_t), r_{t-1}, r_t]
    """
    device = r_prev.device
    dtype = r_prev.dtype

    z_t = _sample_categorical(params.Pz[z_prev], generator)

    f_t = torch.randn(params.F, device=device, dtype=dtype, generator=generator)
    f_t *= params.sigma_f[z_t]

    eps_t = torch.randn(params.N, device=device, dtype=dtype, generator=generator)
    eps_t *= params.sigma_e[z_t]

    r_raw = params.B @ f_t + eps_t

    if params.rho != 0.0:
        r_t = params.rho * r_prev + (1.0 - params.rho) * r_raw
    else:
        r_t = r_raw

    if params.jump_p is not None and params.jump_sigma is not None:
        u = torch.rand((), device=device, dtype=dtype, generator=generator)
        if u < params.jump_p[z_t]:
            jump = torch.randn(params.N, device=device, dtype=dtype, generator=generator)
            jump = jump * params.jump_sigma[z_t] + params.jump_mu
            r_t = r_t + jump

    state_oh = F.one_hot(
        torch.tensor(z_t, device=device),
        num_classes=params.S,
    ).to(dtype=dtype)

    d_t = torch.cat([state_oh, r_prev, r_t], dim=0)

    return z_t, r_t, d_t


def make_default_params(N: int = 5, S: int = 3, F: int = 2, device: str = "cuda", dtype: torch.dtype = torch.float32) -> MarketParams:
    if S != 3:
        raise ValueError("This setup is explicitly for S=3")

    Pz = torch.tensor(
        [
            [0.95, 0.04, 0.01],
            [0.10, 0.80, 0.10],
            [0.05, 0.20, 0.75],
        ],
        device=device,
        dtype=dtype,
    )

    B = torch.randn(N, F, device=device, dtype=dtype) / (F ** 0.5)

    sigma_f = torch.tensor(
        [
            [0.006] * F,
            [0.012] * F,
            [0.030] * F,
        ],
        device=device,
        dtype=dtype,
    )

    sigma_e = torch.zeros(S, N, device=device, dtype=dtype)
    sigma_e[0] = 0.004
    sigma_e[1] = 0.008
    sigma_e[2] = 0.020

    jump_p = torch.tensor([0.002, 0.006, 0.020], device=device, dtype=dtype)
    jump_sigma = torch.tensor([0.03, 0.05, 0.10], device=device, dtype=dtype)

    return MarketParams(
        N=N,
        S=S,
        F=F,
        Pz=Pz,
        B=B,
        sigma_f=sigma_f,
        sigma_e=sigma_e,
        rho=0.0,
        jump_p=jump_p,
        jump_sigma=jump_sigma,
    )


@torch.no_grad()
def simulate_market(
    T: int,
    params: MarketParams,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)

    N, S = params.N, params.S

    D = S + 2 * N
    D_all = torch.empty(T, D, device=device, dtype=dtype)
    Z = torch.empty(T, dtype=torch.long, device=device)
    R = torch.empty(T, N, device=device, dtype=dtype)

    z = 0
    r_prev = torch.zeros(N, device=device, dtype=dtype)

    for t in range(T):
        z, r_t, d_t = generate_day(z, r_prev, params, generator=g)
        Z[t] = z
        R[t] = r_t
        D_all[t] = d_t
        r_prev = r_t

    return Z, R, D_all

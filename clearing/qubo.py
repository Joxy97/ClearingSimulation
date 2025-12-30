from __future__ import annotations

import os
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import dimod


@torch.no_grad()
def build_bqm_accept_clients(
    deltaM: torch.Tensor,
    *,
    B: float,
    U: Optional[torch.Tensor] = None,
    alive: Optional[torch.Tensor] = None,
    lambda_budget: float = 1.0,
    eta_risk: float = 0.0,
    big_M: float = 1e6,
) -> dimod.BinaryQuadraticModel:
    """
    Returns a dimod BQM with variables {0..M-1} in BINARY domain.
    """
    if deltaM.ndim != 1:
        raise ValueError("deltaM must be [M].")
    M = deltaM.numel()

    d = deltaM.detach().cpu().to(torch.float64).numpy()
    d_pos = np.maximum(d, 0.0)
    d_neg = np.minimum(d, 0.0)

    if U is None:
        U_np = np.zeros(M, dtype=np.float64)
    else:
        U_np = U.detach().cpu().to(torch.float64).numpy()
    # Reward risk-reducing trades (negative deltaM) by default.
    U_np = U_np + (-d_neg)

    if alive is None:
        alive_np = np.ones(M, dtype=bool)
    else:
        alive_np = alive.detach().cpu().numpy().astype(bool)

    lam = float(lambda_budget)
    eta = float(eta_risk)
    B = float(B)

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    for i in range(M):
        lin_i = lam * (d_pos[i] ** 2) + (-2.0 * lam * B) * d_pos[i] + eta * d_pos[i] - U_np[i]

        if not alive_np[i]:
            lin_i += big_M

        bqm.add_variable(i, lin_i)

    for i in range(M):
        di = d_pos[i]
        if di == 0.0:
            continue
        for j in range(i + 1, M):
            q = 2.0 * lam * di * d_pos[j]
            if q != 0.0:
                bqm.add_interaction(i, j, q)

    bqm.offset += lam * (B ** 2)

    return bqm


@torch.no_grad()
def solve_bqm(
    bqm: dimod.BinaryQuadraticModel,
    *,
    method: Literal["sa", "hybrid"] = "sa",

    num_reads: int = 200,
    num_sweeps: int = 2000,
    seed: Optional[int] = None,

    time_limit: Optional[float] = None,

    out_device: str = "cpu",
    label: Optional[str] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Returns:
      x: [M] int64 tensor with 0/1 for variables 0..M-1
      energy: best energy found
    """
    if method == "sa":
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError as e:
            raise ImportError("Install SA solver: pip install dwave-neal") from e

        sampler = SimulatedAnnealingSampler()
        ss = sampler.sample(
            bqm,
            num_reads=int(num_reads),
            num_sweeps=int(num_sweeps),
            seed=seed,
        )

    elif method == "hybrid":
        token = os.getenv("DWAVE_API_KEY")
        if not token:
            raise ValueError("DWAVE_API_KEY environment variable is required for the hybrid solver.")
        try:
            from dwave.system import LeapHybridSampler
        except ImportError as e:
            raise ImportError("Install Leap solver: pip install dwave-system") from e

        sampler = LeapHybridSampler(token=token)
        if label is None:
            label = os.getenv("CLSIM_LABEL") or "ClSim"
        if time_limit is None:
            env_limit = os.getenv("CLSIM_TIME_LIMIT", "5")
            try:
                time_limit = float(env_limit)
            except ValueError as e:
                raise ValueError(f"Invalid CLSIM_TIME_LIMIT value: {env_limit}") from e
        ss = sampler.sample(bqm, time_limit=float(time_limit), label=label)

    else:
        raise ValueError(f"Unknown method {method}")

    best = ss.first
    energy = float(best.energy)

    vars_sorted = sorted(bqm.variables)
    x_np = np.array([int(best.sample[v]) for v in vars_sorted], dtype=np.int64)

    x = torch.from_numpy(x_np).to(device=out_device, dtype=torch.int64)
    return x, energy

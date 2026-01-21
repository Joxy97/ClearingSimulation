from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, Optional, Literal

import torch

from .counterparty import (
    CounterpartyNetwork,
    CounterpartyParams,
    compute_bilateral_exposures,
    propagate_default_losses,
)
from .logger import SimLogger
from .margin import margin_es
from .market import MarketParams, generate_day
from .qubo import build_bqm_accept_clients, solve_bqm
from .sampler import sample_scenarios_from_float
from .trading import TradeParams, propose_trades


@dataclass
class SimDayParams:
    alpha: float = 0.99
    budget_B: float = 0.0
    lambda_budget: float = 10.0
    eta_risk: float = 0.0
    utility_weight: float = 0.0
    qubo_solver: Literal["sa", "hybrid"] = "sa"
    sa_num_reads: int = 200
    sa_num_sweeps: int = 2000
    leap_time_limit: Optional[float] = None
    post_full_margin_daily: bool = True
    stress_day: Optional[int] = None
    stress_return_scale: float = 1.0
    stress_return_shift: float = 0.0


@torch.no_grad()
def simulate_day(
    *,
    P: torch.Tensor,
    W: torch.Tensor,
    C: torch.Tensor,
    default_loss: Optional[torch.Tensor] = None,
    alive: torch.Tensor,
    z_prev: int,
    r_prev: torch.Tensor,
    market_params: MarketParams,
    rbm_model,
    rbm_config: Dict[str, Any],
    quantizer,
    trade_params: TradeParams,
    sim_params: SimDayParams,
    counterparty_network: Optional[CounterpartyNetwork] = None,
    counterparty_params: Optional[CounterpartyParams] = None,
    g_market: Optional[torch.Generator] = None,
    g_trade: Optional[torch.Generator] = None,
    num_scenarios: int = 2000,
    burn_in: int = 500,
    thin: int = 10,
    device: Optional[str] = None,
    day: Optional[int] = None,
    logger: Optional[SimLogger] = None,
    log_scenarios: bool = False,
    return_scenarios: bool = False,
) -> Dict[str, Any]:
    """
    One full day step.
    """
    if P.ndim != 2:
        raise ValueError("P must be [M,N]")
    M, N = P.shape

    if W.shape != (M,):
        raise ValueError("W must be [M]")
    if C.shape != (M,):
        raise ValueError("C must be [M]")
    if default_loss is None:
        default_loss = torch.zeros_like(W)
    if default_loss.shape != (M,):
        raise ValueError("default_loss must be [M]")
    if alive.shape != (M,):
        raise ValueError("alive must be [M]")
    if r_prev.shape != (N,):
        raise ValueError("r_prev must be [N]")

    if device is None:
        device = P.device

    dtype = P.dtype

    P = P.to(device=device)
    W = W.to(device=device)
    C = C.to(device=device)
    default_loss = default_loss.to(device=device)
    alive = alive.to(device=device, dtype=torch.bool)
    r_prev = r_prev.to(device=device)

    d = 0 if day is None else int(day)

    if logger is not None:
        logger.log(day=d, phase="start", P=P, W=W, C=C, alive=alive, z_t=int(z_prev), r_t=r_prev, default_loss=default_loss)

    z_t, r_t, _d_t = generate_day(z_prev, r_prev, market_params, generator=g_market)
    if sim_params.stress_day is not None and d == int(sim_params.stress_day):
        r_t = r_t * float(sim_params.stress_return_scale) + float(sim_params.stress_return_shift)

    pnl = (P * r_t[None, :]).sum(dim=1)

    if logger is not None:
        logger.log(day=d, phase="market_move", P=P, W=W, C=C, alive=alive, z_t=int(z_t), r_t=r_t, pnl=pnl, default_loss=default_loss)

    W_settle = W + pnl
    F_pre = (W - C) + pnl
    C_after_f = C + torch.minimum(F_pre, torch.zeros_like(F_pre))
    default_loss_initial = (-C_after_f).clamp(min=0.0)
    C_after_f = C_after_f.clamp(min=0.0)

    # Counterparty loss propagation with default fund
    if counterparty_params is not None and counterparty_params.enabled and counterparty_network is not None:
        exposure_matrix = compute_bilateral_exposures(
            counterparty_network.contracts, r_t, M, str(device), P.dtype
        )
        counterparty_network.exposure_matrix = exposure_matrix

        default_loss_step, W_settle, C_after_f, df_used = propagate_default_losses(
            default_loss_initial, exposure_matrix, W_settle, C_after_f,
            counterparty_network.default_fund_contrib,
            counterparty_network.total_default_fund,
            alive, counterparty_params, logger, d, str(device), P.dtype
        )
        counterparty_network.total_default_fund -= df_used
    else:
        default_loss_step = default_loss_initial

    W_settle = torch.maximum(W_settle, torch.zeros_like(W_settle))
    default_loss = default_loss + default_loss_step
    default_loss_total = float(default_loss.sum().item())

    if logger is not None:
        logger.log(
            day=d,
            phase="settlement",
            P=P,
            W=W_settle,
            C=C_after_f,
            alive=alive,
            z_t=int(z_t),
            r_t=r_t,
            default_loss=default_loss,
            default_loss_total=default_loss_total,
        )

    R_scenarios = sample_scenarios_from_float(
        model=rbm_model,
        config=rbm_config,
        quantizer=quantizer,
        z_t=z_t,
        r_prev_float=r_t,
        num_samples=num_scenarios,
        burn_in=burn_in,
        thin=thin,
        device=str(device),
    )

    M_req_cur, var_cur = margin_es(P, R_scenarios, alpha=sim_params.alpha, return_var=True)

    # Add CVA margin for counterparty credit risk
    cva_margin = None
    if counterparty_params is not None and counterparty_params.enabled and counterparty_params.cva_enabled and counterparty_network is not None:
        if counterparty_network.exposure_matrix is not None:
            total_receivables = counterparty_network.exposure_matrix.sum(dim=1)
            cva_margin = counterparty_params.cva_multiplier * total_receivables.clamp(min=0.0)
            M_req_cur = M_req_cur + cva_margin

    if logger is not None:
        logger.log(
            day=d,
            phase="margin",
            P=P,
            W=W_settle,
            C=C_after_f,
            alive=alive,
            z_t=int(z_t),
            r_t=r_t,
            M_req_cur=M_req_cur,
            cva_margin=cva_margin,
            counterparty_exposure=counterparty_network.exposure_matrix if counterparty_network is not None else None,
            default_fund_balance=counterparty_network.total_default_fund if counterparty_network is not None else None,
            var_cur=var_cur,
            default_loss=default_loss,
            default_loss_total=default_loss_total,
            R_scenarios=(R_scenarios if log_scenarios else None),
        )

    F_available = (W_settle - C_after_f).clamp(min=0.0)
    need = (M_req_cur - C_after_f).clamp(min=0.0)
    can_topup = F_available >= need
    default_now = alive & (~can_topup)
    alive_new = alive & can_topup

    if sim_params.post_full_margin_daily:
        C_new = torch.where(alive_new, C_after_f + need, torch.zeros_like(C_after_f))
    else:
        C_new = torch.where(alive_new, C_after_f + need, torch.zeros_like(C_after_f))

    P_liq = torch.where(alive_new[:, None], P, torch.zeros_like(P))
    W_new = W_settle

    if logger is not None:
        logger.log(
            day=d,
            phase="default",
            P=P_liq,
            W=W_new,
            C=C_new,
            alive=alive_new,
            z_t=int(z_t),
            r_t=r_t,
            M_req_cur=M_req_cur,
            var_cur=var_cur,
            default_loss=default_loss,
            default_loss_total=default_loss_total,
        )

    DeltaP = propose_trades(P_liq, r_t, trade_params, alive=alive_new, generator=g_trade)
    P_tent = P_liq + DeltaP

    M_req_tent, var_tent = margin_es(P_tent, R_scenarios, alpha=sim_params.alpha, return_var=True)
    deltaM = M_req_tent - M_req_cur

    if logger is not None:
        logger.log(
            day=d,
            phase="trades",
            P=P_liq,
            W=W_new,
            C=C_new,
            alive=alive_new,
            z_t=int(z_t),
            r_t=r_t,
            M_req_cur=M_req_cur,
            M_req_tent=M_req_tent,
            var_cur=var_cur,
            var_tent=var_tent,
            deltaM=deltaM,
            DeltaP=DeltaP,
            default_loss=default_loss,
            default_loss_total=default_loss_total,
            R_scenarios=(R_scenarios if log_scenarios else None),
        )

    if sim_params.utility_weight != 0.0:
        U = sim_params.utility_weight * DeltaP.abs().sum(dim=1)
    else:
        U = None

    bqm = build_bqm_accept_clients(
        deltaM=deltaM,
        B=sim_params.budget_B,
        U=U,
        alive=alive_new,
        lambda_budget=sim_params.lambda_budget,
        eta_risk=sim_params.eta_risk,
    )

    qubo_num_vars = len(bqm.variables)
    qubo_num_interactions = len(bqm.quadratic)
    t0 = time.perf_counter()
    label = None
    if sim_params.qubo_solver == "hybrid":
        sim_id = os.getenv("CLSIM_RUN")
        if sim_id:
            label = f"ClSim{sim_id}_day{d}"
    x, energy = solve_bqm(
        bqm,
        method=sim_params.qubo_solver,
        num_reads=sim_params.sa_num_reads,
        num_sweeps=sim_params.sa_num_sweeps,
        time_limit=sim_params.leap_time_limit,
        out_device=str(device),
        label=label,
    )
    solve_time_s = time.perf_counter() - t0

    if logger is not None:
        logger.log(
            day=d,
            phase="decision",
            P=P_liq,
            W=W_new,
            C=C_new,
            alive=alive_new,
            z_t=int(z_t),
            r_t=r_t,
            M_req_cur=M_req_cur,
            M_req_tent=M_req_tent,
            deltaM=deltaM,
            DeltaP=DeltaP,
            x=x,
            qubo_energy=energy,
            qubo_num_vars=qubo_num_vars,
            qubo_num_interactions=qubo_num_interactions,
            qubo_solver=sim_params.qubo_solver,
            qubo_solve_time_s=solve_time_s,
            budget_B=sim_params.budget_B,
            lambda_budget=sim_params.lambda_budget,
            eta_risk=sim_params.eta_risk,
            utility_weight=sim_params.utility_weight,
            var_cur=var_cur,
            var_tent=var_tent,
            default_loss=default_loss,
            default_loss_total=default_loss_total,
            R_scenarios=(R_scenarios if log_scenarios else None),
        )

    x_f = x.to(dtype=P.dtype)
    P_next = P_liq + x_f[:, None] * DeltaP

    M_req_next = margin_es(P_next, R_scenarios, alpha=sim_params.alpha)
    C_next = C_new

    z_prev_next = int(z_t)
    r_prev_next = r_t

    if logger is not None:
        logger.log(
            day=d,
            phase="end",
            P=P_next,
            W=W_new,
            C=C_next,
            alive=alive_new,
            z_t=int(z_t),
            r_t=r_t,
            pnl=pnl,
            M_req_cur=M_req_next,
            x=x,
            qubo_energy=energy,
            default_loss=default_loss,
            default_loss_total=default_loss_total,
        )

    out = {
        "P": P_next,
        "W": W_new,
        "C": C_next,
        "alive": alive_new,
        "z_prev": z_prev_next,
        "r_prev": r_prev_next,
        "z_t": int(z_t),
        "r_t": r_t,
        "pnl": pnl,
        "M_req_cur": M_req_cur,
        "M_req_tent": M_req_tent,
        "deltaM": deltaM,
        "default_loss": default_loss,
        "default_loss_total": float(default_loss_total),
        "default_now": default_now,
        "x": x,
        "qubo_energy": float(energy),
        "qubo_num_vars": qubo_num_vars,
        "qubo_num_interactions": qubo_num_interactions,
        "qubo_solve_time_s": float(solve_time_s),
        "counterparty_network": counterparty_network,
    }
    if return_scenarios:
        out["R_scenarios"] = R_scenarios
    return out

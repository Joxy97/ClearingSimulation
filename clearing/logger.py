from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


def _cpu(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


@dataclass
class LogConfig:
    log_phases: tuple[str, ...] = ("start", "market_move", "settlement", "market", "margin", "default", "trades", "decision", "end")
    store_scenarios: bool = False
    scenarios_max_omega: int = 300
    scenarios_max_assets: int = 50
    store_scenario_stats: bool = True
    scenario_stats_quantiles: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)
    store_counterparty: bool = False


@dataclass
class SimLogger:
    cfg: LogConfig = field(default_factory=LogConfig)
    records: List[Dict[str, Any]] = field(default_factory=list)

    def log(
        self,
        *,
        day: int,
        phase: str,
        P: Optional[torch.Tensor] = None,
        W: Optional[torch.Tensor] = None,
        C: Optional[torch.Tensor] = None,
        alive: Optional[torch.Tensor] = None,
        z_t: Optional[int] = None,
        r_t: Optional[torch.Tensor] = None,
        pnl: Optional[torch.Tensor] = None,
        M_req_cur: Optional[torch.Tensor] = None,
        M_req_tent: Optional[torch.Tensor] = None,
        var_cur: Optional[torch.Tensor] = None,
        var_tent: Optional[torch.Tensor] = None,
        deltaM: Optional[torch.Tensor] = None,
        DeltaP: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        qubo_energy: Optional[float] = None,
        qubo_num_vars: Optional[int] = None,
        qubo_num_interactions: Optional[int] = None,
        qubo_solver: Optional[str] = None,
        qubo_solve_time_s: Optional[float] = None,
        default_loss: Optional[torch.Tensor] = None,
        default_loss_total: Optional[float] = None,
        R_scenarios: Optional[torch.Tensor] = None,
        budget_B: Optional[float] = None,
        lambda_budget: Optional[float] = None,
        eta_risk: Optional[float] = None,
        utility_weight: Optional[float] = None,
        counterparty_exposure: Optional[torch.Tensor] = None,
        default_fund_balance: Optional[float] = None,
        cva_margin: Optional[torch.Tensor] = None,
        contagion_round: Optional[int] = None,
    ) -> None:
        if phase not in self.cfg.log_phases:
            return

        rec: Dict[str, Any] = {"day": int(day), "phase": str(phase)}
        if z_t is not None:
            rec["z_t"] = int(z_t)

        rec["P"] = _cpu(P)
        rec["W"] = _cpu(W)
        rec["C"] = _cpu(C)
        rec["alive"] = _cpu(alive)
        rec["r_t"] = _cpu(r_t)
        rec["pnl"] = _cpu(pnl)
        rec["M_req_cur"] = _cpu(M_req_cur)
        rec["M_req_tent"] = _cpu(M_req_tent)
        rec["var_cur"] = _cpu(var_cur)
        rec["var_tent"] = _cpu(var_tent)
        rec["deltaM"] = _cpu(deltaM)
        rec["DeltaP"] = _cpu(DeltaP)
        rec["x"] = _cpu(x)
        rec["qubo_energy"] = None if qubo_energy is None else float(qubo_energy)
        rec["default_loss"] = _cpu(default_loss)
        if default_loss_total is not None:
            rec["default_loss_total"] = float(default_loss_total)
        if qubo_num_vars is not None:
            rec["qubo_num_vars"] = int(qubo_num_vars)
        if qubo_num_interactions is not None:
            rec["qubo_num_interactions"] = int(qubo_num_interactions)
        if qubo_solver is not None:
            rec["qubo_solver"] = str(qubo_solver)
        if qubo_solve_time_s is not None:
            rec["qubo_solve_time_s"] = float(qubo_solve_time_s)
        if budget_B is not None:
            rec["budget_B"] = float(budget_B)
        if lambda_budget is not None:
            rec["lambda_budget"] = float(lambda_budget)
        if eta_risk is not None:
            rec["eta_risk"] = float(eta_risk)
        if utility_weight is not None:
            rec["utility_weight"] = float(utility_weight)

        if counterparty_exposure is not None and self.cfg.store_counterparty:
            rec["counterparty_exposure"] = _cpu(counterparty_exposure)
        if default_fund_balance is not None:
            rec["default_fund_balance"] = float(default_fund_balance)
        if cva_margin is not None:
            rec["cva_margin"] = _cpu(cva_margin)
        if contagion_round is not None:
            rec["contagion_round"] = int(contagion_round)

        if R_scenarios is not None and self.cfg.store_scenario_stats:
            Rs = R_scenarios.detach()
            mean = Rs.mean(dim=0)
            std = Rs.std(dim=0)
            rec["scen_mean"] = _cpu(mean)
            rec["scen_std"] = _cpu(std)

            qs = torch.tensor(self.cfg.scenario_stats_quantiles, device=Rs.device, dtype=torch.float32)
            rec["scen_q"] = _cpu(torch.quantile(Rs, qs, dim=0))
            rec["scen_q_levels"] = self.cfg.scenario_stats_quantiles

        if R_scenarios is not None and self.cfg.store_scenarios:
            Rs = R_scenarios.detach()
            omega, N = Rs.shape
            omega_keep = min(omega, self.cfg.scenarios_max_omega)
            n_assets = min(N, self.cfg.scenarios_max_assets)
            rec["R_scenarios"] = _cpu(Rs[:omega_keep, :n_assets])
            rec["R_scenarios_shape_full"] = (int(omega), int(N))

        self.records.append(rec)

    def get(self) -> List[Dict[str, Any]]:
        return self.records

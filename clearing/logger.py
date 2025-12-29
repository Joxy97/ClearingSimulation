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
    log_phases: tuple[str, ...] = ("start", "market", "margin", "trades", "decision", "end")
    store_scenarios: bool = False
    scenarios_max_omega: int = 300
    scenarios_max_assets: int = 50
    store_scenario_stats: bool = True
    scenario_stats_quantiles: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)


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
        deltaM: Optional[torch.Tensor] = None,
        DeltaP: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        qubo_energy: Optional[float] = None,
        R_scenarios: Optional[torch.Tensor] = None,
        budget_B: Optional[float] = None,
        lambda_budget: Optional[float] = None,
        eta_risk: Optional[float] = None,
        utility_weight: Optional[float] = None,
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
        rec["deltaM"] = _cpu(deltaM)
        rec["DeltaP"] = _cpu(DeltaP)
        rec["x"] = _cpu(x)
        rec["qubo_energy"] = None if qubo_energy is None else float(qubo_energy)
        if budget_B is not None:
            rec["budget_B"] = float(budget_B)
        if lambda_budget is not None:
            rec["lambda_budget"] = float(lambda_budget)
        if eta_risk is not None:
            rec["eta_risk"] = float(eta_risk)
        if utility_weight is not None:
            rec["utility_weight"] = float(utility_weight)

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

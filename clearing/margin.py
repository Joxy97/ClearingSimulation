from __future__ import annotations

import torch


@torch.no_grad()
def margin_es(
    P: torch.Tensor,
    R_scenarios: torch.Tensor,
    *,
    alpha: float = 0.99,
    return_var: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Scenario-based margin using Expected Shortfall (ES) on losses.

    Returns:
      ES:  [M]
      (optional) VaR: [M]
    """
    if P.ndim != 2:
        raise ValueError("P must be [M, N].")
    if R_scenarios.ndim != 2 or R_scenarios.shape[1] != P.shape[1]:
        raise ValueError("R_scenarios must be [Omega, N] matching P.shape[1].")

    device = P.device
    dtype = torch.float32

    P_ = P.to(device=device, dtype=dtype)
    R_ = R_scenarios.to(device=device, dtype=dtype)

    pnl = R_ @ P_.T
    loss = -pnl

    omega = loss.shape[0]
    if omega < 2:
        raise ValueError("Need at least 2 scenarios for ES/VaR.")

    tail = max(1, int((1.0 - float(alpha)) * omega))

    loss_sorted, _ = torch.sort(loss, dim=0)
    tail_losses = loss_sorted[-tail:, :]

    es = tail_losses.mean(dim=0)

    if not return_var:
        return es

    q_idx = min(omega - 1, max(0, int(alpha * (omega - 1))))
    var = loss_sorted[q_idx, :]
    return es, var

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from boltzmann import RBM  # noqa: E402

from clearing.logger import LogConfig, SimLogger  # noqa: E402
from clearing.margin import margin_es  # noqa: E402
from clearing.market import make_default_params, simulate_market  # noqa: E402
from clearing.quantization import fit_return_quantizer  # noqa: E402
from clearing.sampler import sample_scenarios_from_float  # noqa: E402
from clearing.simulation import SimDayParams, simulate_day  # noqa: E402
from clearing.trading import TradeParams  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clearing simulation with RBM scenarios.")
    parser.add_argument("--M", type=int, default=10, help="Number of clients.")
    parser.add_argument("--N", type=int, default=5, help="Number of instruments.")
    parser.add_argument("--T", type=int, default=10, help="Number of simulation days.")
    parser.add_argument("--Omega", type=int, default=500, help="Scenarios per day.")
    parser.add_argument("--model-run", type=str, default="models/model_mini", help="RBM run folder.")
    parser.add_argument("--quantizer", type=str, default="data/quantizer.pt", help="Quantizer path (optional).")
    parser.add_argument("--quantizer-fit-T", type=int, default=10000, help="Days for fitting quantizer if not found.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, or cuda.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Tensor dtype.")
    parser.add_argument("--seed-market", type=int, default=123, help="Market RNG seed.")
    parser.add_argument("--seed-trade", type=int, default=456, help="Trade RNG seed.")
    parser.add_argument("--burn-in", type=int, default=200, help="RBM burn-in.")
    parser.add_argument("--thin", type=int, default=5, help="RBM thinning.")
    parser.add_argument("--qubo-solver", type=str, default="sa", choices=["sa", "hybrid", "leap_hybrid"], help="QUBO solver.")
    parser.add_argument("--budget", type=float, default=0.0, help="Risk budget B.")
    parser.add_argument("--lambda-budget", type=float, default=10.0, help="Budget penalty lambda.")
    parser.add_argument("--eta-risk", type=float, default=0.0, help="Risk penalty eta.")
    parser.add_argument("--utility-weight", type=float, default=0.0, help="Utility weight.")
    parser.add_argument("--trade-base-scale", type=float, default=5.0, help="Base trade size.")
    parser.add_argument("--trade-noise-scale", type=float, default=0.3, help="Trade noise scale.")
    parser.add_argument("--trade-max-abs", type=float, default=100.0, help="Max absolute trade size.")
    parser.add_argument("--log-path", type=str, default="simulations/logs/run_log.pt", help="Output path for log tensor.")
    return parser.parse_args()


def _safe_mean(v: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        return float(v.mean().item())
    vv = v[mask]
    return float(vv.mean().item()) if vv.numel() > 0 else 0.0


def _load_or_fit_quantizer(
    *,
    quantizer_path: str,
    market_params,
    device: str,
    dtype: torch.dtype,
    fit_T: int,
    K_bits: int = 4,
) -> object:
    if quantizer_path and os.path.exists(quantizer_path):
        try:
            return torch.load(quantizer_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(quantizer_path, map_location="cpu")

    _, R, _ = simulate_market(T=fit_T, params=market_params, device=device, dtype=dtype, seed=42)
    return fit_return_quantizer(R, K=K_bits, q_low=0.001, q_high=0.999)


def main() -> None:
    args = _parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    market_params = make_default_params(N=args.N, S=3, F=2, device=device, dtype=dtype)

    g_market = torch.Generator(device=device).manual_seed(args.seed_market)
    g_trade = torch.Generator(device=device).manual_seed(args.seed_trade)

    model, cfg = RBM.from_run_folder(args.model_run, device=device)

    quantizer = _load_or_fit_quantizer(
        quantizer_path=args.quantizer,
        market_params=market_params,
        device=device,
        dtype=dtype,
        fit_T=args.quantizer_fit_T,
    )

    g_init = torch.Generator(device=device).manual_seed(args.seed_market + 999)

    P_raw = torch.rand((args.M, args.N), device=device, dtype=dtype, generator=g_init) * 2000.0 - 1000.0
    mask = (torch.rand((args.M, args.N), device=device, generator=g_init) < 0.30).to(dtype=dtype)
    P = P_raw * mask

    W = torch.normal(mean=20000.0, std=5000.0, size=(args.M,), device=device, dtype=dtype, generator=g_init)
    C = torch.zeros((args.M,), device=device, dtype=dtype)
    alive = torch.ones((args.M,), device=device, dtype=torch.bool)

    z_prev = 0
    r_prev = torch.zeros((args.N,), device=device, dtype=dtype)

    trade_params = TradeParams(
        base_scale=args.trade_base_scale,
        noise_scale=args.trade_noise_scale,
        p_trade_when_zero=0.10,
        zero_trade_std=1.0,
        max_abs_trade=args.trade_max_abs,
    )

    sim_params = SimDayParams(
        alpha=0.99,
        budget_B=args.budget,
        lambda_budget=args.lambda_budget,
        eta_risk=args.eta_risk,
        utility_weight=args.utility_weight,
        qubo_solver=args.qubo_solver,
        sa_num_reads=100,
        sa_num_sweeps=2000,
        post_full_margin_daily=True,
    )

    R_init = sample_scenarios_from_float(
        model=model,
        config=cfg,
        quantizer=quantizer,
        z_t=z_prev,
        r_prev_float=r_prev,
        num_samples=args.Omega,
        burn_in=args.burn_in,
        thin=args.thin,
        device=device,
    )
    M_req_init = margin_es(P, R_init, alpha=sim_params.alpha)
    C = M_req_init * 1.10

    logger = SimLogger(
        LogConfig(
            store_scenarios=True,
            store_scenario_stats=True,
            scenarios_max_omega=200,
            scenarios_max_assets=min(20, args.N),
        )
    )

    for t in range(args.T):
        out = simulate_day(
            P=P,
            W=W,
            C=C,
            alive=alive,
            z_prev=z_prev,
            r_prev=r_prev,
            market_params=market_params,
            rbm_model=model,
            rbm_config=cfg,
            quantizer=quantizer,
            trade_params=trade_params,
            sim_params=sim_params,
            g_market=g_market,
            g_trade=g_trade,
            num_scenarios=args.Omega,
            burn_in=args.burn_in,
            thin=args.thin,
            device=device,
            day=t,
            logger=logger,
            log_scenarios=True,
            return_scenarios=False,
        )

        P, W, C, alive = out["P"], out["W"], out["C"], out["alive"]
        z_prev, r_prev = out["z_prev"], out["r_prev"]

        default_now = out["default_now"]
        x = out["x"]
        pnl = out["pnl"]
        M_req_cur = out["M_req_cur"]
        deltaM = out["deltaM"]

        n_alive = int(alive.sum().item())
        n_def = int(default_now.sum().item())
        acc_rate = float((x[alive].float().mean().item()) if n_alive > 0 else 0.0)

        print(
            f"Day {t:02d} | state z={out['z_t']} | alive={n_alive:02d} | defaulted={n_def:02d} | "
            f"acc_rate={acc_rate:.2f} | "
            f"mean_pnl={_safe_mean(pnl, alive):.2f} | "
            f"mean_margin={_safe_mean(M_req_cur, alive):.2f} | "
            f"mean_deltaM={_safe_mean(deltaM, alive):.2f} | "
            f"qubo_E={out['qubo_energy']:.2f}"
        )

    if args.log_path:
        out_dir = os.path.dirname(args.log_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(logger.get(), args.log_path)
        print(f"Saved log to {args.log_path}")


if __name__ == "__main__":
    main()

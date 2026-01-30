from __future__ import annotations

import argparse
import os
import sys
import time

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
    parser.add_argument("--M", type=int, default=100, help="Number of clients.")
    parser.add_argument("--T", type=int, default=10, help="Number of simulation days.")
    parser.add_argument("--N", type=int, default=500, help="Number of instruments. Change requires new RBM.")
    parser.add_argument("--Omega", type=int, default=1000, help="Scenarios per day.")
    parser.add_argument("--model-run", type=str, default="models/model", help="RBM run folder.")
    parser.add_argument("--quantizer", type=str, default="data/quantizer", help="Quantizer path (optional).")
    parser.add_argument("--quantizer-fit-T", type=int, default=10000, help="Days for fitting quantizer if not found.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, or cuda.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Tensor dtype.")
    parser.add_argument("--seed-market", type=int, default=42, help="Market RNG seed.")
    parser.add_argument("--seed-trade", type=int, default=42, help="Trade RNG seed.")
    parser.add_argument("--burn-in", type=int, default=200, help="RBM burn-in.")
    parser.add_argument("--thin", type=int, default=5, help="RBM thinning.")
    parser.add_argument("--qubo-solver", type=str, default="sa", choices=["sa", "hybrid"], help="QUBO solver.")
    parser.add_argument("--budget", type=float, default=0.0, help="Risk budget B.")
    parser.add_argument("--lambda-budget", type=float, default=10.0, help="Budget penalty lambda.")
    parser.add_argument("--eta-risk", type=float, default=0.0, help="Risk penalty eta.")
    parser.add_argument("--utility-weight", type=float, default=0.0, help="Utility weight.")
    parser.add_argument("--trade-base-scale", type=float, default=5.0, help="Base trade size.")
    parser.add_argument("--trade-noise-scale", type=float, default=0.3, help="Trade noise scale.")
    parser.add_argument("--trade-max-abs", type=float, default=100.0, help="Max absolute trade size.")
    parser.add_argument("--init-collateral-buffer", type=float, default=0.10, help="Initial collateral buffer as fraction of ES.")
    parser.add_argument("--init-liquidity-buffer", type=float, default=0.20, help="Initial liquidity buffer as fraction of ES.")
    parser.add_argument("--init-position-scale", type=float, default=1000.0, help="Initial absolute position scale.")
    parser.add_argument("--init-position-density", type=float, default=0.30, help="Initial position density (0..1).")
    parser.add_argument("--init-position-long-only", action="store_true", help="Use long-only initial positions.")
    parser.add_argument("--stress", action="store_true", help="Enable stress-test preset.")
    parser.add_argument("--start-state", type=int, default=None, help="Initial market state z_0 (default: 0, or 2 with --stress).")
    parser.add_argument("--stress-level", type=float, default=3.0, help="Stress multiplier for market volatility/jumps and position scale.")
    parser.add_argument("--stress-day", type=int, default=0, help="Day index to apply return shock in stress mode.")
    parser.add_argument("--stress-return-scale", type=float, default=3.0, help="Return scale on the stress day.")
    parser.add_argument("--stress-return-shift", type=float, default=-0.05, help="Return shift on the stress day.")
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
    stress_on = bool(args.stress)
    stress_level = float(args.stress_level) if stress_on else 1.0
    if stress_on:
        if stress_level < 1.0:
            stress_level = 1.0
        market_params.sigma_f = market_params.sigma_f * stress_level
        market_params.sigma_e = market_params.sigma_e * stress_level
        if market_params.jump_sigma is not None:
            market_params.jump_sigma = market_params.jump_sigma * stress_level
        if market_params.jump_p is not None:
            market_params.jump_p = torch.clamp(market_params.jump_p * stress_level, max=0.5)
        bias = min(0.8, 0.15 * (stress_level - 1.0))
        if bias > 0.0:
            crisis = torch.tensor([0.05, 0.15, 0.80], device=device, dtype=dtype)
            market_params.Pz = (1.0 - bias) * market_params.Pz + bias * crisis[None, :]

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

    init_position_scale = float(args.init_position_scale)
    init_position_density = float(args.init_position_density)
    if stress_on:
        init_position_scale = init_position_scale * stress_level
        init_position_density = min(1.0, init_position_density * stress_level)
    if not (0.0 <= init_position_density <= 1.0):
        raise ValueError("init-position-density must be in [0, 1].")

    if args.init_position_long_only:
        P_raw = torch.rand((args.M, args.N), device=device, dtype=dtype, generator=g_init) * init_position_scale
    else:
        P_raw = torch.rand((args.M, args.N), device=device, dtype=dtype, generator=g_init) * (2.0 * init_position_scale) - init_position_scale
    mask = (torch.rand((args.M, args.N), device=device, generator=g_init) < init_position_density).to(dtype=dtype)
    P = P_raw * mask

    C = torch.zeros((args.M,), device=device, dtype=dtype)
    W = torch.zeros((args.M,), device=device, dtype=dtype)
    default_loss = torch.zeros((args.M,), device=device, dtype=dtype)
    alive = torch.ones((args.M,), device=device, dtype=torch.bool)

    init_collateral_buffer = float(args.init_collateral_buffer)
    init_liquidity_buffer = float(args.init_liquidity_buffer)
    if stress_on:
        init_collateral_buffer = min(init_collateral_buffer, 0.02)
        init_liquidity_buffer = min(init_liquidity_buffer, 0.05)

    if args.start_state is None:
        z_prev = 2 if stress_on else 0
    else:
        z_prev = int(args.start_state)
    start_state = z_prev
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
        stress_day=(args.stress_day if stress_on else None),
        stress_return_scale=(args.stress_return_scale if stress_on else 1.0),
        stress_return_shift=(args.stress_return_shift if stress_on else 0.0),
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
    M_req_init = torch.clamp(M_req_init, min=0.0)
    C = M_req_init * (1.0 + init_collateral_buffer)
    F_init = M_req_init * init_liquidity_buffer
    W = C + F_init
    W_start_total = float(W.sum().item())

    logger = SimLogger(
        LogConfig(
            store_scenarios=True,
            store_scenario_stats=True,
            scenarios_max_omega=200,
            scenarios_max_assets=min(20, args.N),
        )
    )

    qubo_sizes: list[float] = []
    qubo_times: list[float] = []
    qubo_energies: list[float] = []
    sim_t0 = time.perf_counter()
    for t in range(args.T):
        out = simulate_day(
            P=P,
            W=W,
            C=C,
            default_loss=default_loss,
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
        default_loss = out.get("default_loss", default_loss)
        r_prev = out["r_prev"]

        default_now = out["default_now"]
        x = out["x"]
        pnl = out["pnl"]
        M_req_cur = out["M_req_cur"]
        deltaM = out["deltaM"]
        qubo_energy = out.get("qubo_energy")
        qubo_num_vars = out.get("qubo_num_vars")
        qubo_num_interactions = out.get("qubo_num_interactions")
        qubo_solve_time_s = out.get("qubo_solve_time_s")
        if qubo_energy is not None:
            qubo_energies.append(float(qubo_energy))
        if qubo_num_vars is not None and qubo_num_interactions is not None:
            qubo_sizes.append(float(qubo_num_vars) + float(qubo_num_interactions))
        elif qubo_num_vars is not None:
            qubo_sizes.append(float(qubo_num_vars))
        if qubo_solve_time_s is not None:
            qubo_times.append(float(qubo_solve_time_s))

        n_alive = int(alive.sum().item())
        n_def = int(default_now.sum().item())
        acc_rate = float((x[alive].float().mean().item()) if n_alive > 0 else 0.0)

        print(
            f"Day {t + 1:02d}/{args.T:02d} | alive={n_alive:02d} | defaulted={n_def:02d} | "
            f"acc_rate={acc_rate:.2f} | "
            f"mean_pnl={_safe_mean(pnl, alive):.2f} | "
            f"mean_margin={_safe_mean(M_req_cur, alive):.2f} | "
            f"mean_deltaM={_safe_mean(deltaM, alive):.2f} | "
            f"qubo_E={out['qubo_energy']:.2f}"
        )

    sim_time_s = time.perf_counter() - sim_t0
    W_end_total = float(W.sum().item())
    clients_gain = W_end_total - W_start_total
    total_dl = float(default_loss.sum().item())
    total_defaulted = int((~alive).sum().item())
    mean_qubo_size = sum(qubo_sizes) / len(qubo_sizes) if qubo_sizes else 0.0
    mean_qubo_time = sum(qubo_times) / len(qubo_times) if qubo_times else 0.0
    mean_qubo_energy = sum(qubo_energies) / len(qubo_energies) if qubo_energies else 0.0

    print("Simulation summary")
    print(f"Total DL: {total_dl:.2f}")
    print(f"Clients' gain: {clients_gain:.2f}")
    print(f"Total defaulted clients: {total_defaulted}")
    print(f"Mean QUBO size (vars + interactions): {mean_qubo_size:.2f}")
    print(f"Mean QUBO time (s): {mean_qubo_time:.4f}")
    print(f"Mean QUBO energy: {mean_qubo_energy:.2f}")
    print(f"Total simulation time (s): {sim_time_s:.2f}")

    if args.log_path:
        out_dir = os.path.dirname(args.log_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        metadata = {
            "M": int(args.M),
            "N": int(args.N),
            "T": int(args.T),
            "Omega": int(args.Omega),
            "model_run": str(args.model_run),
            "quantizer": str(args.quantizer),
            "quantizer_fit_T": int(args.quantizer_fit_T),
            "device": str(device),
            "dtype": str(args.dtype),
            "seed_market": int(args.seed_market),
            "seed_trade": int(args.seed_trade),
            "burn_in": int(args.burn_in),
            "thin": int(args.thin),
            "qubo_solver": str(args.qubo_solver),
            "budget_B": float(args.budget),
            "lambda_budget": float(args.lambda_budget),
            "eta_risk": float(args.eta_risk),
            "utility_weight": float(args.utility_weight),
            "trade_base_scale": float(args.trade_base_scale),
            "trade_noise_scale": float(args.trade_noise_scale),
            "trade_max_abs": float(args.trade_max_abs),
            "init_collateral_buffer": float(init_collateral_buffer),
            "init_liquidity_buffer": float(init_liquidity_buffer),
            "init_position_scale": float(init_position_scale),
            "init_position_density": float(init_position_density),
            "init_position_long_only": bool(args.init_position_long_only),
            "start_state": int(start_state),
            "stress": bool(stress_on),
            "stress_level": float(stress_level),
            "stress_day": None if not stress_on else int(args.stress_day),
            "stress_return_scale": float(sim_params.stress_return_scale),
            "stress_return_shift": float(sim_params.stress_return_shift),
            "trade_momentum_weight": float(trade_params.momentum_weight),
            "trade_gamma_exposure": float(trade_params.gamma_exposure),
            "p_trade_when_zero": float(trade_params.p_trade_when_zero),
            "zero_trade_std": float(trade_params.zero_trade_std),
            "alpha": float(sim_params.alpha),
            "post_full_margin_daily": bool(sim_params.post_full_margin_daily),
        }
        torch.save({"records": logger.get(), "meta": metadata}, args.log_path)
        print(f"Saved log to {args.log_path}")


if __name__ == "__main__":
    main()

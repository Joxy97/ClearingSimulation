from __future__ import annotations

import unittest
from unittest import mock

import torch

from clearing.market import make_default_params
from clearing.simulation import SimDayParams, simulate_day
from clearing.trading import TradeParams


class TestSimulation(unittest.TestCase):
    def test_simulate_day_plumbing_and_defaults(self) -> None:
        device = "cpu"
        dtype = torch.float32

        M = 2
        N = 2

        P = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)
        W = torch.tensor([10.0, 0.5], dtype=dtype)
        C = torch.zeros((M,), dtype=dtype)
        alive = torch.ones((M,), dtype=torch.bool)

        z_prev = 0
        r_prev = torch.zeros((N,), dtype=dtype)

        market_params = make_default_params(N=N, S=3, F=1, device=device, dtype=dtype)

        fixed_z_t = 1
        fixed_r_t = torch.tensor([1.0, -1.0], dtype=dtype)
        fixed_d_t = torch.zeros((market_params.S + 2 * N,), dtype=dtype)

        scenarios = torch.tensor(
            [
                [-1.0, -1.0],
                [-2.0, -0.5],
            ],
            dtype=dtype,
        )

        trade_params = TradeParams(
            base_scale=1.0,
            noise_scale=0.0,
            p_trade_when_zero=1.0,
            momentum_weight=1.0,
            gamma_exposure=0.0,
        )

        sim_params = SimDayParams(
            alpha=0.99,
            budget_B=0.0,
            lambda_budget=1.0,
            eta_risk=0.0,
            utility_weight=0.0,
            qubo_solver="sa",
            sa_num_reads=10,
            sa_num_sweeps=10,
            post_full_margin_daily=True,
        )

        with mock.patch("clearing.simulation.generate_day", return_value=(fixed_z_t, fixed_r_t, fixed_d_t)), \
            mock.patch("clearing.simulation.sample_scenarios_from_float", return_value=scenarios), \
            mock.patch("clearing.simulation.solve_bqm", return_value=(torch.tensor([1, 0]), 0.0)):
            out = simulate_day(
                P=P,
                W=W,
                C=C,
                alive=alive,
                z_prev=z_prev,
                r_prev=r_prev,
                market_params=market_params,
                rbm_model=None,
                rbm_config={},
                quantizer=None,
                trade_params=trade_params,
                sim_params=sim_params,
                g_market=None,
                g_trade=None,
                num_scenarios=2,
                burn_in=1,
                thin=1,
                device=device,
                day=0,
                logger=None,
                log_scenarios=False,
                return_scenarios=False,
            )

        self.assertEqual(out["z_t"], fixed_z_t)
        self.assertTrue(torch.equal(out["default_now"], torch.tensor([False, True])))
        self.assertTrue(torch.equal(out["alive"], torch.tensor([True, False])))
        self.assertTrue(torch.equal(out["x"], torch.tensor([1, 0])))
        self.assertAlmostEqual(out["qubo_energy"], 0.0, places=6)

        expected_p0 = torch.tensor([2.0, -1.0])
        self.assertTrue(torch.allclose(out["P"][0], expected_p0))
        self.assertTrue(torch.allclose(out["P"][1], torch.zeros(2)))

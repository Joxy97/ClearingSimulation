from __future__ import annotations

import unittest

import torch

from clearing.trading import TradeParams, propose_trades


class TestTrading(unittest.TestCase):
    def test_zero_positions_no_trade_when_prob_zero(self) -> None:
        P = torch.zeros((3, 2), dtype=torch.float32)
        r_t = torch.tensor([1.0, -1.0])
        params = TradeParams(
            base_scale=10.0,
            noise_scale=1.0,
            p_trade_when_zero=0.0,
            zero_trade_std=1.0,
        )
        g = torch.Generator(device="cpu").manual_seed(123)
        DeltaP = propose_trades(P, r_t, params, generator=g)
        self.assertTrue(torch.allclose(DeltaP, torch.zeros_like(DeltaP)))

    def test_alive_mask_zeros_trades(self) -> None:
        P = torch.ones((2, 2), dtype=torch.float32)
        r_t = torch.tensor([1.0, 1.0])
        params = TradeParams(base_scale=1.0, noise_scale=0.0, p_trade_when_zero=1.0)
        alive = torch.tensor([True, False])

        DeltaP = propose_trades(P, r_t, params, alive=alive)
        self.assertTrue(torch.all(DeltaP[1] == 0.0))
        self.assertTrue(torch.any(DeltaP[0] != 0.0))

    def test_max_abs_trade_clamp(self) -> None:
        P = torch.ones((1, 2), dtype=torch.float32)
        r_t = torch.tensor([10.0, -10.0])
        params = TradeParams(base_scale=10.0, noise_scale=0.0, p_trade_when_zero=1.0, max_abs_trade=5.0)
        DeltaP = propose_trades(P, r_t, params)
        self.assertTrue(torch.all(torch.abs(DeltaP) <= 5.0 + 1e-6))

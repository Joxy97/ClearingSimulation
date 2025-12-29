from __future__ import annotations

import unittest

import torch

from clearing.market import MarketParams, generate_day, make_default_params, simulate_market


class TestMarket(unittest.TestCase):
    def test_make_default_params_requires_S3(self) -> None:
        with self.assertRaises(ValueError):
            make_default_params(S=2, device="cpu")

    def test_generate_day_shapes(self) -> None:
        device = "cpu"
        dtype = torch.float32
        params = make_default_params(N=3, S=3, F=2, device=device, dtype=dtype)

        z_prev = 0
        r_prev = torch.zeros(params.N, device=device, dtype=dtype)
        g = torch.Generator(device=device).manual_seed(123)

        z_t, r_t, d_t = generate_day(z_prev, r_prev, params, generator=g)

        self.assertIsInstance(z_t, int)
        self.assertEqual(r_t.shape, (params.N,))
        self.assertEqual(d_t.shape, (params.S + 2 * params.N,))

        state = d_t[: params.S]
        self.assertAlmostEqual(float(state.sum().item()), 1.0)
        self.assertTrue(torch.all((state == 0.0) | (state == 1.0)))

        r_prev_slice = d_t[params.S : params.S + params.N]
        self.assertTrue(torch.allclose(r_prev_slice, r_prev))

    def test_simulate_market_deterministic(self) -> None:
        device = "cpu"
        dtype = torch.float32
        params = make_default_params(N=2, S=3, F=1, device=device, dtype=dtype)

        Z1, R1, D1 = simulate_market(T=20, params=params, device=device, dtype=dtype, seed=7)
        Z2, R2, D2 = simulate_market(T=20, params=params, device=device, dtype=dtype, seed=7)

        torch.testing.assert_close(Z1, Z2)
        torch.testing.assert_close(R1, R2)
        torch.testing.assert_close(D1, D2)

        self.assertTrue(torch.all((Z1 >= 0) & (Z1 < params.S)))

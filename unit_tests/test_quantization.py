from __future__ import annotations

import unittest

import torch

from clearing.quantization import ReturnQuantizer, decode_d, decode_returns, encode_d, encode_returns, fit_return_quantizer


class TestQuantization(unittest.TestCase):
    def test_encode_decode_returns_roundtrip(self) -> None:
        R = torch.tensor(
            [
                [-1.0, 0.0],
                [0.5, 1.0],
                [1.5, -0.5],
                [0.25, 0.75],
            ],
            dtype=torch.float32,
        )
        q = fit_return_quantizer(R, K=2, q_low=0.0, q_high=1.0)

        r = torch.tensor([0.0, 0.5], dtype=torch.float32)
        bits = encode_returns(r, q)
        self.assertEqual(bits.shape, (2 * q.K,))
        self.assertTrue(torch.all((bits == 0.0) | (bits == 1.0)))

        r_dec = decode_returns(bits, q)
        self.assertEqual(r_dec.shape, (2,))
        self.assertTrue(torch.all(r_dec >= q.lo - 1e-6))
        self.assertTrue(torch.all(r_dec <= q.hi + 1e-6))

    def test_encode_decode_d(self) -> None:
        q = ReturnQuantizer(lo=torch.tensor([-1.0, -2.0]), hi=torch.tensor([1.0, 2.0]), K=2)
        S = 3
        r_prev = torch.tensor([0.5, -1.5])
        r_t = torch.tensor([-0.5, 1.5])
        state = torch.tensor([0.0, 1.0, 0.0])
        d = torch.cat([state, r_prev, r_t], dim=0)

        d_bits = encode_d(d, S=S, q=q)
        self.assertEqual(d_bits.shape, (S + 2 * 2 * q.K,))

        d_dec = decode_d(d_bits, S=S, q=q)
        self.assertEqual(d_dec.shape, (S + 2 * 2,))

        state_dec = d_dec[:S]
        self.assertTrue(torch.all((state_dec == 0.0) | (state_dec == 1.0)))
        self.assertAlmostEqual(float(state_dec.sum().item()), 1.0)

        r_dec = d_dec[S:]
        self.assertTrue(torch.all(r_dec >= q.lo.repeat(2) - 1e-6))
        self.assertTrue(torch.all(r_dec <= q.hi.repeat(2) + 1e-6))

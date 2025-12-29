from __future__ import annotations

import unittest

import torch

from clearing.margin import margin_es


class TestMargin(unittest.TestCase):
    def test_margin_es_values(self) -> None:
        P = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        R = torch.tensor(
            [
                [-1.0, -1.0],
                [-2.0, -0.5],
                [-0.5, -0.5],
            ]
        )
        es, var = margin_es(P, R, alpha=0.5, return_var=True)

        # Client 0 losses: [2.0, 2.5, 1.0] sorted -> [1.0, 2.0, 2.5]
        # tail=1 -> ES=2.5, VaR index=1 -> 2.0
        self.assertAlmostEqual(float(es[0].item()), 2.5, places=5)
        self.assertAlmostEqual(float(var[0].item()), 2.0, places=5)

        # Client 1 losses: [1.0, 0.5, 0.5] sorted -> [0.5, 0.5, 1.0]
        self.assertAlmostEqual(float(es[1].item()), 1.0, places=5)
        self.assertAlmostEqual(float(var[1].item()), 0.5, places=5)

    def test_margin_zero_positions(self) -> None:
        P = torch.zeros((2, 3))
        R = torch.randn((5, 3))
        es = margin_es(P, R, alpha=0.9)
        self.assertTrue(torch.allclose(es, torch.zeros(2)))

    def test_requires_multiple_scenarios(self) -> None:
        P = torch.ones((1, 2))
        R = torch.zeros((1, 2))
        with self.assertRaises(ValueError):
            margin_es(P, R, alpha=0.9)

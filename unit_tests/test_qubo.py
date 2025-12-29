from __future__ import annotations

import unittest

import dimod
import torch

from clearing.qubo import build_bqm_accept_clients, solve_bqm


class TestQubo(unittest.TestCase):
    def test_build_bqm_energy(self) -> None:
        deltaM = torch.tensor([1.0, 2.0])
        bqm = build_bqm_accept_clients(deltaM, B=1.0, lambda_budget=1.0, eta_risk=0.0)

        energy_x10 = bqm.energy({0: 1, 1: 0})
        energy_x00 = bqm.energy({0: 0, 1: 0})

        self.assertLess(energy_x10, energy_x00)
        self.assertAlmostEqual(float(energy_x10), 0.0, places=5)

    def test_build_bqm_alive_penalty(self) -> None:
        deltaM = torch.tensor([1.0, 1.0])
        bqm = build_bqm_accept_clients(deltaM, B=0.0, lambda_budget=0.0, eta_risk=0.0, alive=torch.tensor([0, 1]))

        energy_dead = bqm.energy({0: 1, 1: 0})
        energy_ok = bqm.energy({0: 0, 1: 0})
        self.assertGreater(energy_dead, energy_ok)

    def test_solve_bqm_sa(self) -> None:
        deltaM = torch.tensor([1.0])
        bqm = build_bqm_accept_clients(deltaM, B=1.0, lambda_budget=1.0, eta_risk=0.0)

        x, energy = solve_bqm(bqm, method="sa", num_reads=10, num_sweeps=50, seed=123, out_device="cpu")

        self.assertEqual(x.shape, (1,))
        self.assertEqual(x.dtype, torch.int64)

        sample = {0: int(x.item())}
        self.assertAlmostEqual(float(energy), float(bqm.energy(sample)), places=5)

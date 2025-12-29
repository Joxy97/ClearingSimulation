from __future__ import annotations

import unittest

import torch

from clearing.logger import LogConfig, SimLogger


class TestLogger(unittest.TestCase):
    def test_logger_records_stats(self) -> None:
        logger = SimLogger(LogConfig(store_scenarios=True, store_scenario_stats=True, scenarios_max_omega=2, scenarios_max_assets=2))

        R_scenarios = torch.tensor(
            [
                [0.1, -0.1, 0.2],
                [0.0, 0.0, 0.1],
                [-0.2, 0.1, -0.1],
            ]
        )

        logger.log(
            day=1,
            phase="margin",
            P=torch.zeros((2, 3)),
            W=torch.ones(2),
            C=torch.zeros(2),
            alive=torch.tensor([1, 0], dtype=torch.bool),
            z_t=1,
            r_t=torch.tensor([0.01, -0.01, 0.0]),
            pnl=torch.tensor([0.1, -0.2]),
            M_req_cur=torch.tensor([1.0, 2.0]),
            R_scenarios=R_scenarios,
        )

        recs = logger.get()
        self.assertEqual(len(recs), 1)
        rec = recs[0]

        self.assertIn("scen_mean", rec)
        self.assertIn("scen_std", rec)
        self.assertIn("scen_q", rec)
        self.assertIn("R_scenarios", rec)

        self.assertEqual(rec["R_scenarios"].shape, (2, 2))
        self.assertEqual(rec["R_scenarios_shape_full"], (3, 3))

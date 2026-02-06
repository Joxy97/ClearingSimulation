"""Unit tests for counterparty network module."""

from __future__ import annotations

import unittest

import torch

from clearing.counterparty import (
    BilateralContract,
    CounterpartyNetwork,
    CounterpartyParams,
    compute_bilateral_exposures,
    generate_counterparty_network,
    initialize_default_fund,
    propagate_default_losses,
    value_contract,
)


class TestCounterpartyNetwork(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.dtype = torch.float32
        self.M = 10
        self.N = 5

    def test_network_generation_random(self):
        """Test random network topology generation."""
        P = torch.randn(self.M, self.N)
        params = CounterpartyParams(enabled=True, network_density=0.2, network_type="random")
        generator = torch.Generator().manual_seed(42)

        network = generate_counterparty_network(self.M, P, params, generator)

        self.assertIsInstance(network, CounterpartyNetwork)
        self.assertGreater(len(network.contracts), 0)
        for contract in network.contracts:
            self.assertIsInstance(contract, BilateralContract)
            self.assertTrue(0 <= contract.creditor < self.M)
            self.assertTrue(0 <= contract.debtor < self.M)
            self.assertNotEqual(contract.creditor, contract.debtor)

    def test_contract_valuation_equity_forward(self):
        """Test MTM calculation for equity forwards."""
        contract = BilateralContract(
            creditor=0,
            debtor=1,
            contract_type="equity_forward",
            notional=1000.0,
            maturity_days=90.0,
            strike=1.0,
            underlying_idx=0,
        )
        r_t = torch.tensor([0.05, -0.02, 0.01, 0.03, -0.01])

        mtm_value = value_contract(contract, r_t)

        self.assertAlmostEqual(mtm_value, 50.0, places=2)

    def test_contract_valuation_interest_rate_swap(self):
        """Test MTM calculation for interest rate swaps."""
        contract = BilateralContract(
            creditor=0,
            debtor=1,
            contract_type="interest_rate_swap",
            notional=10000.0,
            maturity_days=365.0,
            strike=0.0,
            underlying_idx=0,
        )
        r_t = torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0])

        mtm_value = value_contract(contract, r_t)

        self.assertGreater(mtm_value, 0)
        self.assertAlmostEqual(mtm_value, 200.0, delta=10.0)

    def test_default_fund_initialization(self):
        """Test DF contributions sum correctly."""
        P = torch.randn(self.M, self.N)
        params = CounterpartyParams(enabled=True, default_fund_ratio=0.05)
        generator = torch.Generator().manual_seed(42)

        network = generate_counterparty_network(self.M, P, params, generator)
        M_req_init = torch.rand(self.M) * 100 + 50

        initialize_default_fund(network, M_req_init, params)

        total_margin = M_req_init.sum().item()
        expected_df = total_margin * 0.05

        self.assertAlmostEqual(network.total_default_fund, expected_df, places=2)
        self.assertEqual(network.default_fund_contrib.shape, (self.M,))
        self.assertAlmostEqual(network.default_fund_contrib.sum().item(), expected_df, places=2)

    def test_exposure_matrix_structure(self):
        """Test exposure matrix has correct structure."""
        contracts = [
            BilateralContract(
                creditor=0, debtor=1, contract_type="equity_forward",
                notional=1000.0, maturity_days=90.0, strike=1.0, underlying_idx=0
            ),
            BilateralContract(
                creditor=1, debtor=2, contract_type="equity_forward",
                notional=500.0, maturity_days=60.0, strike=1.0, underlying_idx=1
            ),
        ]
        r_t = torch.zeros(self.N)

        exposure_matrix = compute_bilateral_exposures(
            contracts, r_t, self.M, self.device, self.dtype
        )

        self.assertEqual(exposure_matrix.shape, (self.M, self.M))
        self.assertTrue(torch.all(exposure_matrix >= 0))
        self.assertAlmostEqual(exposure_matrix.sum().item(), 0.0, places=4)

    def test_single_default_df_absorption(self):
        """Test single default absorbed by DF."""
        default_loss_initial = torch.zeros(self.M)
        default_loss_initial[0] = 100.0

        exposure_matrix = torch.zeros((self.M, self.M))
        exposure_matrix[1, 0] = 50.0
        exposure_matrix[2, 0] = 50.0

        W = torch.full((self.M,), 50.0)
        C = torch.full((self.M,), 20.0)
        default_fund_contrib = torch.full((self.M,), 20.0)
        total_default_fund = 200.0
        alive = torch.ones(self.M, dtype=torch.bool)

        params = CounterpartyParams(enabled=True, second_order_defaults=False)

        losses, W_after, C_after, df_used = propagate_default_losses(
            default_loss_initial, exposure_matrix, W, C,
            default_fund_contrib, total_default_fund, alive, params,
            logger=None, day=0, device=self.device, dtype=self.dtype
        )

        self.assertGreater(df_used, 0)
        self.assertLessEqual(df_used, 100.0)
        self.assertEqual(W_after[1].item(), 50.0)
        self.assertEqual(W_after[2].item(), 50.0)

    def test_default_fund_exhaustion(self):
        """Test losses propagate when DF exhausted."""
        default_loss_initial = torch.zeros(self.M)
        default_loss_initial[0] = 100.0

        exposure_matrix = torch.zeros((self.M, self.M))
        exposure_matrix[1, 0] = 60.0
        exposure_matrix[2, 0] = 40.0

        W = torch.full((self.M,), 10.0)
        C = torch.full((self.M,), 5.0)
        default_fund_contrib = torch.full((self.M,), 2.0)
        total_default_fund = 30.0
        alive = torch.ones(self.M, dtype=torch.bool)

        params = CounterpartyParams(enabled=True, second_order_defaults=False)

        losses, W_after, C_after, df_used = propagate_default_losses(
            default_loss_initial, exposure_matrix, W, C,
            default_fund_contrib, total_default_fund, alive, params,
            logger=None, day=0, device=self.device, dtype=self.dtype
        )

        self.assertEqual(df_used, 30.0)
        self.assertLess(W_after[1].item(), 10.0)
        self.assertLess(W_after[2].item(), 10.0)

        # Total owed = 100, available from debtor = 15 (W+C), shortfall = 85
        # DF covers 30, remaining shortfall = 55 split proportionally
        proportional_loss_1 = 60.0 / 100.0 * 55.0
        proportional_loss_2 = 40.0 / 100.0 * 55.0
        self.assertAlmostEqual(losses[1].item(), proportional_loss_1, places=2)
        self.assertAlmostEqual(losses[2].item(), proportional_loss_2, places=2)

    def test_no_contagion_without_exposure(self):
        """Test no contagion when there's no counterparty exposure."""
        default_loss_initial = torch.zeros(self.M)
        default_loss_initial[0] = 50.0

        exposure_matrix = torch.zeros((self.M, self.M))

        W = torch.full((self.M,), 100.0)
        C = torch.full((self.M,), 50.0)
        default_fund_contrib = torch.full((self.M,), 10.0)
        total_default_fund = 100.0
        alive = torch.ones(self.M, dtype=torch.bool)

        params = CounterpartyParams(enabled=True, second_order_defaults=True)

        losses, W_after, C_after, df_used = propagate_default_losses(
            default_loss_initial, exposure_matrix, W, C,
            default_fund_contrib, total_default_fund, alive, params,
            logger=None, day=0, device=self.device, dtype=self.dtype
        )

        self.assertEqual(df_used, 0.0)
        self.assertTrue(torch.allclose(W_after, W))
        self.assertTrue(torch.all(losses == 0.0))

    def test_cva_margin_calculation(self):
        """Test CVA correctly increases margin."""
        contracts = [
            BilateralContract(
                creditor=0, debtor=1, contract_type="equity_forward",
                notional=1000.0, maturity_days=90.0, strike=1.0, underlying_idx=0
            ),
        ]
        r_t = torch.tensor([0.10, 0.0, 0.0, 0.0, 0.0])

        exposure_matrix = compute_bilateral_exposures(
            contracts, r_t, self.M, self.device, self.dtype
        )

        total_receivables = exposure_matrix.sum(dim=1)
        cva_multiplier = 0.10
        cva_margin = cva_multiplier * total_receivables.clamp(min=0.0)

        self.assertGreater(cva_margin[0].item(), 0)
        self.assertAlmostEqual(cva_margin[0].item(), 10.0, places=1)


if __name__ == "__main__":
    unittest.main()

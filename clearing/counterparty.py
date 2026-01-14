"""
Counterparty network effects and default fund implementation.

This module implements bilateral derivative contracts between clients,
mark-to-market exposure tracking, and a mutualized default fund for
absorbing losses from client defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch


@dataclass
class CounterpartyParams:
    """Configuration for counterparty network and default fund."""

    # Network generation
    enabled: bool = False
    network_density: float = 0.15
    network_type: Literal["random", "preferential", "hedging"] = "random"
    preferential_alpha: float = 1.5

    # Contract sizing
    contract_notional_scale: float = 1000.0
    contract_notional_std: float = 0.3

    # Contract types and their relative frequencies
    contract_types: Dict[str, float] = field(default_factory=lambda: {
        "equity_forward": 0.4,
        "interest_rate_swap": 0.4,
        "fx_forward": 0.2,
    })

    # Contract parameters
    maturity_mean_days: float = 90.0
    maturity_std_days: float = 30.0
    strike_offset_mean: float = 0.0
    strike_offset_std: float = 0.02

    # Default fund
    default_fund_ratio: float = 0.05
    contribution_method: Literal["equal", "proportional", "exposure"] = "proportional"

    # CVA adjustment to margin
    cva_enabled: bool = True
    cva_multiplier: float = 0.10

    # Loss propagation
    second_order_defaults: bool = True
    max_propagation_rounds: int = 3


@dataclass
class BilateralContract:
    """Represents a single bilateral derivative contract."""
    creditor: int
    debtor: int
    contract_type: str
    notional: float
    maturity_days: float
    strike: float
    underlying_idx: int
    inception_value: float = 0.0


class CounterpartyNetwork:
    """
    Sparse representation of bilateral exposure network.
    """
    def __init__(
        self,
        contracts: List[BilateralContract],
        default_fund_contrib: torch.Tensor,
        total_default_fund: float,
    ):
        self.contracts = contracts
        self.default_fund_contrib = default_fund_contrib
        self.total_default_fund = total_default_fund
        self.exposure_matrix: Optional[torch.Tensor] = None


def generate_counterparty_network(
    M: int,
    P: torch.Tensor,
    params: CounterpartyParams,
    generator: Optional[torch.Generator] = None,
) -> CounterpartyNetwork:
    """
    Generate bilateral contract network at t=0.

    Args:
        M: Number of clients
        P: [M, N] initial portfolio positions
        params: Network configuration parameters
        generator: Random number generator

    Returns:
        CounterpartyNetwork with initialized contracts and default fund
    """
    if not params.enabled:
        raise ValueError("Cannot generate network when params.enabled=False")

    device = P.device
    dtype = P.dtype
    N = P.shape[1]

    # Generate network topology
    if params.network_type == "random":
        edges = _generate_random_network(M, params.network_density, generator)
    elif params.network_type == "preferential":
        edges = _generate_preferential_network(M, params.network_density, params.preferential_alpha, generator)
    elif params.network_type == "hedging":
        edges = _generate_hedging_network(M, P, params.network_density, generator)
    else:
        raise ValueError(f"Unknown network_type: {params.network_type}")

    # Generate contracts for each edge
    contracts = []
    for creditor, debtor in edges:
        num_contracts = 1 + int(torch.poisson(torch.tensor(0.5), generator=generator).item())
        for _ in range(num_contracts):
            contract = _create_contract(
                creditor, debtor, P, N, params, generator
            )
            contracts.append(contract)

    # Initialize default fund (needs initial margins, will be set later)
    # For now, create placeholder contributions
    default_fund_contrib = torch.zeros(M, device=device, dtype=dtype)
    total_default_fund = 0.0

    return CounterpartyNetwork(contracts, default_fund_contrib, total_default_fund)


def initialize_default_fund(
    network: CounterpartyNetwork,
    M_req_init: torch.Tensor,
    params: CounterpartyParams,
) -> None:
    """
    Initialize default fund contributions based on initial margins.

    Args:
        network: CounterpartyNetwork to update
        M_req_init: [M] initial margin requirements
        params: Network configuration parameters
    """
    M = M_req_init.shape[0]
    device = M_req_init.device
    dtype = M_req_init.dtype

    total_initial_margin = M_req_init.sum()
    target_df_size = total_initial_margin * params.default_fund_ratio

    if params.contribution_method == "equal":
        contrib = torch.full((M,), target_df_size / M, device=device, dtype=dtype)

    elif params.contribution_method == "proportional":
        contrib = M_req_init * (target_df_size / total_initial_margin)

    elif params.contribution_method == "exposure":
        total_notional = sum(c.notional for c in network.contracts)
        notional_per_client = torch.zeros(M, device=device, dtype=dtype)
        for c in network.contracts:
            notional_per_client[c.creditor] += c.notional
            notional_per_client[c.debtor] += c.notional
        total_client_notional = notional_per_client.sum()
        if total_client_notional > 0:
            contrib = notional_per_client * (target_df_size / total_client_notional)
        else:
            contrib = torch.full((M,), target_df_size / M, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown contribution_method: {params.contribution_method}")

    network.default_fund_contrib = contrib
    network.total_default_fund = float(contrib.sum().item())


def compute_bilateral_exposures(
    contracts: List[BilateralContract],
    r_t: torch.Tensor,
    M: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute current MTM value of all bilateral contracts.

    Args:
        contracts: List of bilateral contracts
        r_t: [N] current returns
        M: Number of clients
        device: Device for computation
        dtype: Data type for computation

    Returns:
        exposure_matrix: [M, M] where [i,j] = value client i is owed from j
    """
    exposure_matrix = torch.zeros((M, M), device=device, dtype=dtype)

    for contract in contracts:
        mtm_value = value_contract(contract, r_t)

        if mtm_value > 0:
            exposure_matrix[contract.creditor, contract.debtor] += mtm_value
        else:
            exposure_matrix[contract.debtor, contract.creditor] += abs(mtm_value)

    return exposure_matrix


def value_contract(
    contract: BilateralContract,
    r_t: torch.Tensor,
) -> float:
    """
    Compute MTM value of a single contract.

    Args:
        contract: Bilateral contract to value
        r_t: [N] current returns

    Returns:
        MTM value (positive = creditor is owed, negative = debtor is owed)
    """
    underlying_return = float(r_t[contract.underlying_idx].item())

    if contract.contract_type == "equity_forward":
        value = contract.notional * underlying_return

    elif contract.contract_type == "interest_rate_swap":
        maturity_factor = np.sqrt(contract.maturity_days / 365.0)
        value = contract.notional * underlying_return * maturity_factor

    elif contract.contract_type == "fx_forward":
        value = contract.notional * underlying_return
    else:
        value = 0.0

    return value


@torch.no_grad()
def propagate_default_losses(
    default_loss_initial: torch.Tensor,
    exposure_matrix: torch.Tensor,
    W: torch.Tensor,
    C: torch.Tensor,
    default_fund_contrib: torch.Tensor,
    total_default_fund: float,
    alive: torch.Tensor,
    params: CounterpartyParams,
    logger,
    day: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Multi-round default loss propagation with default fund waterfall.

    Waterfall per defaulting client:
    1. Use remaining collateral
    2. Use remaining wealth
    3. Allocate losses to DEFAULT FUND (mutualized)
    4. If DF exhausted, losses propagate to creditors

    Args:
        default_loss_initial: [M] initial losses from portfolio liquidation
        exposure_matrix: [M, M] counterparty exposures
        W: [M] wealth
        C: [M] collateral
        default_fund_contrib: [M] DF contributions
        total_default_fund: float amount of default fund available
        alive: [M] bool alive status
        params: Network configuration parameters
        logger: SimLogger instance
        day: Current day
        device: Device for computation
        dtype: Data type for computation

    Returns:
        final_default_losses: [M] total default losses per client
        W_after: [M] updated wealth
        C_after: [M] updated collateral
        df_used: float amount of default fund used
    """
    M = W.shape[0]
    alive_mask = alive.clone()
    accumulated_losses = torch.zeros_like(W)
    df_remaining = total_default_fund
    W_current = W.clone()
    C_current = C.clone()

    defaulted_this_round = (default_loss_initial > 0) & alive_mask

    contagion_round = 0
    while defaulted_this_round.any() and contagion_round < params.max_propagation_rounds:
        contagion_round += 1

        if logger is not None:
            logger.log(
                day=day,
                phase=f"contagion_round_{contagion_round}",
                P=None,
                W=W_current,
                C=C_current,
                alive=alive_mask,
                z_t=None,
                r_t=None,
                contagion_round=contagion_round,
            )

        for debtor_idx in defaulted_this_round.nonzero(as_tuple=True)[0]:
            debtor_idx = int(debtor_idx.item())

            total_owed = exposure_matrix[:, debtor_idx].sum()

            if total_owed <= 0:
                continue

            available = C_current[debtor_idx] + W_current[debtor_idx]
            shortfall = (total_owed - available).clamp(min=0.0)

            if shortfall > 0:
                df_allocation = min(float(shortfall), df_remaining)
                df_remaining -= df_allocation
                shortfall -= df_allocation

                if shortfall > 0:
                    creditor_exposures = exposure_matrix[:, debtor_idx]
                    total_creditor_exposure = creditor_exposures.sum()
                    if total_creditor_exposure > 0:
                        creditor_losses = (creditor_exposures / total_creditor_exposure) * shortfall
                    else:
                        creditor_losses = torch.zeros_like(creditor_exposures)

                    W_current = W_current - creditor_losses
                    accumulated_losses = accumulated_losses + creditor_losses

            alive_mask[debtor_idx] = False
            C_current[debtor_idx] = 0.0
            W_current[debtor_idx] = 0.0

        if not params.second_order_defaults:
            break

        F_new = (W_current - C_current)
        C_check = C_current + torch.minimum(F_new, torch.zeros_like(F_new))
        new_default_losses = (-C_check).clamp(min=0.0)

        defaulted_this_round = (new_default_losses > 0) & alive_mask

        if not defaulted_this_round.any():
            break

    df_used = total_default_fund - df_remaining

    return accumulated_losses, W_current, C_current, df_used


def _generate_random_network(
    M: int,
    density: float,
    generator: Optional[torch.Generator],
) -> List[Tuple[int, int]]:
    """Generate Erdős-Rényi random graph."""
    edges = []
    for i in range(M):
        for j in range(i + 1, M):
            if torch.rand((), generator=generator).item() < density:
                if torch.rand((), generator=generator).item() < 0.5:
                    edges.append((i, j))
                else:
                    edges.append((j, i))
    return edges


def _generate_preferential_network(
    M: int,
    density: float,
    alpha: float,
    generator: Optional[torch.Generator],
) -> List[Tuple[int, int]]:
    """Generate scale-free network with preferential attachment."""
    edges = []
    degrees = torch.zeros(M)

    m0 = min(5, M)
    for i in range(m0):
        for j in range(i + 1, m0):
            edges.append((i, j))
            degrees[i] += 1
            degrees[j] += 1

    for i in range(m0, M):
        num_edges = max(1, int(density * M))
        probs = (degrees + 1) ** alpha
        probs = probs / probs.sum()

        targets = torch.multinomial(probs, num_edges, replacement=False, generator=generator)
        for target in targets:
            target = int(target.item())
            if target != i:
                edges.append((i, target))
                degrees[i] += 1
                degrees[target] += 1

    return edges


def _generate_hedging_network(
    M: int,
    P: torch.Tensor,
    density: float,
    generator: Optional[torch.Generator],
) -> List[Tuple[int, int]]:
    """Generate network based on portfolio similarity (hedging)."""
    edges = []

    P_norm = P / (P.norm(dim=1, keepdim=True) + 1e-8)

    similarity = torch.mm(P_norm, P_norm.t())

    for i in range(M):
        for j in range(i + 1, M):
            if similarity[i, j] < -0.3 and torch.rand((), generator=generator).item() < density * 2:
                edges.append((i, j))

    return edges


def _create_contract(
    creditor: int,
    debtor: int,
    P: torch.Tensor,
    N: int,
    params: CounterpartyParams,
    generator: Optional[torch.Generator],
) -> BilateralContract:
    """Create a single bilateral contract between two clients."""

    types = list(params.contract_types.keys())
    weights = list(params.contract_types.values())
    weights_tensor = torch.tensor(weights)
    contract_type_idx = torch.multinomial(weights_tensor, 1, generator=generator).item()
    contract_type = types[contract_type_idx]

    portfolio_scale_creditor = P[creditor].abs().sum().item()
    portfolio_scale_debtor = P[debtor].abs().sum().item()
    avg_scale = (portfolio_scale_creditor + portfolio_scale_debtor) / 2.0

    notional_noise = 1.0 + torch.randn((), generator=generator).item() * params.contract_notional_std
    notional = max(1.0, avg_scale * params.contract_notional_scale * notional_noise)

    maturity_days = max(7.0, torch.randn((), generator=generator).item() * params.maturity_std_days + params.maturity_mean_days)

    if contract_type == "equity_forward":
        common_assets = ((P[creditor] != 0) & (P[debtor] != 0)).nonzero(as_tuple=True)[0]
        if len(common_assets) > 0:
            underlying_idx = int(common_assets[torch.randint(len(common_assets), (), generator=generator).item()].item())
        else:
            underlying_idx = torch.randint(N, (), generator=generator).item()
        strike_offset = torch.randn((), generator=generator).item() * params.strike_offset_std + params.strike_offset_mean
        strike = 1.0 + strike_offset

    elif contract_type == "interest_rate_swap":
        underlying_idx = 0
        strike = 0.0

    elif contract_type == "fx_forward":
        underlying_idx = torch.randint(N, (), generator=generator).item()
        strike_offset = torch.randn((), generator=generator).item() * params.strike_offset_std + params.strike_offset_mean
        strike = 1.0 + strike_offset
    else:
        underlying_idx = 0
        strike = 0.0

    return BilateralContract(
        creditor=creditor,
        debtor=debtor,
        contract_type=contract_type,
        notional=notional,
        maturity_days=maturity_days,
        strike=strike,
        underlying_idx=underlying_idx,
        inception_value=0.0,
    )

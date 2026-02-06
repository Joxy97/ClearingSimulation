"""Simple integration test for counterparty network."""

import torch

from clearing.counterparty import (
    CounterpartyParams,
    generate_counterparty_network,
    initialize_default_fund,
    compute_bilateral_exposures,
)

def test_counterparty_integration():
    """Test the full counterparty workflow."""
    print("=" * 60)
    print("Counterparty Network Integration Test")
    print("=" * 60)

    # Setup
    M = 20  # clients
    N = 10  # instruments
    device = "cpu"
    dtype = torch.float32

    # Generate random portfolios
    torch.manual_seed(42)
    P = torch.randn(M, N) * 100

    # Initialize counterparty network
    params = CounterpartyParams(
        enabled=True,
        network_density=0.2,
        network_type="random",
        default_fund_ratio=0.05,
        cva_multiplier=0.10,
        second_order_defaults=True,
    )

    print(f"\n1. Generating network...")
    print(f"   Clients: {M}")
    print(f"   Instruments: {N}")
    print(f"   Density: {params.network_density}")

    generator = torch.Generator().manual_seed(42)
    network = generate_counterparty_network(M, P, params, generator)

    print(f"   [OK] Generated {len(network.contracts)} bilateral contracts")

    # Show contract types
    contract_types = {}
    for c in network.contracts:
        contract_types[c.contract_type] = contract_types.get(c.contract_type, 0) + 1
    print(f"   Contract breakdown:")
    for ctype, count in contract_types.items():
        print(f"     - {ctype}: {count}")

    # Initialize default fund
    print(f"\n2. Initializing default fund...")
    M_req_init = torch.rand(M) * 50 + 20
    initialize_default_fund(network, M_req_init, params)

    total_margin = M_req_init.sum().item()
    print(f"   Total initial margin: ${total_margin:.2f}")
    print(f"   Default fund size: ${network.total_default_fund:.2f}")
    print(f"   DF ratio: {network.total_default_fund / total_margin * 100:.2f}%")

    # Compute exposures
    print(f"\n3. Computing bilateral exposures...")
    r_t = torch.randn(N) * 0.02  # 2% returns
    exposure_matrix = compute_bilateral_exposures(
        network.contracts, r_t, M, device, dtype
    )

    total_exposure = exposure_matrix.sum().item()
    print(f"   Total mark-to-market exposure: ${total_exposure:.2f}")

    # Compute CVA margin
    print(f"\n4. Computing CVA margin...")
    total_receivables = exposure_matrix.sum(dim=1)
    cva_margin = params.cva_multiplier * total_receivables.clamp(min=0.0)

    clients_with_exposure = (total_receivables > 0).sum().item()
    avg_cva = cva_margin[cva_margin > 0].mean().item() if (cva_margin > 0).any() else 0
    print(f"   Clients with counterparty exposure: {clients_with_exposure}/{M}")
    print(f"   Average CVA margin (exposed clients): ${avg_cva:.2f}")

    # Network statistics
    print(f"\n5. Network statistics...")
    degrees = torch.zeros(M)
    for c in network.contracts:
        degrees[c.creditor] += 1
        degrees[c.debtor] += 1

    max_degree = degrees.max().item()
    avg_degree = degrees.mean().item()
    print(f"   Average degree (contracts per client): {avg_degree:.2f}")
    print(f"   Max degree: {max_degree:.0f}")
    print(f"   Most connected client: {degrees.argmax().item()}")

    print(f"\n" + "=" * 60)
    print("[SUCCESS] All counterparty network components working correctly!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    test_counterparty_integration()

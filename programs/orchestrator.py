"""
MODULE 6: ORCHESTRATOR
========================
Main entry point. Ties all modules together:

    FTSO Oracle  →  QAOA Pathfinder  →  Mixing Pool  →  FDC Settlement
         ↓               ↓                  ↓                ↓
    Price data    Route optimisation   Fund mixing     Payment proofs
         ↓               ↓                  ↓                ↓
                      Visualiser (all panels)

FILE STRUCTURE:
    ftso_oracle.py        - FTSO price feeds + volatility (Module 1)
    qaoa_pathfinder.py    - Quantum QAOA circuit + route solving (Module 2)
    mixing_pool.py        - Pool aggregation + random hops (Module 3)
    fdc_settlement.py     - FDC attestation verification (Module 4)
    visualiser.py         - Matplotlib plots (Module 5)
    PoolController.sol    - On-chain pool + FDC verification (Solidity)
    orchestrator.py       - THIS FILE: main pipeline (Module 6)

RUN:
    python orchestrator.py
"""

import os
import sys
import time
import numpy as np

# Import our modules
from ftso_oracle import FTSOOracle
from qaoa_pathfinder import QAOAPathfinder
from mixing_pool import MixingPool
from fdc_settlement import FDCSettlement
from visualiser import plot_full_analysis, plot_qaoa_state


def main():
    print("=" * 70)
    print("  QUANTUM ANONYMOUS SWAP — FULL PIPELINE")
    print("  QAOA-Optimised Cross-Chain Mixing with FTSO + FDC")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Initialise FTSO Oracle
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 1: FTSO ORACLE — Historical Price Data")
    print(f"{'─'*70}")

    ftso = FTSOOracle(seed=42)
    ftso.summary()

    # Coin safety ranking
    print(f"\n  Coin safety ranking (10-min hold):")
    safety = ftso.rank_coins_by_safety(10)
    for rank, (sym, risk) in enumerate(safety):
        bar = "█" * int(risk * 500) + "░" * (30 - int(risk * 500))
        print(f"    {rank+1}. {sym:<6} risk={risk:.5f}  {bar}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Build QAOA Pathfinder
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 2: QAOA PATHFINDER — Quantum Route Optimisation")
    print(f"{'─'*70}")

    qaoa = QAOAPathfinder(ftso, qaoa_depth=2, penalty_weight=5.0)
    n_edges = qaoa.build_graph(max_edges=10)
    print(f"  Graph: {len(qaoa.nodes)} nodes, {n_edges} edges")

    print(f"\n  Running QAOA (p=2, {n_edges} qubits, 2^{n_edges}={2**n_edges} states)...")
    qaoa_result = qaoa.solve(max_iter=100)

    print(f"\n  QAOA Result:")
    print(f"    Optimal cost: {qaoa_result['cost']:.6f}")
    print(f"    Gamma: {qaoa_result.get('gamma', [])}")
    print(f"    Beta:  {qaoa_result.get('beta', [])}")
    print(f"    Route edges: {len(qaoa_result['route'])}")

    if qaoa_result['route']:
        print(f"\n  Optimal route (lowest risk path):")
        for e in qaoa_result['route']:
            print(f"    {e.from_coin}@{e.from_chain} -> {e.to_coin}@{e.to_chain}  "
                  f"cost={e.cost:.4f} hold={e.hold_minutes:.0f}min")

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Configure Mixing Pool
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 3: MIXING POOL — Aggregate + Redistribute")
    print(f"{'─'*70}")

    pool = MixingPool(
        ftso=ftso,
        qaoa_pathfinder=qaoa,
        hop_interval_minutes=5.0,  # FIXED interval between hops
        n_hops=4,                   # 4 redistribution rounds
        n_temp_wallets=6,           # 6 temp wallets per round
    )

    # Add buyers (all go into one pool)
    buyers = [
        ("0xBuyer1_aaa111bbb222ccc333ddd444eee555fff666", 25000),
        ("0xBuyer2_777888999aaabbbcccdddeeefff000111222", 15000),
        ("0xBuyer3_333444555666777888999aaabbbcccdddeee", 18000),
        ("0xBuyer4_fffeeeddddcccbbbaaa999888777666555444", 12000),
        ("0xBuyer5_111222333444555666777888999aaabbbcccc", 10000),
    ]

    for addr, amount in buyers:
        pool.add_buyer(addr, amount)
        print(f"  Buyer: {addr[:24]}... deposits ${amount:,.2f} USDC")

    # Add sellers (each gets exact amount — default to Plasma for zero-fee settlement)
    sellers = [
        ("0xSeller1_abc123def456abc123def456abc123def456", 22000, "plasma"),
        ("0xSeller2_789012ghi345789012ghi345789012ghi345", 18500, "plasma"),
        ("0xSeller3_456789jkl012456789jkl012456789jkl012", 14000, "plasma"),
        ("0xSeller4_321098mno654321098mno654321098mno654", 12500, "plasma"),
        ("0xSeller5_654321pqr987654321pqr987654321pqr987", 13000, "plasma"),
    ]

    for addr, amount, chain in sellers:
        pool.add_seller(addr, amount, chain)
        print(f"  Seller: {addr[:24]}... expects ${amount:,.2f} USDC on {chain}")

    total_in = sum(a for _, a in buyers)
    total_out = sum(a for _, a, _ in sellers)
    print(f"\n  Total in:  ${total_in:,.2f} USDC")
    print(f"  Total out: ${total_out:,.2f} USDC")

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Execute Mixing
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 4: EXECUTE MIXING — Pool → Random Split → QAOA Hops → Settle")
    print(f"{'─'*70}")

    settlement_result = pool.run_full_mixing()

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: FDC Verification
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 5: FDC VERIFICATION — Prove Sellers Received Funds")
    print(f"{'─'*70}")

    fdc = FDCSettlement()
    fdc_results = fdc.verify_batch(settlement_result["settlements"])

    report = fdc.generate_report()
    print(f"\n{report}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: Visualisation
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 6: VISUALISATION")
    print(f"{'─'*70}")

    out_dir = os.path.dirname(os.path.abspath(__file__))

    main_path = os.path.join(out_dir, "quantum_swap_analysis.png")
    plot_full_analysis(ftso, pool, qaoa_result, fdc_results, main_path)

    qaoa_path = os.path.join(out_dir, "qaoa_quantum_state.png")
    plot_qaoa_state(qaoa_result, qaoa_path)

    # ─────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE — SUMMARY")
    print(f"{'='*70}")

    print(f"""
  FTSO Oracle:
    Coins tracked:         {len(ftso.coins)}
    Historical data:       30 days × 1-min resolution
    Safest coin (10min):   {safety[0][0]} (risk={safety[0][1]:.6f})
    Riskiest coin (10min): {safety[-1][0]} (risk={safety[-1][1]:.6f})

  QAOA Pathfinder:
    Qubits:                {qaoa_result['n_qubits']}
    Hilbert space:         2^{qaoa_result['n_qubits']} = {2**qaoa_result['n_qubits']} states
    Circuit depth (p):     {len(qaoa_result.get('gamma', []))}
    Optimal cost:          {qaoa_result['cost']:.6f}
    Route edges:           {len(qaoa_result['route'])}

  Mixing Pool (Plasma Native):
    Buyers:                {len(pool.buyers)}
    Sellers:               {len(pool.sellers)}
    Pool total:            ${pool.pool_total:,.2f} USDT
    Reserve:               NONE (100% mixed for max anonymity)
    Chain:                 Plasma (zero-fee USDT via paymaster)
    Hop interval:          {pool.hop_interval:.0f} min (FIXED)
    Total hops:            {len(pool.hop_history)}
    Wallets created:       {settlement_result['total_wallets_created']}
    Wallets BURNED:        {settlement_result['total_wallets_burned']}
    Plasma free transfers: {settlement_result['plasma_free_transfers']}
    Cross-chain swaps:     {settlement_result['cross_chain_swaps']}
    Overall loss:          {settlement_result['overall_loss_pct']:.3f}%

  Settlement:
    Recovered:             ${settlement_result['total_recovered']:,.2f}
    Total owed:            ${settlement_result['total_owed']:,.2f}
    Surplus/deficit:       ${settlement_result['surplus_deficit']:,.2f}
    All sent simultaneously: YES (zero fee on Plasma)

  FDC Verification:
    Settlements verified:  {sum(1 for r in fdc_results if r.verified)}/{len(fdc_results)}
    Merkle proofs:         {sum(r.merkle_proof_nodes for r in fdc_results)} total nodes

  Smart Contracts:
    PoolController.sol     Pool deposits + commitment hashes + FDC verification
    FDCVerifiedSettlement.sol  (from previous session)

  Output Files:
    {main_path}
    {qaoa_path}
""")

    print(f"  BUILT ON PLASMA: Zero-fee USDT transfers mean on-chain hops")
    print(f"  cost nothing. Ephemeral wallets are burned after each hop —")
    print(f"  private keys zeroed, records wiped. Combined with QAOA quantum")
    print(f"  routing and Flare FTSO/FDC, this creates a fully anonymous,")
    print(f"  cryptographically verified payment system.")
    print(f"\n{'='*70}")

    return {
        "ftso": ftso,
        "qaoa_result": qaoa_result,
        "pool": pool,
        "settlement": settlement_result,
        "fdc_results": fdc_results,
    }


if __name__ == "__main__":
    result = main()

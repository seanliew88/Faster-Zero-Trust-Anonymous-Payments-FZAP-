"""
MODULE 5: VISUALISER
=====================
Matplotlib visualisation of:
    1. QAOA convergence (cost function over iterations)
    2. Route graph (payer -> temp wallets -> payee with edge weights)
    3. Pool value over hops (tracking losses)
    4. QAOA probability distribution (quantum state)
    5. Coin volatility heatmap (FTSO data)
    6. Complete flow diagram: buyers -> pool -> hops -> sellers
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
from typing import List, Dict, Optional


# Colour scheme
COLORS = {
    "USDC": "#2ecc71", "BTC": "#f7931a", "ETH": "#627eea",
    "SOL": "#9945ff", "AVAX": "#e84142", "XRP": "#23292f",
    "FLR": "#e62058", "MATIC": "#8247e5", "LTC": "#345d9d",
    "DOGE": "#c3a634", "pool": "#3498db", "buyer": "#2c3e50",
    "seller": "#27ae60",
}


def plot_full_analysis(ftso, pool, qaoa_result, fdc_results,
                       save_path: str = "quantum_swap_analysis.png"):
    """Generate the complete 6-panel analysis figure."""

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("Quantum Anonymous Swap — QAOA-Optimised Mixing Analysis",
                 fontsize=16, fontweight='bold', y=0.98)

    # Layout: 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.96, top=0.93, bottom=0.05)

    # === Panel 1: Complete Flow Diagram ===
    ax1 = fig.add_subplot(gs[0, :])
    _plot_flow_diagram(ax1, pool)

    # === Panel 2: QAOA Convergence ===
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_qaoa_convergence(ax2, qaoa_result, pool)

    # === Panel 3: Pool Value Over Hops ===
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_pool_value(ax3, pool)

    # === Panel 4: Route Graph ===
    ax4 = fig.add_subplot(gs[2, 0])
    _plot_route_graph(ax4, pool, ftso)

    # === Panel 5: Coin Risk Heatmap ===
    ax5 = fig.add_subplot(gs[2, 1])
    _plot_risk_heatmap(ax5, ftso)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n  [VIS] Saved to {save_path}")
    plt.close()

    return save_path


def _plot_flow_diagram(ax, pool):
    """Panel 1: Buyers -> Pool -> Hops -> Sellers flow diagram."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Transaction Flow: Buyers → Pool → QAOA Hops → Sellers",
                 fontsize=12, fontweight='bold', pad=10)

    # Buyers (left)
    n_buyers = min(len(pool.buyers), 5)
    for i in range(n_buyers):
        y = 2.5 - i * (4.0 / max(n_buyers - 1, 1))
        box = FancyBboxPatch((0, y - 0.25), 1.2, 0.5,
                             boxstyle="round,pad=0.1",
                             facecolor=COLORS["buyer"], alpha=0.8)
        ax.add_patch(box)
        ax.text(0.6, y, f"Buyer {i+1}\n${pool.buyers[i].amount_usdt:,.0f}",
                ha='center', va='center', fontsize=6, color='white', fontweight='bold')

    # Pool (centre-left)
    pool_box = FancyBboxPatch((2, -0.5), 1.5, 2,
                              boxstyle="round,pad=0.15",
                              facecolor=COLORS["pool"], alpha=0.9)
    ax.add_patch(pool_box)
    ax.text(2.75, 0.5, f"USDC\nPOOL\n${pool.pool_total:,.0f}",
            ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # No reserve in Plasma version — label the direct flow
    ax.text(6.25, 2.8, "100% mixed via Plasma (zero-fee USDT hops)",
            ha='center', va='center', fontsize=7, color='#2ecc71',
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))

    # Arrows: buyers -> pool
    for i in range(n_buyers):
        y = 2.5 - i * (4.0 / max(n_buyers - 1, 1))
        ax.annotate('', xy=(2, 0.5), xytext=(1.2, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS["buyer"],
                                    lw=1.5, alpha=0.5))

    # Hop stages
    hop_x_positions = [4.2, 5.8, 7.4]
    for h_idx, hop_x in enumerate(hop_x_positions):
        hop_num = min(h_idx + 1, len(pool.hop_history))
        if h_idx < len(pool._viz_wallet_snapshots):
            wallets_data = pool._viz_wallet_snapshots[h_idx]
        else:
            wallets_data = [{"coin": "?", "amount": 0}] * 3

        # Show up to 3 wallets per hop
        n_show = min(len(wallets_data), 3)
        for w_idx in range(n_show):
            wd = wallets_data[w_idx]
            y = 1.5 - w_idx * 1.5
            color = COLORS.get(wd.get("coin", ""), "#95a5a6")
            box = FancyBboxPatch((hop_x, y - 0.3), 1.0, 0.6,
                                 boxstyle="round,pad=0.08",
                                 facecolor=color, alpha=0.75)
            ax.add_patch(box)
            ax.text(hop_x + 0.5, y, f"{wd.get('coin','?')}\n${wd.get('amount',0):,.0f}",
                    ha='center', va='center', fontsize=5.5, color='white',
                    fontweight='bold')

        # Hop label
        ax.text(hop_x + 0.5, 2.8, f"Hop {h_idx+1}",
                ha='center', fontsize=7, fontstyle='italic', color='gray')

        # Arrow from previous stage
        if h_idx == 0:
            ax.annotate('', xy=(hop_x, 0.5), xytext=(3.5, 0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        else:
            prev_x = hop_x_positions[h_idx - 1] + 1.0
            ax.annotate('', xy=(hop_x, 0.5), xytext=(prev_x, 0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Sellers (right)
    n_sellers = min(len(pool.sellers), 5)
    for i in range(n_sellers):
        y = 2.5 - i * (4.0 / max(n_sellers - 1, 1))
        box = FancyBboxPatch((9, y - 0.25), 1.2, 0.5,
                             boxstyle="round,pad=0.1",
                             facecolor=COLORS["seller"], alpha=0.8)
        ax.add_patch(box)
        ax.text(9.6, y, f"Seller {i+1}\n${pool.sellers[i].amount_usdt:,.0f}",
                ha='center', va='center', fontsize=6, color='white', fontweight='bold')

    # Arrow from last hop to sellers
    last_hop_x = hop_x_positions[-1] + 1.0
    ax.annotate('', xy=(9, 0.5), xytext=(last_hop_x, 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS["seller"], lw=2))
    ax.text((last_hop_x + 9) / 2, 1.1, "Simultaneous\nSettlement",
            ha='center', fontsize=7, fontstyle='italic', color=COLORS["seller"])

    # QAOA label
    ax.text(5.8, -2.0, "QAOA optimises each hop to minimise volatility risk",
            ha='center', fontsize=8, fontstyle='italic',
            color=COLORS["pool"], alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))


def _plot_qaoa_convergence(ax, qaoa_result, pool):
    """Panel 2: QAOA cost function convergence over iterations."""
    # Collect convergence from standalone QAOA result only
    all_convergence = []
    conv = qaoa_result.get("convergence", [])
    if conv:
        all_convergence.append(conv)

    if not all_convergence:
        ax.text(0.5, 0.5, "No convergence data", ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("QAOA Convergence")
        return

    # Plot each hop's convergence
    cmap = plt.cm.viridis
    for i, conv in enumerate(all_convergence):
        color = cmap(i / max(len(all_convergence) - 1, 1))
        ax.plot(conv, color=color, alpha=0.7, linewidth=1.5,
                label=f'Hop {i+1}')

        # Mark minimum
        min_idx = np.argmin(conv)
        ax.plot(min_idx, conv[min_idx], 'v', color=color, markersize=8)

    ax.set_xlabel('QAOA Iteration', fontsize=10)
    ax.set_ylabel('Cost Function ⟨ψ|H_C|ψ⟩', fontsize=10)
    ax.set_title('QAOA Convergence per Hop', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    # Add QAOA info box
    n_qubits = qaoa_result.get("n_qubits", "?")
    depth = len(qaoa_result.get("gamma", []))
    info = f"Qubits: {n_qubits}\nDepth p={depth}\nGates: {n_qubits * depth * 2}"
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


def _plot_pool_value(ax, pool):
    """Panel 3: Pool value tracking through hops."""
    hops = [0]
    values = [pool.pool_total]

    for h in pool.hop_history:
        hops.append(h.hop_number)
        values.append(h.total_value)

    ax.plot(hops, values, 'b-o', linewidth=2, markersize=7, label='Pool Value')
    ax.axhline(pool.pool_total, color='gray', linestyle='--', alpha=0.4,
               label=f'Initial: ${pool.pool_total:,.0f}')
    ax.fill_between(hops, values, pool.pool_total,
                    where=[v < pool.pool_total for v in values],
                    alpha=0.15, color='red', label='Loss region')

    for i, (h, v) in enumerate(zip(hops, values)):
        loss = (1 - v / pool.pool_total) * 100
        ax.annotate(f'${v:,.0f}\n(-{loss:.2f}%)',
                    (h, v), textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=6.5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Annotate free vs swap hops
    total_free = sum(h.plasma_transfers for h in pool.hop_history)
    total_swap = sum(h.cross_chain_swaps for h in pool.hop_history)
    ax.text(0.02, 0.02, f"Plasma free: {total_free}\nCross-chain: {total_swap}",
            transform=ax.transAxes, fontsize=7, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Hop Number', fontsize=10)
    ax.set_ylabel('Value (USDT)', fontsize=10)
    ax.set_title('Pool Value Through Hops (no reserve)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xticks(hops)


def _plot_route_graph(ax, pool, ftso):
    """Panel 4: Network graph showing wallet connections via viz snapshots."""
    G = nx.DiGraph()

    # Build nodes from viz snapshots
    for snap_idx, snapshot in enumerate(pool._viz_wallet_snapshots):
        for wd in snapshot:
            wid = wd["id"]
            coin = wd["coin"]
            G.add_node(wid, label=f"{coin}", coin=coin,
                       hop=snap_idx, amount=wd["amount"])

    # Edges between consecutive snapshots
    for s_idx in range(1, len(pool._viz_wallet_snapshots)):
        prev = pool._viz_wallet_snapshots[s_idx - 1]
        curr = pool._viz_wallet_snapshots[s_idx]
        for p in prev:
            for c in curr:
                G.add_edge(p["id"], c["id"], weight=0.3)

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "No route data", ha='center', va='center',
                transform=ax.transAxes)
        return

    # Layout: arrange by hop number
    pos = {}
    hop_groups = {}
    for node, data in G.nodes(data=True):
        h = data.get('hop', 0)
        if h not in hop_groups:
            hop_groups[h] = []
        hop_groups[h].append(node)

    for h, nodes in hop_groups.items():
        for i, node in enumerate(nodes):
            x = h * 2.0
            y = (i - len(nodes)/2) * 1.2
            pos[node] = (x, y)

    # Draw
    node_colors = [COLORS.get(G.nodes[n].get('coin', ''), '#95a5a6') for n in G.nodes]
    node_sizes = [max(G.nodes[n].get('amount', 100) / 50, 100) for n in G.nodes]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray',
                           arrows=True, arrowsize=10, width=0.8,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.8, edgecolors='black',
                           linewidths=0.5)

    # Labels
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=5)

    ax.set_title('Wallet Network (coloured by coin)', fontsize=11, fontweight='bold')

    # Legend
    unique_coins = set(G.nodes[n].get('coin', '') for n in G.nodes)
    patches = [mpatches.Patch(color=COLORS.get(c, '#95a5a6'), label=c)
               for c in sorted(unique_coins) if c]
    ax.legend(handles=patches, fontsize=6, loc='lower right', ncol=2)


def _plot_risk_heatmap(ax, ftso):
    """Panel 5: Coin volatility risk heatmap for different hold times."""
    coins = sorted([s for s in ftso.coins.keys() if s != "USDC"])
    hold_times = [1, 3, 5, 10, 15, 30, 60]

    risk_matrix = np.zeros((len(coins), len(hold_times)))
    for i, coin in enumerate(coins):
        for j, t in enumerate(hold_times):
            risk_matrix[i, j] = ftso.get_risk_score(coin, t) * 100

    im = ax.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(hold_times)))
    ax.set_xticklabels([f'{t}min' for t in hold_times], fontsize=8)
    ax.set_yticks(range(len(coins)))
    ax.set_yticklabels(coins, fontsize=9)

    # Annotate cells
    for i in range(len(coins)):
        for j in range(len(hold_times)):
            color = 'white' if risk_matrix[i, j] > risk_matrix.max() * 0.5 else 'black'
            ax.text(j, i, f'{risk_matrix[i,j]:.3f}%',
                    ha='center', va='center', fontsize=6, color=color)

    ax.set_title('Volatility Risk by Coin × Hold Time (FTSO)',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Risk (%)', shrink=0.8)


def plot_qaoa_state(qaoa_result, save_path: str = "qaoa_state.png"):
    """
    Separate detailed view of QAOA quantum state.
    Shows probability distribution over all basis states.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Probability distribution
    ax1 = axes[0]
    probs = qaoa_result.get("probabilities", np.array([]))
    if len(probs) > 0:
        n_states = len(probs)
        n_qubits = qaoa_result.get("n_qubits", int(np.log2(n_states)))

        # Show top 30 most probable states
        top_indices = np.argsort(probs)[-30:][::-1]
        top_probs = probs[top_indices]
        top_labels = [format(i, f'0{n_qubits}b') for i in top_indices]

        bars = ax1.bar(range(len(top_probs)), top_probs, color='#3498db', alpha=0.8)

        # Highlight the solution
        counts = qaoa_result.get("counts", {})
        if counts:
            best_bitstring = max(counts, key=counts.get)
            for i, label in enumerate(top_labels):
                if label == best_bitstring:
                    bars[i].set_color('#e74c3c')
                    bars[i].set_alpha(1.0)

        ax1.set_xticks(range(len(top_labels)))
        ax1.set_xticklabels(top_labels, rotation=90, fontsize=6)
        ax1.set_ylabel('Probability', fontsize=10)
        ax1.set_title(f'QAOA State: Top {len(top_probs)} of {n_states} basis states',
                      fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No state data", ha='center', va='center',
                 transform=ax1.transAxes)

    # Panel 2: Convergence
    ax2 = axes[1]
    conv = qaoa_result.get("convergence", [])
    if conv:
        ax2.plot(conv, 'b-', alpha=0.6, linewidth=0.8)
        # Smoothed line
        window = max(len(conv) // 20, 1)
        smoothed = np.convolve(conv, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(conv)), smoothed, 'r-', linewidth=2,
                 label='Smoothed')
        ax2.axhline(min(conv), color='green', linestyle='--', alpha=0.5,
                    label=f'Best: {min(conv):.4f}')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('⟨ψ|H_C|ψ⟩')
        ax2.set_title('QAOA Variational Optimisation', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [VIS] QAOA state saved to {save_path}")
    plt.close()

    return save_path

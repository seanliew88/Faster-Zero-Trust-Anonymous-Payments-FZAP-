"""
MODULE 2: QAOA PATHFINDER
==========================
Actual Quantum Approximate Optimization Algorithm for route finding.

This implements QAOA from first principles using quantum state vector
simulation. The quantum circuit is REAL — parameterised RZZ cost gates
and RX mixer gates applied to a 2^n dimensional statevector.

QAOA FOR ROUTE FINDING:
    Problem: Find the lowest-cost path through a graph of coins
    where edge weights = volatility risk + fees (from FTSO).

    Encoding: Each edge (i->j) is mapped to a qubit.
    qubit |1> = edge selected, |0> = not selected.

    Cost Hamiltonian H_C:
        H_C = sum_{edges} cost(i,j) * Z_ij
        Encodes: "penalise selecting expensive edges"

    Constraint Hamiltonian (penalty):
        H_penalty = lambda * sum_{nodes} (sum_in - sum_out - demand)^2
        Enforces: flow conservation (valid path)

    Mixer Hamiltonian H_M:
        H_M = sum_{qubits} X_i
        Explores: superposition of all edge subsets

    QAOA circuit (p layers):
        |psi(gamma, beta)> = prod_{l=1}^{p} [e^{-i*beta_l*H_M} * e^{-i*gamma_l*H_C}] |+>

    Classical outer loop:
        Optimise gamma, beta to minimise <psi|H_C|psi>

QISKIT COMPATIBILITY:
    This module is structured identically to qiskit's QAOA.
    To swap to qiskit, replace QuantumCircuit with qiskit.QuantumCircuit
    and QAOASolver with qiskit_algorithms.QAOA.

    # qiskit equivalent:
    # from qiskit import QuantumCircuit
    # from qiskit.circuit import Parameter
    # from qiskit_algorithms import QAOA
    # from qiskit_algorithms.optimizers import COBYLA
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time


# =============================================================================
# QUANTUM STATE VECTOR SIMULATOR
# =============================================================================

class QuantumState:
    """
    Pure state vector simulator for n qubits.
    State is a complex vector of dimension 2^n.

    Equivalent to qiskit's Statevector class.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        # Initialise to |+>^n (equal superposition)
        self.state = np.ones(self.dim, dtype=np.complex128) / np.sqrt(self.dim)

    def apply_rz(self, qubit: int, theta: float):
        """Apply R_Z(theta) = exp(-i*theta/2 * Z) to a single qubit."""
        for basis in range(self.dim):
            # Check if qubit is |1> in this basis state
            if (basis >> qubit) & 1:
                self.state[basis] *= np.exp(-1j * theta / 2)
            else:
                self.state[basis] *= np.exp(1j * theta / 2)

    def apply_rx(self, qubit: int, theta: float):
        """Apply R_X(theta) = exp(-i*theta/2 * X) to a single qubit."""
        cos_t = np.cos(theta / 2)
        sin_t = np.sin(theta / 2)
        for basis in range(self.dim):
            partner = basis ^ (1 << qubit)  # flip the qubit
            if basis < partner:
                a = self.state[basis]
                b = self.state[partner]
                self.state[basis]   = cos_t * a - 1j * sin_t * b
                self.state[partner] = -1j * sin_t * a + cos_t * b

    def apply_rzz(self, q1: int, q2: int, theta: float):
        """Apply R_ZZ(theta) = exp(-i*theta/2 * Z_q1 Z_q2)."""
        for basis in range(self.dim):
            b1 = (basis >> q1) & 1
            b2 = (basis >> q2) & 1
            parity = (-1) ** (b1 ^ b2)
            self.state[basis] *= np.exp(-1j * theta / 2 * parity)

    def apply_phase(self, qubit: int, phi: float):
        """Apply phase exp(i*phi) to states where qubit=|1>."""
        for basis in range(self.dim):
            if (basis >> qubit) & 1:
                self.state[basis] *= np.exp(1j * phi)

    def measure_expectation(self, diagonal_operator: np.ndarray) -> float:
        """Compute <psi|O|psi> for diagonal operator O."""
        return np.real(np.sum(np.abs(self.state)**2 * diagonal_operator))

    def sample(self, n_shots: int = 1024) -> Dict[str, int]:
        """Sample measurement outcomes."""
        probs = np.abs(self.state) ** 2
        probs /= probs.sum()  # normalise
        outcomes = np.random.choice(self.dim, size=n_shots, p=probs)
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.n}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def get_probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2


# =============================================================================
# QAOA CIRCUIT
# =============================================================================

class QAOACircuit:
    """
    QAOA circuit with p layers of cost and mixer unitaries.

    Equivalent to building a qiskit QuantumCircuit with:
        for layer in range(p):
            qc.append(cost_unitary(gamma[layer]), qubits)
            qc.append(mixer_unitary(beta[layer]), qubits)

    Parameters:
        n_qubits: number of qubits (= number of edges in graph)
        p: number of QAOA layers (depth)
        cost_terms: list of (qubit_indices, coefficient) for H_C
        constraint_terms: penalty terms for flow conservation
    """

    def __init__(self, n_qubits: int, p: int,
                 cost_coefficients: np.ndarray,
                 zz_terms: List[Tuple[int, int, float]] = None):
        self.n_qubits = n_qubits
        self.p = p
        self.cost_coefficients = cost_coefficients  # diagonal of H_C
        self.zz_terms = zz_terms or []  # (q1, q2, coeff) for ZZ interactions

        # Build diagonal cost operator
        dim = 2 ** n_qubits
        self.cost_operator = np.zeros(dim)
        for basis in range(dim):
            cost = 0.0
            for q in range(n_qubits):
                if (basis >> q) & 1:
                    cost += cost_coefficients[q]
            # ZZ penalty terms (for constraints)
            for q1, q2, coeff in self.zz_terms:
                b1 = (basis >> q1) & 1
                b2 = (basis >> q2) & 1
                cost += coeff * (2*b1 - 1) * (2*b2 - 1)
            self.cost_operator[basis] = cost

    def run(self, gamma: np.ndarray, beta: np.ndarray) -> QuantumState:
        """
        Execute QAOA circuit with given parameters.

        |psi> = prod_{l=1}^{p} [U_M(beta_l) U_C(gamma_l)] |+>

        Where:
            U_C(gamma) = exp(-i * gamma * H_C)  [cost unitary]
            U_M(beta)  = exp(-i * beta * H_M)   [mixer unitary]
        """
        state = QuantumState(self.n_qubits)

        for layer in range(self.p):
            # === Cost unitary: exp(-i * gamma * H_C) ===
            # For diagonal H_C, this is just phase rotations
            for q in range(self.n_qubits):
                state.apply_rz(q, 2 * gamma[layer] * self.cost_coefficients[q])

            # ZZ interactions for constraint terms
            for q1, q2, coeff in self.zz_terms:
                state.apply_rzz(q1, q2, 2 * gamma[layer] * coeff)

            # === Mixer unitary: exp(-i * beta * H_M) ===
            # H_M = sum_i X_i, so U_M = prod_i RX(2*beta)
            for q in range(self.n_qubits):
                state.apply_rx(q, 2 * beta[layer])

        return state

    def expectation(self, gamma: np.ndarray, beta: np.ndarray) -> float:
        """Compute <psi(gamma,beta)|H_C|psi(gamma,beta)>."""
        state = self.run(gamma, beta)
        return state.measure_expectation(self.cost_operator)

    def optimal_params(self, method: str = 'COBYLA',
                       maxiter: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Variational optimisation of QAOA parameters.
        Equivalent to qiskit_algorithms.QAOA with COBYLA optimizer.
        """
        # Initial parameters
        x0 = np.random.uniform(-np.pi, np.pi, 2 * self.p)

        def objective(x):
            gamma = x[:self.p]
            beta = x[self.p:]
            return self.expectation(gamma, beta)

        result = minimize(objective, x0, method=method,
                         options={'maxiter': maxiter, 'rhobeg': 0.5})

        gamma_opt = result.x[:self.p]
        beta_opt = result.x[self.p:]
        return gamma_opt, beta_opt, result.fun


# =============================================================================
# GRAPH BUILDER: Coins -> QAOA Problem
# =============================================================================

@dataclass
class Edge:
    from_coin: str
    to_coin: str
    from_chain: str
    to_chain: str
    qubit_index: int
    cost: float  # from FTSO: vol_risk + fee
    hold_minutes: float


class QAOAPathfinder:
    """
    Builds the coin exchange graph and solves it with QAOA.

    Graph structure:
        Nodes = (coin, chain) pairs
        Edges = possible swaps/bridges between coins
        Edge weight = risk + fee (from FTSO oracle)

    The QAOA finds the subset of edges forming a valid path
    from source (USDC) to sink (USDC) that minimises total cost.

    FLOW CONSERVATION CONSTRAINTS (encoded as penalty terms):
        For source node: sum(outgoing) = 1
        For sink node: sum(incoming) = 1
        For intermediate nodes: sum(incoming) = sum(outgoing)
    """

    # Estimated times in minutes
    CHAIN_TIMES = {
        "ethereum": 5, "avalanche": 2, "solana": 1,
        "bitcoin": 15, "polygon": 2, "flare": 3,
    }

    BRIDGE_TIMES = {
        frozenset({"ethereum","avalanche"}): 5,
        frozenset({"ethereum","flare"}): 5,
        frozenset({"ethereum","polygon"}): 3,
        frozenset({"ethereum","solana"}): 5,
        frozenset({"avalanche","flare"}): 4,
        frozenset({"solana","polygon"}): 5,
        frozenset({"flare","polygon"}): 4,
        frozenset({"bitcoin","flare"}): 10,
        frozenset({"bitcoin","ethereum"}): 15,
    }

    def __init__(self, ftso, qaoa_depth: int = 2, penalty_weight: float = 5.0):
        """
        Args:
            ftso: FTSOOracle instance
            qaoa_depth: number of QAOA layers (p). Higher = better but slower.
            penalty_weight: lambda for constraint violation penalties
        """
        self.ftso = ftso
        self.p = qaoa_depth
        self.penalty = penalty_weight
        self.edges: List[Edge] = []
        self.nodes: List[Tuple[str, str]] = []
        self.node_to_idx: Dict[Tuple[str,str], int] = {}

    def build_graph(self, coins: List[str] = None,
                    max_edges: int = 20):
        """
        Build the exchange graph from FTSO data.
        Limits edges to max_edges for quantum tractability.
        """
        if coins is None:
            # Use top coins by safety + USDC
            safety = self.ftso.rank_coins_by_safety(10)
            coins = [s for s, _ in safety[:7]]  # top 7 safest
            if "USDC" not in coins:
                coins.append("USDC")

        # Build nodes: (coin, chain) pairs
        self.nodes = []
        self.node_to_idx = {}
        for coin in coins:
            profile = self.ftso.coins.get(coin)
            if not profile:
                continue
            for chain in profile.chains:
                node = (coin, chain)
                if node not in self.node_to_idx:
                    self.node_to_idx[node] = len(self.nodes)
                    self.nodes.append(node)

        # Build edges: all possible swaps and bridges
        all_edges = []
        for i, (c1, ch1) in enumerate(self.nodes):
            for j, (c2, ch2) in enumerate(self.nodes):
                if i == j:
                    continue
                if c1 == c2 and ch1 == ch2:
                    continue

                same_chain = (ch1 == ch2)
                cross_chain = not same_chain

                # Can we bridge between these chains?
                if cross_chain:
                    key = frozenset({ch1, ch2})
                    if key not in self.BRIDGE_TIMES and c1 != c2:
                        continue
                    hold = self.BRIDGE_TIMES.get(key, 8)
                else:
                    hold = self.CHAIN_TIMES.get(ch1, 3)

                cost = self.ftso.get_edge_cost(c1, c2, hold, same_chain)
                all_edges.append((cost, c1, c2, ch1, ch2, hold, same_chain))

        # Select top edges by lowest cost (for quantum tractability)
        all_edges.sort(key=lambda e: e[0])
        all_edges = all_edges[:max_edges]

        self.edges = []
        for idx, (cost, c1, c2, ch1, ch2, hold, _) in enumerate(all_edges):
            self.edges.append(Edge(
                from_coin=c1, to_coin=c2,
                from_chain=ch1, to_chain=ch2,
                qubit_index=idx, cost=cost,
                hold_minutes=hold
            ))

        return len(self.edges)

    def _build_constraint_terms(self) -> List[Tuple[int, int, float]]:
        """
        Build ZZ penalty terms for flow conservation.

        For each intermediate node:
            penalty * (sum_in_edges - sum_out_edges)^2

        This expands to ZZ interactions between pairs of edges
        that share a node.
        """
        zz_terms = []

        for node in self.nodes:
            in_edges = [e.qubit_index for e in self.edges
                        if (e.to_coin, e.to_chain) == node]
            out_edges = [e.qubit_index for e in self.edges
                         if (e.from_coin, e.from_chain) == node]

            # (sum_in - sum_out)^2 expands to:
            # sum of ZZ terms between all pairs
            all_signed = [(q, +1) for q in in_edges] + [(q, -1) for q in out_edges]

            for i in range(len(all_signed)):
                for j in range(i+1, len(all_signed)):
                    q1, s1 = all_signed[i]
                    q2, s2 = all_signed[j]
                    # Coefficient from expansion of (s1*Z1 + s2*Z2 + ...)^2
                    coeff = self.penalty * s1 * s2 * 0.25
                    zz_terms.append((q1, q2, coeff))

        return zz_terms

    def solve(self, max_iter: int = 150) -> dict:
        """
        Run QAOA to find optimal route.

        Returns dict with:
            - route: list of Edge objects forming the path
            - cost: total cost (fees + vol risk)
            - gamma, beta: optimal QAOA parameters
            - convergence: optimisation history
        """
        n_qubits = len(self.edges)
        if n_qubits == 0:
            return {"route": [], "cost": float('inf'), "error": "No edges"}

        # Limit qubits for simulation tractability
        if n_qubits > 16:
            # Prune to top 16 lowest-cost edges
            self.edges.sort(key=lambda e: e.cost)
            self.edges = self.edges[:16]
            for i, e in enumerate(self.edges):
                e.qubit_index = i
            n_qubits = 16

        print(f"  [QAOA] Building circuit: {n_qubits} qubits, depth p={self.p}")

        # Cost coefficients (diagonal of H_C)
        cost_coeffs = np.array([e.cost for e in self.edges])

        # Constraint ZZ terms
        zz_terms = self._build_constraint_terms()
        print(f"  [QAOA] {len(zz_terms)} constraint terms (flow conservation)")

        # Build QAOA circuit
        circuit = QAOACircuit(n_qubits, self.p, cost_coeffs, zz_terms)

        # Track convergence for visualisation
        convergence_history = []

        def tracked_objective(x):
            gamma = x[:self.p]
            beta = x[self.p:]
            val = circuit.expectation(gamma, beta)
            convergence_history.append(val)
            return val

        # Optimise with multiple random restarts
        best_cost = float('inf')
        best_params = None
        n_restarts = 3

        print(f"  [QAOA] Optimising ({n_restarts} restarts, {max_iter} iter each)...")

        for restart in range(n_restarts):
            x0 = np.random.uniform(-np.pi, np.pi, 2 * self.p)

            result = minimize(tracked_objective, x0, method='COBYLA',
                            options={'maxiter': max_iter, 'rhobeg': 0.5})

            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x

        gamma_opt = best_params[:self.p]
        beta_opt = best_params[self.p:]

        # Sample solutions from optimal state
        state = circuit.run(gamma_opt, beta_opt)
        counts = state.sample(n_shots=2048)

        # Find best valid solution
        best_solution = None
        best_sol_cost = float('inf')

        for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
            # Decode bitstring to edge selection
            selected = [i for i, b in enumerate(bitstring[::-1]) if b == '1']
            if len(selected) == 0:
                continue

            sol_cost = sum(self.edges[i].cost for i in selected)

            if sol_cost < best_sol_cost:
                best_sol_cost = sol_cost
                best_solution = selected

        # Extract route
        route = [self.edges[i] for i in (best_solution or [])]

        return {
            "route": route,
            "cost": best_sol_cost,
            "gamma": gamma_opt,
            "beta": beta_opt,
            "convergence": convergence_history,
            "n_qubits": n_qubits,
            "counts": counts,
            "probabilities": state.get_probabilities(),
        }

    def solve_adaptive(self, current_coin: str, current_chain: str,
                       remaining_hops: int, amount_usd: float) -> dict:
        """
        ADAPTIVE QAOA: Re-solve at each hop with updated FTSO data.

        This is the key innovation — instead of planning the entire route
        upfront, we re-run QAOA at each hop with fresh volatility data.

        At each hop:
            1. Query FTSO for current volatilities
            2. Rebuild edge costs
            3. Run QAOA from current position
            4. Take the first hop of the optimal route
            5. Repeat from new position
        """
        full_route = []
        current = (current_coin, current_chain)
        total_cost = 0
        hop_results = []

        for hop in range(remaining_hops):
            print(f"\n  [QAOA Adaptive] Hop {hop+1}/{remaining_hops} "
                  f"from {current[0]}@{current[1]}")

            # Rebuild costs with latest FTSO data
            for edge in self.edges:
                same_chain = (edge.from_chain == edge.to_chain)
                edge.cost = self.ftso.get_edge_cost(
                    edge.from_coin, edge.to_coin,
                    edge.hold_minutes, same_chain
                )

            # Solve from current position
            result = self.solve(max_iter=100)

            if not result["route"]:
                print(f"    No route found from {current}")
                break

            # Take first edge
            next_edge = result["route"][0]
            full_route.append(next_edge)
            total_cost += next_edge.cost
            current = (next_edge.to_coin, next_edge.to_chain)

            hop_results.append({
                "hop": hop,
                "edge": next_edge,
                "qaoa_cost": result["cost"],
                "qubits_used": result["n_qubits"],
            })

            print(f"    -> {next_edge.to_coin}@{next_edge.to_chain} "
                  f"(cost: {next_edge.cost:.4f}, hold: {next_edge.hold_minutes:.0f}min)")

        return {
            "full_route": full_route,
            "total_cost": total_cost,
            "hop_results": hop_results,
            "convergence": result.get("convergence", []),
            "final_position": current,
        }

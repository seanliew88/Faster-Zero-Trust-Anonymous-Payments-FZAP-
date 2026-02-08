"""
MODULE 1: FTSO ORACLE — Stablecoin-Native
===========================================
Flare Time Series Oracle for STABLECOIN price feeds.

STABLECOIN-ONLY DESIGN:
    Routes EXCLUSIVELY through stablecoins: USDT, USDC, DAI, FRAX, LUSD, PYUSD, cUSD
    No BTC/ETH/SOL — zero volatility risk during hops.
    Arbitrage from DEPEG SPREADS between stables on different chains.

OMNIBUS ACCOUNT REQUIREMENT:
    Omnibus = single wallet holding pooled funds from many users.

    ✅ ELIGIBLE:
        USDT (Tether), USDC (Circle), DAI (MakerDAO), FRAX (Frax),
        LUSD (Liquity), PYUSD (PayPal/Paxos), cUSD (Celo/Mento)

    ❌ NOT ELIGIBLE:
        BTC, ETH, SOL (volatile), stETH/rETH (rebasing), CBDCs (regulated)

    OMNIBUS-COMPATIBLE VENUES:
        ✅ Fireblocks (institutional omnibus vaults)
        ✅ Circle (native USDC omnibus)
        ✅ Curve Finance (permissionless stablecoin pools)
        ✅ Aave/Compound (lending pool = omnibus structure)
        ✅ Plasma DEX (native USDT omnibus via paymaster)
        ❌ Binance/Coinbase retail (segregated, not omnibus)

PRICE MODEL:
    Ornstein-Uhlenbeck (mean-reverting around $1.00 peg):
        dP = theta * (1.0 - P) * dt + sigma * dW
    Depeg spreads between stables on different chains = arb opportunity.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class StablecoinProfile:
    symbol: str
    peg_target: float
    peg_tightness: float    # std dev of depeg
    annual_vol: float
    chains: List[str]
    swap_fee: float
    bridge_fee: float
    omnibus_eligible: bool
    omnibus_venues: List[str]
    issuer: str
    collateral: str


STABLECOINS = {
    "USDT": StablecoinProfile(
        "USDT", 1.0, 0.0008, 0.01,
        ["ethereum", "avalanche", "polygon", "solana", "flare", "plasma"],
        0.0004, 0.001, True,
        ["fireblocks", "curve", "aave", "plasma_dex"],
        "Tether", "USD reserves + commercial paper"),
    "USDC": StablecoinProfile(
        "USDC", 1.0, 0.0005, 0.008,
        ["ethereum", "avalanche", "polygon", "solana", "flare", "plasma"],
        0.0004, 0.001, True,
        ["fireblocks", "circle", "curve", "aave", "compound", "plasma_dex"],
        "Circle", "USD reserves (short-dated treasuries)"),
    "DAI": StablecoinProfile(
        "DAI", 1.0, 0.002, 0.03,
        ["ethereum", "polygon", "avalanche", "plasma"],
        0.0004, 0.0015, True,
        ["curve", "aave", "compound", "maker_psm"],
        "MakerDAO", "ETH/USDC over-collateralised (150%+)"),
    "FRAX": StablecoinProfile(
        "FRAX", 1.0, 0.0015, 0.02,
        ["ethereum", "avalanche", "polygon", "plasma"],
        0.0004, 0.002, True,
        ["curve", "fraxswap", "aave"],
        "Frax Finance", "USDC + algorithmic (hybrid)"),
    "LUSD": StablecoinProfile(
        "LUSD", 1.0, 0.003, 0.04,
        ["ethereum", "polygon", "plasma"],
        0.0005, 0.002, True,
        ["curve", "uniswap_v3"],
        "Liquity", "ETH over-collateralised (110%+), immutable"),
    "PYUSD": StablecoinProfile(
        "PYUSD", 1.0, 0.001, 0.012,
        ["ethereum", "solana", "plasma"],
        0.0005, 0.0015, True,
        ["curve", "uniswap_v3"],
        "PayPal/Paxos", "USD deposits + treasuries"),
    "cUSD": StablecoinProfile(
        "cUSD", 1.0, 0.003, 0.035,
        ["ethereum", "polygon", "plasma"],
        0.0005, 0.002, True,
        ["curve", "uniswap_v3"],
        "Celo/Mento", "Diversified crypto reserves"),
}


class FTSOOracle:
    """
    FTSO Oracle for stablecoin price feeds.
    Uses Ornstein-Uhlenbeck process for mean-reverting peg dynamics.
    """

    def __init__(self, seed: int = None):
        self.coins = STABLECOINS
        self._seed = seed or int(time.time()) % 100000
        np.random.seed(self._seed)
        self.n_steps = 43200
        self.prices: Dict[str, np.ndarray] = {}
        self.log_returns: Dict[str, np.ndarray] = {}
        self._generate_histories()

    def _generate_histories(self):
        dt = 1.0 / (365 * 24 * 60)
        for sym, profile in self.coins.items():
            prices = np.zeros(self.n_steps)
            prices[0] = profile.peg_target + np.random.normal(0, profile.peg_tightness)
            theta = {
                "USDT": 100.0, "USDC": 100.0, "PYUSD": 80.0,
                "DAI": 20.0, "FRAX": 40.0, "LUSD": 15.0, "cUSD": 15.0,
            }.get(sym, 30.0)
            mu = profile.peg_target
            sigma = profile.annual_vol
            for t in range(1, self.n_steps):
                dW = np.random.standard_normal() * np.sqrt(dt)
                dp = theta * (mu - prices[t-1]) * dt + sigma * dW
                prices[t] = np.clip(prices[t-1] + dp, 0.95, 1.05)
            self.prices[sym] = prices
            self.log_returns[sym] = np.diff(np.log(prices))

    def get_current_price(self, symbol: str) -> float:
        return float(self.prices[symbol][-1]) if symbol in self.prices else 1.0

    def get_depeg(self, symbol: str) -> float:
        return self.get_current_price(symbol) - 1.0

    def get_volatility(self, symbol: str, window_minutes: int = 1440) -> float:
        rets = self.log_returns[symbol][-window_minutes:]
        return float(np.std(rets) * np.sqrt(525600))

    def get_risk_score(self, symbol: str, hold_minutes: int) -> float:
        vol = self.get_volatility(symbol)
        return vol * np.sqrt(hold_minutes / (365 * 24 * 60))

    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        symbols = sorted(self.log_returns.keys())
        rets = np.column_stack([self.log_returns[s][-10080:] for s in symbols])
        return symbols, np.corrcoef(rets.T)

    def rank_coins_by_safety(self, hold_minutes: int = 10) -> List[Tuple[str, float]]:
        scores = [(sym, self.get_risk_score(sym, hold_minutes)) for sym in self.coins]
        return sorted(scores, key=lambda x: x[1])

    def get_depeg_spreads(self) -> List[Tuple[str, str, float]]:
        spreads = []
        syms = list(self.coins.keys())
        for i, a in enumerate(syms):
            for b in syms[i+1:]:
                pa, pb = self.get_current_price(a), self.get_current_price(b)
                spread = abs(pa - pb) / min(pa, pb) * 100
                spreads.append((a, b, spread))
        return sorted(spreads, key=lambda x: -x[2])

    def get_edge_cost(self, from_coin: str, to_coin: str,
                      hold_minutes: float, same_chain: bool = True) -> float:
        c = self.coins.get(to_coin)
        if not c:
            return 999.0
        fee = c.swap_fee if same_chain else c.bridge_fee
        return fee + self.get_risk_score(to_coin, hold_minutes)

    def get_omnibus_venues(self, symbol: str) -> List[str]:
        c = self.coins.get(symbol)
        return c.omnibus_venues if c else []

    def summary(self):
        print(f"\n  {'Coin':<7} {'Price':>8} {'Depeg':>8} {'Vol(ann)':>9} "
              f"{'5min Risk':>10} {'Omnibus':>8} {'Chains':>4}")
        print(f"  {'-'*60}")
        for sym in sorted(self.coins.keys()):
            p = self.get_current_price(sym)
            dpg = self.get_depeg(sym) * 100
            v = self.get_volatility(sym)
            r5 = self.get_risk_score(sym, 5) * 100
            nc = len(self.coins[sym].chains)
            print(f"  {sym:<7} ${p:>7.5f} {dpg:>+7.3f}% {v:>8.3%} "
                  f"{r5:>9.5f}% {'✓':>8} {nc:>4}")
        spreads = self.get_depeg_spreads()
        print(f"\n  Top depeg spreads (arbitrage signals):")
        for a, b, s in spreads[:5]:
            pa, pb = self.get_current_price(a), self.get_current_price(b)
            direction = f"buy {a if pa < pb else b}, sell {b if pa < pb else a}"
            print(f"    {a}/{b}: {s:.4f}% — {direction}")
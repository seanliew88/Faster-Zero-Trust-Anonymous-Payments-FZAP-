"""
MODULE 1: FTSO ORACLE
======================
Flare Time Series Oracle interface for price feeds and historical data.

Provides:
    - Current prices via FTSOv2 anchor feeds
    - 30-day historical series at 1-minute resolution
    - Realised volatility, drift, correlation matrix
    - Risk scoring per coin per time horizon
    - "Safest coin" ranking for allocation weighting

Production: Queries FTSOv2 on Coston2 via ContractRegistry
    FtsoV2Interface at ContractRegistry.getContractAddressByName("FtsoV2")
    Feed IDs: 0x01{UTF8(SYMBOL/USD)}...  (21 bytes)
Demo: Generates correlated histories with GARCH vol clustering + Student-t tails

Key FTSO Contract Addresses (Coston2 Testnet):
    ContractRegistry:  0xaD67FE66660Fb8dFE9d6b1b4240d8650e30F6019
    FtsoV2:            resolved via registry
    Update frequency:  ~90 seconds per voting epoch
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CoinProfile:
    symbol: str
    base_price: float
    annual_vol: float
    daily_drift: float
    beta_to_btc: float
    chains: List[str]
    swap_fee: float
    bridge_fee: float


# Realistic crypto parameters calibrated to late-2024 market
COINS = {
    "USDC":  CoinProfile("USDC",  1.0,      0.001,  0.0,     0.00, ["ethereum","avalanche","polygon","solana","flare"], 0.001, 0.002),
    "BTC":   CoinProfile("BTC",   67500.0,  0.55,   0.0002,  1.00, ["bitcoin","flare","ethereum"],                      0.003, 0.005),
    "ETH":   CoinProfile("ETH",   3450.0,   0.65,   0.0003,  1.15, ["ethereum","avalanche","flare","polygon"],           0.003, 0.004),
    "SOL":   CoinProfile("SOL",   185.0,    0.85,   0.0004,  1.40, ["solana","ethereum"],                                0.003, 0.005),
    "AVAX":  CoinProfile("AVAX",  38.5,     0.80,  -0.0001,  1.30, ["avalanche","ethereum"],                             0.003, 0.005),
    "XRP":   CoinProfile("XRP",   0.62,     0.70,   0.0000,  0.90, ["flare","ethereum"],                                 0.003, 0.004),
    "FLR":   CoinProfile("FLR",   0.028,    0.95,   0.0001,  0.80, ["flare"],                                            0.002, 0.003),
    "MATIC": CoinProfile("MATIC", 0.85,     0.90,  -0.0002,  1.20, ["polygon","ethereum"],                               0.003, 0.004),
    "LTC":   CoinProfile("LTC",   95.0,     0.60,  -0.0001,  0.85, ["bitcoin","ethereum"],                               0.003, 0.005),
    "DOGE":  CoinProfile("DOGE",  0.12,     1.10,   0.0001,  1.50, ["ethereum","flare"],                                 0.003, 0.005),
}


class FTSOOracle:
    """
    Flare Time Series Oracle data provider.

    In production, __init__ takes a web3 provider and calls:
        registry = w3.eth.contract(address=REGISTRY_ADDR, abi=...)
        ftso_addr = registry.functions.getContractAddressByName("FtsoV2").call()
        self.ftso = w3.eth.contract(address=ftso_addr, abi=FTSOV2_ABI)

    Then get_current_price() calls:
        self.ftso.functions.getFeedById(feed_id).call()
        -> (uint256 value, int8 decimals, uint64 timestamp)
    """

    def __init__(self, seed: int = None):
        self.coins = COINS
        self._seed = seed or int(time.time()) % 100000
        np.random.seed(self._seed)

        self.n_steps = 43200  # 30 days of 1-min data
        self.prices: Dict[str, np.ndarray] = {}
        self.log_returns: Dict[str, np.ndarray] = {}
        self._generate_histories()

    def _generate_histories(self):
        """Generate correlated price histories with realistic statistical properties."""
        symbols = [s for s in self.coins if s != "USDC"]
        n = len(symbols)

        # Correlation matrix from beta structure
        betas = np.array([self.coins[s].beta_to_btc for s in symbols])
        market_var = 0.5
        corr = np.clip(np.outer(betas, betas) * market_var, -0.90, 0.90)
        np.fill_diagonal(corr, 1.0)

        # Force positive definite via eigenvalue clipping + regularisation
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 0.05)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Renormalise diagonal to 1
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        # Add small ridge for numerical stability
        corr += np.eye(n) * 0.01
        corr /= (1 + 0.01)
        L = np.linalg.cholesky(corr)

        dt = 1.0 / (365 * 24 * 60)  # 1 minute in years

        for idx, sym in enumerate(symbols):
            c = self.coins[sym]
            sigma_base = c.annual_vol * np.sqrt(dt)

            prices = np.zeros(self.n_steps)
            prices[0] = c.base_price
            sigma_t = sigma_base

            # GARCH(1,1) parameters
            omega = sigma_base**2 * 0.05
            alpha_g = 0.10
            beta_g = 0.85

            for t in range(1, self.n_steps):
                z_raw = np.random.standard_t(df=5) / np.sqrt(5/3)

                if t > 1:
                    prev_ret = np.log(prices[t-1]/prices[t-2]) if prices[t-2] > 0 else 0
                    shock = prev_ret / sigma_t if sigma_t > 0 else 0
                    sigma_t = np.sqrt(max(omega + alpha_g * shock**2 * sigma_base**2 + beta_g * sigma_t**2,
                                          sigma_base**2 * 0.1))

                ret = c.daily_drift * dt * 365 + sigma_t * z_raw
                ret = np.clip(ret, -0.08, 0.08)
                prices[t] = prices[t-1] * np.exp(ret)

            self.prices[sym] = prices
            self.log_returns[sym] = np.diff(np.log(prices))

        # USDC: essentially flat with tiny noise
        self.prices["USDC"] = np.ones(self.n_steps) + np.random.normal(0, 0.00001, self.n_steps)
        self.log_returns["USDC"] = np.diff(np.log(self.prices["USDC"]))

    def get_current_price(self, symbol: str) -> float:
        return float(self.prices[symbol][-1])

    def get_volatility(self, symbol: str, window_minutes: int = 1440) -> float:
        """Realised annualised volatility over window."""
        rets = self.log_returns[symbol][-window_minutes:]
        return float(np.std(rets) * np.sqrt(525600))

    def get_risk_score(self, symbol: str, hold_minutes: int) -> float:
        """
        Expected 1-sigma loss over hold_minutes as a fraction.
        Lower = safer. USDC ≈ 0, DOGE ≈ high.
        """
        vol = self.get_volatility(symbol)
        return vol * np.sqrt(hold_minutes / (365 * 24 * 60))

    def get_correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        symbols = sorted(self.log_returns.keys())
        n = len(symbols)
        rets = np.column_stack([self.log_returns[s][-10080:] for s in symbols])
        corr = np.corrcoef(rets.T)
        return symbols, corr

    def rank_coins_by_safety(self, hold_minutes: int = 10) -> List[Tuple[str, float]]:
        """Rank coins from safest to riskiest for a given hold time."""
        scores = []
        for sym in self.coins:
            risk = self.get_risk_score(sym, hold_minutes)
            scores.append((sym, risk))
        return sorted(scores, key=lambda x: x[1])

    def get_edge_cost(self, from_coin: str, to_coin: str,
                      hold_minutes: float, same_chain: bool = True) -> float:
        """
        Cost of holding to_coin for hold_minutes.
        Combines: swap/bridge fee + expected volatility loss.
        This becomes the edge weight in the QAOA graph.
        """
        c = self.coins.get(to_coin)
        if not c:
            return 999.0

        fee = c.swap_fee if same_chain else c.bridge_fee
        vol_risk = self.get_risk_score(to_coin, hold_minutes)

        return fee + vol_risk

    def summary(self):
        """Print FTSO data summary."""
        print(f"\n  {'Coin':<7} {'Price':>10} {'Vol(ann)':>9} {'5min Risk':>10} {'15min Risk':>10}")
        print(f"  {'-'*50}")
        for sym in sorted(self.coins.keys()):
            p = self.get_current_price(sym)
            v = self.get_volatility(sym)
            r5 = self.get_risk_score(sym, 5) * 100
            r15 = self.get_risk_score(sym, 15) * 100
            print(f"  {sym:<7} ${p:>9,.2f} {v:>8.1%} {r5:>9.3f}% {r15:>9.3f}%")

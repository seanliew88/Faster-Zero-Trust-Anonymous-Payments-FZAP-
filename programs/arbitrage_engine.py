"""
MODULE 5: STABLECOIN ARBITRAGE ENGINE
=======================================
Captures depeg spreads between stablecoins across chains.

STABLECOIN ARBITRAGE:
    Unlike volatile coin arb (BTC cheap on one exchange, expensive on another),
    stablecoin arb exploits DEPEG events:

    Normal:  USDC = $1.0000 everywhere
    Depeg:   DAI = $0.9970 on Ethereum, $1.0020 on Polygon  -> 0.5% spread!

    These depegs happen because:
    - Different liquidity depths on different chains
    - PSM (Peg Stability Module) latency
    - Bridge delays creating temporary imbalances
    - Curve pool imbalances after large trades

    Since we're routing through stablecoins anyway, we can
    CHOOSE which stablecoin to route through based on which
    has the best depeg spread on our target chains.

OMNIBUS VENUE REQUIREMENT:
    All arbitrage must go through omnibus-compatible venues:
    ✅ Curve 3pool (USDT/USDC/DAI) — permissionless, omnibus
    ✅ Aave lending pool — omnibus via aTokens
    ✅ Maker PSM — USDC/DAI 1:1 swap, omnibus
    ✅ Plasma native pool — paymaster-sponsored omnibus
    ❌ Binance/Coinbase — segregated accounts, NOT omnibus
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time


@dataclass
class DEXQuote:
    coin: str
    chain: str
    dex: str
    price_usd: float
    liquidity_usd: float
    slippage_bps: float
    omnibus_compatible: bool
    timestamp: float

@dataclass
class ArbitrageOpportunity:
    coin: str
    buy_chain: str
    buy_dex: str
    buy_price: float
    sell_chain: str
    sell_dex: str
    sell_price: float
    spread_pct: float
    net_profit_pct: float
    bridge_fee_pct: float
    estimated_profit_usd: float
    confidence: float
    plasma_intermediate: bool


class ArbitrageScanner:
    """Scans stablecoin prices across omnibus-compatible DEXs."""

    # ONLY omnibus-compatible venues
    OMNIBUS_DEXS = {
        "ethereum": ["curve_3pool", "curve_frax", "aave_v3", "maker_psm"],
        "avalanche": ["curve_avax", "aave_v3"],
        "polygon":   ["curve_polygon", "aave_v3"],
        "solana":    ["solend"],
        "flare":     ["blazeswap", "sparkdex"],
        "plasma":    ["plasma_pool"],
    }

    BRIDGE_FEES = {
        frozenset({"plasma", "ethereum"}): 0.0008,
        frozenset({"plasma", "avalanche"}): 0.0008,
        frozenset({"plasma", "polygon"}): 0.0006,
        frozenset({"plasma", "solana"}): 0.001,
        frozenset({"plasma", "flare"}): 0.0004,
        frozenset({"ethereum", "avalanche"}): 0.0015,
        frozenset({"ethereum", "polygon"}): 0.001,
        frozenset({"ethereum", "solana"}): 0.0015,
        frozenset({"ethereum", "flare"}): 0.001,
        frozenset({"avalanche", "polygon"}): 0.001,
        frozenset({"avalanche", "flare"}): 0.0008,
        frozenset({"polygon", "flare"}): 0.0008,
    }

    def __init__(self, ftso, trade_size_usd=10000):
        self.ftso = ftso
        self.trade_size = trade_size_usd
        self._quote_cache: Dict[str, List[DEXQuote]] = {}

    def scan_all_quotes(self) -> Dict[str, List[DEXQuote]]:
        quotes = {}
        now = time.time()
        for sym, profile in self.ftso.coins.items():
            base = self.ftso.get_current_price(sym)
            coin_quotes = []
            for chain in profile.chains:
                dexs = self.OMNIBUS_DEXS.get(chain, [])
                for dex in dexs:
                    # Stablecoin-specific: deviation from PEG (not from each other)
                    # Each DEX/chain has slightly different depeg
                    dev = np.random.normal(0, profile.peg_tightness * 0.5)
                    if chain in ("flare", "polygon"):
                        dev *= 1.3  # less liquid = wider depeg
                    dex_price = base + dev  # additive around peg

                    # Curve pools have deep liquidity
                    if "curve" in dex:
                        liq = 10_000_000
                    elif "aave" in dex:
                        liq = 5_000_000
                    elif dex == "plasma_pool":
                        liq = 2_000_000
                    else:
                        liq = 1_000_000

                    slippage = (self.trade_size / liq) * 10  # bps

                    coin_quotes.append(DEXQuote(
                        sym, chain, dex, dex_price, liq,
                        slippage, omnibus_compatible=True, timestamp=now))

            quotes[sym] = coin_quotes
        self._quote_cache = quotes
        return quotes

    def find_opportunities(self, min_spread_bps=2.0) -> List[ArbitrageOpportunity]:
        if not self._quote_cache:
            self.scan_all_quotes()
        opportunities = []
        for coin, quotes in self._quote_cache.items():
            for i, qb in enumerate(quotes):
                for j, qs in enumerate(quotes):
                    if i == j: continue
                    if qb.chain == qs.chain and qb.dex == qs.dex: continue
                    if not qb.omnibus_compatible or not qs.omnibus_compatible: continue

                    spread = (qs.price_usd - qb.price_usd) / qb.price_usd
                    if spread <= 0: continue

                    bridge_key = frozenset({qb.chain, qs.chain})
                    bridge_fee = self.BRIDGE_FEES.get(bridge_key, 0.002)
                    if qb.chain == qs.chain:
                        bridge_fee = 0

                    total_slip = (qb.slippage_bps + qs.slippage_bps) / 10000
                    net = spread - bridge_fee - total_slip

                    if net * 10000 < min_spread_bps: continue

                    plasma_route = qb.chain == "plasma" or qs.chain == "plasma"
                    min_liq = min(qb.liquidity_usd, qs.liquidity_usd)
                    confidence = min(1.0, min_liq / (self.trade_size * 5)) * 0.6 + \
                                 min(1.0, spread / 0.005) * 0.4

                    opportunities.append(ArbitrageOpportunity(
                        coin=coin,
                        buy_chain=qb.chain, buy_dex=qb.dex, buy_price=qb.price_usd,
                        sell_chain=qs.chain, sell_dex=qs.dex, sell_price=qs.price_usd,
                        spread_pct=spread * 100, net_profit_pct=net * 100,
                        bridge_fee_pct=bridge_fee * 100,
                        estimated_profit_usd=net * self.trade_size,
                        confidence=confidence,
                        plasma_intermediate=plasma_route))

        opportunities.sort(key=lambda o: o.estimated_profit_usd, reverse=True)
        return opportunities


class ArbitrageAwareRouter:
    """Adjusts QAOA edge costs for stablecoin depeg arbitrage."""

    def __init__(self, ftso, arb_scanner, arb_weight=0.8):
        self.ftso = ftso
        self.scanner = arb_scanner
        self.arb_weight = arb_weight
        self.opportunities: List[ArbitrageOpportunity] = []
        self._arb_map: Dict[str, float] = {}

    def scan_and_build_map(self):
        self.scanner.scan_all_quotes()
        self.opportunities = self.scanner.find_opportunities(min_spread_bps=1.0)
        self._arb_map = {}
        for opp in self.opportunities:
            key = f"{opp.coin}:{opp.buy_chain}->{opp.sell_chain}"
            if key not in self._arb_map or opp.net_profit_pct > self._arb_map[key]:
                self._arb_map[key] = opp.net_profit_pct / 100
        return self._arb_map

    def get_adjusted_edge_cost(self, from_coin, to_coin,
                                from_chain, to_chain, base_cost):
        arb = 0.0
        for key_pattern in [f"{to_coin}:{from_chain}->{to_chain}",
                            f"{from_coin}:{from_chain}->{to_chain}",
                            f"{to_coin}:{to_chain}->{from_chain}"]:
            if key_pattern in self._arb_map:
                arb = max(arb, self._arb_map[key_pattern])
        return base_cost - (arb * self.arb_weight), arb

    def print_opportunities(self, top_n=10):
        if not self.opportunities:
            self.scan_and_build_map()
        print(f"\n  [ARB] {len(self.opportunities)} stablecoin depeg opportunities "
              f"(top {top_n}):")
        print(f"  {'Coin':<7} {'Buy@Venue':<25} {'Sell@Venue':<25} "
              f"{'Spread':>8} {'Net':>8} {'Profit':>10}")
        print(f"  {'-'*90}")
        total = 0
        for opp in self.opportunities[:top_n]:
            bl = f"{opp.buy_dex}@{opp.buy_chain}"
            sl = f"{opp.sell_dex}@{opp.sell_chain}"
            tag = " [P]" if opp.plasma_intermediate else ""
            print(f"  {opp.coin:<7} {bl:<25} {sl:<25} "
                  f"{opp.spread_pct:>7.4f}% {opp.net_profit_pct:>7.4f}% "
                  f"${opp.estimated_profit_usd:>8,.2f}{tag}")
            total += opp.estimated_profit_usd
        print(f"\n  [ARB] Total capturable: ${total:,.2f} "
              f"| Plasma-routed: {sum(1 for o in self.opportunities if o.plasma_intermediate)}")
        return total
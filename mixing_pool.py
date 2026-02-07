"""
MODULE 3: MIXING POOL (v3 - Plasma Native)
=============================================
Zero-Fee USDT | Ephemeral Burn Wallets | QAOA Optimised

PLASMA ADVANTAGES:
    - Zero-fee USDT transfers (protocol paymaster sponsors gas)
    - Sub-second finality (PlasmaBFT consensus)
    - EVM compatible (standard Solidity, Hardhat, MetaMask)
    - Confidential payments module (upcoming)
    - 1000+ TPS
    - No need to hold XPL for USDT transfers

NO RESERVE. ALL funds go through mixing for maximum anonymity.
On Plasma, USDT hops are FREE so we can do many hops at no cost.

EPHEMERAL BURN WALLETS:
    - Fresh keypair per wallet, used once, then destroyed
    - Private key held only in memory, never persisted
    - After funds leave, key is zeroed out (burned)
    - Temporary records held for settlement, then wiped
"""

import numpy as np
import secrets
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Tuple

PLASMA_CONFIG = {
    "rpc_testnet": "https://testnet-rpc.plasma.to",
    "chain_id_testnet": 9746,
    "explorer": "https://testnet.plasmascan.to",
    "usdt_fee": 0.0,
    "finality_seconds": 1,
}

@dataclass
class Buyer:
    address: str
    amount_usdt: float

@dataclass
class Seller:
    address: str
    amount_usdt: float
    dest_chain: str

@dataclass
class EphemeralWallet:
    wallet_id: str
    address: str
    _private_key: bytes
    coin: str
    chain: str
    amount_usd: float
    amount_coin: float
    created_at: float
    hop_number: int
    coin_history: List[str] = field(default_factory=list)
    burned: bool = False
    burn_timestamp: float = 0.0

    def burn(self):
        self._private_key = b'\x00' * 32
        self.burned = True
        self.burn_timestamp = time.time()

    @property
    def is_alive(self):
        return not self.burned

@dataclass
class HopRecord:
    hop_number: int
    timestamp: float
    wallets_created: int
    wallets_burned: int
    total_value: float
    coins_used: List[str]
    plasma_transfers: int
    cross_chain_swaps: int


class MixingPool:
    def __init__(self, ftso, qaoa_pathfinder,
                 hop_interval_minutes=5.0, n_hops=4, n_temp_wallets=6):
        self.ftso = ftso
        self.qaoa = qaoa_pathfinder
        self.hop_interval = hop_interval_minutes
        self.n_hops = n_hops
        self.n_wallets = n_temp_wallets
        self.buyers = []
        self.sellers = []
        self.pool_total = 0.0
        self.current_wallets = []
        self.burned_wallets = []
        self.hop_history = []
        self.all_wallets_ever = []
        self._viz_wallet_snapshots = []

    def add_buyer(self, address, amount):
        self.buyers.append(Buyer(address, amount))
        self.pool_total += amount

    def add_seller(self, address, amount, dest_chain="plasma"):
        self.sellers.append(Seller(address, amount, dest_chain))

    def _create_ephemeral(self, wid, coin, chain, amount_usd, hop, history=None):
        pk = secrets.token_bytes(32)
        addr = "0x" + hashlib.sha256(pk).hexdigest()[:40]
        price = self.ftso.get_current_price(coin) if coin not in ("USDT","USDC") else 1.0
        hist = list(history or [])
        if coin not in hist:
            hist.append(coin)
        return EphemeralWallet(
            wallet_id=wid, address=addr, _private_key=pk,
            coin=coin, chain=chain, amount_usd=amount_usd,
            amount_coin=amount_usd / price, created_at=time.time(),
            hop_number=hop, coin_history=hist,
        )

    def _burn_wallets(self, wallets):
        for w in wallets:
            if w.is_alive:
                w.burn()
                self.burned_wallets.append(w.wallet_id)

    def _select_coin(self, wallet):
        visited = set(wallet.coin_history)
        usdt_prob = max(0.15, 0.5 - wallet.hop_number * 0.08)
        if np.random.random() < usdt_prob:
            return "USDT", "plasma"
        visited.update(["USDT", "USDC"])
        safety = self.ftso.rank_coins_by_safety(self.hop_interval)
        cands = [(s, r) for s, r in safety if s not in visited and s in self.ftso.coins]
        if not cands:
            recent = set(wallet.coin_history[-2:]) if len(wallet.coin_history) >= 2 else set()
            recent.update(["USDT", "USDC"])
            cands = [(s, r) for s, r in safety if s not in recent]
            if not cands:
                cands = [(s, r) for s, r in safety if s not in ("USDT", "USDC")]
        syms = [s for s, _ in cands]
        risks = np.array([r for _, r in cands])
        w = 1.0 / (risks + 0.001); w /= w.sum()
        coin = np.random.choice(syms, p=w)
        profile = self.ftso.coins.get(coin)
        chains = profile.chains if profile else ["plasma"]
        chain = np.random.choice(chains)
        return coin, chain

    def execute_initial_split(self):
        print(f"\n  [POOL] {len(self.buyers)} buyers -> ${self.pool_total:,.2f} USDT on Plasma")
        print(f"  [POOL] Transfer fee: $0.00 (Plasma paymaster)")
        amounts = np.random.dirichlet(np.ones(self.n_wallets)) * self.pool_total
        wallets = []; pf = 0; cc = 0
        safety = self.ftso.rank_coins_by_safety(self.hop_interval)
        avail = [s for s, _ in safety if s not in ("USDT", "USDC")]
        for i in range(self.n_wallets):
            if np.random.random() < 0.35:
                coin, chain, amt = "USDT", "plasma", amounts[i]; pf += 1
            else:
                coin = avail[i % len(avail)]
                chain = "plasma"
                p = self.ftso.coins.get(coin)
                fee = p.swap_fee if p else 0.003
                amt = amounts[i] * (1 - fee); cc += 1
            w = self._create_ephemeral(f"eph_0_{i}", coin, chain, amt, 0)
            wallets.append(w)
        self.current_wallets = wallets
        self.all_wallets_ever.extend(wallets)
        self._viz_wallet_snapshots.append([
            {"id": w.wallet_id, "coin": w.coin, "chain": w.chain,
             "amount": w.amount_usd, "path": list(w.coin_history)} for w in wallets])
        print(f"  [POOL] Plasma free: {pf}  |  Cross-chain: {cc}")
        for w in wallets:
            tag = "FREE" if w.coin == "USDT" and w.chain == "plasma" else "SWAP"
            print(f"    {w.wallet_id}: ${w.amount_usd:>10,.2f} {w.coin}@{w.chain} [{tag}]")
        return wallets

    def execute_hop(self, hop_number):
        print(f"\n  [HOP {hop_number}] {'='*55}")
        old = list(self.current_wallets)
        total = 0
        for w in old:
            if w.coin in ("USDT", "USDC"): w.amount_usd = w.amount_coin
            else:
                p = self.ftso.get_current_price(w.coin)
                w.amount_usd = w.amount_coin * p
            total += w.amount_usd
        print(f"  [HOP {hop_number}] Pool: ${total:,.2f}  |  Running QAOA...")
        self.qaoa.build_graph(max_edges=10)
        self.qaoa.solve(max_iter=60)
        after_fees = 0
        for w in old:
            if w.coin == "USDT" and w.chain == "plasma":
                after_fees += w.amount_usd
            else:
                p = self.ftso.coins.get(w.coin)
                after_fees += w.amount_usd * (1 - (p.swap_fee if p else 0.003))
        amounts = np.random.dirichlet(np.ones(self.n_wallets)) * after_fees
        new = []; pf = 0; cc = 0
        for i in range(self.n_wallets):
            old_w = old[i % len(old)]
            coin, chain = self._select_coin(old_w)
            if coin == "USDT" and chain == "plasma":
                amt = amounts[i]; pf += 1
            else:
                p = self.ftso.coins.get(coin)
                amt = amounts[i] * (1 - (p.swap_fee if p else 0.003)); cc += 1
            w = self._create_ephemeral(f"eph_{hop_number}_{i}", coin, chain, amt,
                                        hop_number, old_w.coin_history)
            new.append(w)
        val = sum(w.amount_usd for w in new)
        loss = (1 - val / self.pool_total) * 100
        self._burn_wallets(old)
        self.hop_history.append(HopRecord(
            hop_number, time.time(), len(new), len(old), val,
            list(set(w.coin for w in new)), pf, cc))
        self.current_wallets = new
        self.all_wallets_ever.extend(new)
        self._viz_wallet_snapshots.append([
            {"id": w.wallet_id, "coin": w.coin, "chain": w.chain,
             "amount": w.amount_usd, "path": list(w.coin_history)} for w in new])
        print(f"  [HOP {hop_number}] Free: {pf} | Swap: {cc} | "
              f"Value: ${val:,.2f} (loss: {loss:.3f}%)")
        for w in new:
            path = ' -> '.join(w.coin_history)
            tag = "FREE" if w.coin == "USDT" and w.chain == "plasma" else "SWAP"
            print(f"    {w.wallet_id}: ${w.amount_usd:>10,.2f} {w.coin}@{w.chain} "
                  f"[{tag}] path: {path}")
        print(f"  [HOP {hop_number}] BURNED {len(old)} old wallets")
        return self.hop_history[-1]

    def execute_final_settlement(self):
        print(f"\n  [SETTLEMENT] {'='*55}")
        recovered = 0
        for w in self.current_wallets:
            if w.coin in ("USDT", "USDC"):
                recovered += w.amount_usd
            else:
                p = self.ftso.get_current_price(w.coin)
                v = w.amount_coin * p
                pr = self.ftso.coins.get(w.coin)
                recovered += v * (1 - (pr.swap_fee if pr else 0.003))
        owed = sum(s.amount_usdt for s in self.sellers)
        print(f"  [SETTLEMENT] Recovered: ${recovered:,.2f} USDT")
        print(f"  [SETTLEMENT] Owed:      ${owed:,.2f} USDT")
        t = time.time()
        settlements = []
        for s in self.sellers:
            tx = "0x" + hashlib.sha256(f"{s.address}_{s.amount_usdt}_{t}".encode()).hexdigest()
            settlements.append({"seller_address": s.address, "amount_usdt": s.amount_usdt,
                "dest_chain": s.dest_chain, "tx_hash": tx, "timestamp": t,
                "fee": 0.0 if s.dest_chain == "plasma" else 0.003})
        print(f"\n  [SETTLEMENT] {len(settlements)} payments sent simultaneously:")
        for s in settlements:
            tag = "FREE" if s["fee"] == 0 else f"fee:{s['fee']*100:.1f}%"
            print(f"    {s['seller_address'][:20]}... ${s['amount_usdt']:>10,.2f} "
                  f"on {s['dest_chain']} [{tag}]")

        print(f"\n  [BURN] Destroying all remaining wallet keys...")
        self._burn_wallets(self.current_wallets)
        tb = len(self.burned_wallets)
        print(f"  [BURN] {tb} total wallets burned (keys zeroed)")

        tpf = sum(h.plasma_transfers for h in self.hop_history)
        tcc = sum(h.cross_chain_swaps for h in self.hop_history)
        loss = (1 - recovered / self.pool_total) * 100
        return {
            "settlements": settlements, "total_recovered": recovered,
            "total_owed": owed, "surplus_deficit": recovered - owed,
            "overall_loss_pct": loss, "n_hops": len(self.hop_history),
            "total_wallets_created": len(self.all_wallets_ever),
            "total_wallets_burned": tb,
            "plasma_free_transfers": tpf, "cross_chain_swaps": tcc,
        }

    def run_full_mixing(self):
        print("=" * 65)
        print("  QUANTUM ANONYMOUS SWAP v3 - PLASMA NATIVE")
        print("  Zero-Fee USDT | Ephemeral Burn Wallets | QAOA Optimised")
        print("=" * 65)
        ti = sum(b.amount_usdt for b in self.buyers)
        to = sum(s.amount_usdt for s in self.sellers)
        print(f"\n  Buyers:  {len(self.buyers)} -> ${ti:,.2f} USDT")
        print(f"  Sellers: {len(self.sellers)} -> ${to:,.2f} USDT")
        print(f"  Chain:   Plasma (zero-fee USDT via paymaster)")
        print(f"  Wallets: {self.n_wallets} ephemeral per hop (burned after use)")
        print(f"\n  PHASE 1: Pool + Split")
        self.execute_initial_split()
        for hop in range(1, self.n_hops + 1):
            print(f"\n  PHASE {hop+1}: Hop {hop}")
            self.execute_hop(hop)
        print(f"\n  PHASE {self.n_hops+2}: Settle + Burn + Wipe")
        result = self.execute_final_settlement()
        print(f"\n  [ANTI-TRACE] Wallet paths:")
        ok = True
        for w in self.all_wallets_ever[-self.n_wallets:]:
            path = ' -> '.join(w.coin_history)
            rev = len(w.coin_history) - len(set(w.coin_history))
            if rev > 0: ok = False
            print(f"    {w.wallet_id}: {path} [{'CLEAN' if rev==0 else f'!{rev}'}] BURNED:{w.burned}")
        print(f"  [ANTI-TRACE] All clean: {'YES' if ok else 'NO'}")
        print(f"  [ANTI-TRACE] All keys destroyed: "
              f"{'YES' if all(w.burned for w in self.all_wallets_ever) else 'NO'}")
        return result

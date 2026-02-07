# Faster-Zero-Trust-Anonymous-Payments-FZAP-

## Abstraction
Faster Zero-Trust Anonymous Payments (FZAP) is a privacy-preserving on-chain settlement protocol for stablecoin payments. It enables buyers to pay merchants using stablecoins while preventing third-party observers from reliably linking who paid, what was purchased and when a specific merchant was settled.

## Introduction 
A typical stablecoin transfer reveals the buyer’s address, the merchant’s address, the exact payment amount and the precise time of settlement. This metadata leakage creates risks for consumer privacy, merchant confidentiality and MEV exploitation.
This protocol brings settlement-layer privacy primitives to blockchains without anonymizing funds, without breaking provenance, and without introducing arbitrary withdrawal paths. Instead, it applies settlement abstraction, timing uncertainty, and liquidity-layer transformation to reduce metadata leakage while preserving correctness and auditability.
The protocol separates payments, settlement, and liquidity management into distinct layers. Instead of routing funds directly from buyers to merchants, all payments flow through a Settlement Router (Name TBC) that acts as an on-chain enforcement and accounting layer.

## System Architecture
<img width="803" height="452" alt="image" src="https://github.com/user-attachments/assets/6aaa7d4b-084d-4a75-aaee-16e5b2f01146" />
### 1. Buyer Payments
Buyers pay using ephemeral wallets where each payment includes a one-time identifier to prevent replay. Funds are then deposited into the Settlement Router, not directly to the merchant. 

### 2. Internal Accounting
The Settlement Router maintains an internal ledger that credits merchant balances using commitment identifiers and aggregates multiple buyer deposits. It is important to note that deposits do not trigger swaps nor do they correspond to one-to-one with settlements.

### 3. Privacy-Preserving Settlement TIming
Each deposit specifies a randomised settlement unlock time within a bounded window (5-10 minutes). The settlement timing is unpredictable to observers and withdawals cannot occur immediately after a payment. Merchants are paid according to protocol-enforced settlement policy, not buyer actions.

### 4. Multi-Hop Conversion
The pooled stablecoins from buyers' payments are then pooled and converted across low-volatility assets. These conversions are batched, decoupled from individual payments and most importantly non-deterministic. This ensures that asset hops add uncertainty without creating traceable routing graphs.

### 5. Merchant Settlement
Merchants withdraw funds only after settlement conditions are met. Withdrawals are paid in the original currency and are drawn from the router's settlement buffer. At no point is a merchant paid directly from a buyer or a per-payment path.

## Privacy Model
Privacy is achieved through structural decoupling of payments, settlement, and liquidity management which prevents on-chain observers from reliably linking buyers to merchants or correlating deposits with withdrawals. Liquidity transformations, including optional multi-hop conversions across low-volatility assets, occur asynchronously at the liquidity layer and are explicitly decoupled from user actions. These operations are batched, non-deterministic, and independent of payment or settlement events, ensuring they do not create traceable routing graphs. As a result, privacy emerges as a by-product of aggregation, timing uncertainty, and liquidity abstraction, rather than from obfuscation or anonymity guarantees.

## Verification & Correctness 
The protocol ensures that each payment identifier can only be used once, that merchant withdrawals cannot exceed credited balances, and that settlement timing rules are strictly enforced by on-chain timestamps. All transfers occur atomically, and no privileged role can bypass accounting or settlement constraints.

The contract is immutable and permissionless, with no owner or administrator keys. This ensures that no single party can censor withdrawals, forge balances, or redirect funds. All users interact with the same publicly verifiable rules, and any deviation from protocol logic is cryptographically impossible.

## Legal Positioning
This protocol is not a mixer and is not designed to anonymize funds. Its purpose is to enable confidential commerce . All withdrawals are tied to explicit merchant commitments and users cannot withdraw to arbitrary addresses or sever provenance from funds. The protocol minimises unnecessary data exposure while preserving verifiable correctness, aligning more closely with established financial privacy norms than with obfuscation-based systems.

## Summary
FZAP demonstrates how privacy improvements can be achieved in on-chain payments without anonymising funds or breaking auditability. By separating payment, settlement, and liquidity into distinct layers, the protocol reduces transaction linkability while remaining legally defensible. Privacy is achieved through aggregation, timing uncertainty, and liquidity abstraction.




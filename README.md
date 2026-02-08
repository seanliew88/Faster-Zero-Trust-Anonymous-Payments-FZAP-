# Faster Zero-Trust Anonymous Payments FZAP

## Abstraction
Faster Zero-Trust Anonymous Payments (FZAP) is a privacy-preserving cross-chain settlement protocol for stablecoin payments. It enables buyers to pay merchants using stablecoins while preventing third-party observers from reliably linking Buyers, Merchants, and specific payments.

## Introduction 
A typical on-chain stablecoin transfer publicly reveals the Buyer’s address, the Merchant’s address, the exact payment amount and the precise time of settlement. This metadata leakage creates risks for Buyer privacy and Merchant confidentiality. If a wallet address was found to be linked to a Buyer, an observer can determine exactly what transactions that they have been making.
This protocol ensures that there is never a direct blockchain transaction between a Buyer and a Merchant. Instead, This privacy is achieved through three complementary layers:

1. One-time-use buyer identities, generated per transaction
2. Aggregation based settlement, where Buyer payments are pooled before settlement
3. Multi-hop stablecoin transformations optimised by the BB84-inspired randomisation techniques

## Protocol Architecture
<img width="534" height="271" alt="image" src="https://github.com/user-attachments/assets/dc54a200-e66b-4a7c-b4c4-2444ef42401f" />

### 1. Initalisation of Protocol
Buyers provide the Merchant address and the payment amount they wish to transfer using the FZAP protocol. The protocol generates a set of ephemeral wallet identifiers used to derive temporary input and output wallets. Each wallet identifier is generated using a cryptographically secure source of entropy and is guaranteed to be unique within the lifetime of the protocol.

### 2. Merchant Payment
Upon Buyer payment into the input wallet, the protocol detects the payment and forwards the funds into a shared settlement pool governed by the smart contracts. Individual Buyer deposits are not settled immediately and not routed directly to Merchants. Instead, all buyer deposits within a settlement window are first aggregated into a single pool total. Once aggregated, individual payment amounts then lose their one-to-one correspondence with specific wallets and the protocol only tracks the total pooled value required to satisfy downstream settlements. This pooling and aggregation layer could be conceptually understood as an omnibus settlement account. It is important to note that the protocol does not immediately propogate funds through subsequent settlement stages as this deliberate delay further weakens timing-based correlation attacks. 

### 3. Anonymisation of funds
<img width="563" height="227" alt="image" src="https://github.com/user-attachments/assets/2422590a-6ad0-4839-9cf8-5754a7b32de1" />

<img width="621" height="271" alt="image" src="https://github.com/user-attachments/assets/0b307e33-4d7d-4373-b5ba-4c0ec5a73222" />


The aggregated funds are then converted into amounts of different stablecoins with low volatility without excessive value loss. The protocol employs an arbitrage-aware routing layer optimised using the Quantum Approximate Optimisation Algorithm (QAOA). At each hop, possible stablecoin and chain transitions are modeled as edges, where nodes represent currencies and the weights represent expected swap fees and bridge fees. QAOA is used to then efficiently search this cost landscape and identify routing configurations that minimise aggregate loss across all ephemeral wallets, allowing privacy-enhancing hops to scale without introducing deterministic patterns.

Real-time market data for this optimisation is sourced from the Flare Time Series Oracle (FTSO) API. These oracle feeds capture temporary stablecoin depegs and cross-chain pricing inefficiencies. By treating favorable pricing discrepancies as negative cost offsets—rather than profit targets—the protocol preserves pooled value while maintaining correctness and auditability.


### 4. FDC Settlement
To ensure that Merchants receive the correct settlement amount for each transaction, the protocol integrates Flare Data Connector(FDC)-based settlement verification. For each settlement, an attestation request is generated referencing the transaction hash, destination chain, and payment type. This request is processed during an FDC voting round, where decentralised data providers verify the transsaction against the destination chain. Once the round finalises, the verified result is committed to a Merkle tree and a Merkle proof is made available via Flare's Data Availability layer. This proof allows any party to verify, on-chain or off-chain, that a specific Merchant payment was included in the finalised FDC attestation set.

Merchants then submit the Merkle proof to the settlement verification contract, which validates inclusion against the FDC root published on Flare. A settlement is considered complete only once this proof is verified, ensuring that the Merchant received the exact amount owed on the correct chain. Because verification is based solely on transaction hashes and attested data, no linkage is created between Buyers, intermediate hops, or the final Merchant settlement.

## Legal Positioning
Unlike traditional privacy mixers, the protocol does not provide arbitrary anonymisation or user-controlled withdrawal paths. Funds are never withdrawn freely and there is no mechanism for users to extract value independently of a legitimate commercial settlement. All value flows are explicitly tied to merchant payments, and withdrawals are restricted to predefined settlement addresses.

## Summary
FZAP demonstrates how privacy improvements can be achieved for on-chain stablecoin payments by eliminating direct blockchain linkage between Buyers and Merchants. The protocol introduces a settlement abstraction that decouples Buyer deposits from Merchant payouts and obscures the timing and structure of settlement execution. Privacy is achieved through a combination of one-time-use identifiers, aggregation-based settlement and mult-hop stablecoin transformations. As a result, transaction graph analysis can no loneger rely on direct links and instead requires costly, heuristic-based inference. 

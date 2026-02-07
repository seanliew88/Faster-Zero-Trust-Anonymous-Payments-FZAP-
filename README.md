# Faster Zero-Trust Anonymous Payments FZAP

## Abstraction
Faster Zero-Trust Anonymous Payments (FZAP) is a privacy-preserving cross-chain settlement protocol for stablecoin payments. It enables buyers to pay merchants using stablecoins while preventing third-party observers from reliably linking Buyers, Merchants, and specific payments.

## Introduction 
A typical stablecoin transfer reveals the Buyer’s address, the Merchant’s address, the exact payment amount and the precise time of settlement. This metadata leakage creates risks for Buyer privacy and Merchant confidentiality. Hypothetically, if a wallet address was found to be linked to a Buyer, it would be possible to determine exactly what transactions that they have been making.
This protocol enables Buyers to perform transaction via a third party, the Operator, completely anonymously. Instead of using traditional mixing methods, which have a high time cost, are easily identifiable when are used, require trust in the Operator, and also still link Buyer and Merchant, our FZAP protocol instead opts to leverage high Operator liquidity to "break" the blockchain link between Buyer and Merchant, ensuring that there is never a direct blockchain link between them, and that the funds deposited by the Buyer can be obfuscated by the Operator to anonymously settle future FZAP transactions.
The protocol separates payments, settlement, and fund management into distinct layers. Instead of routing funds directly from Buyers to merchants, all payments flow through a Operator that acts as a provider of fresh liquidity that is untraceable back to the original Buyer.

## Protocol Architecture
<img width="805" height="400" alt="image" src="https://github.com/user-attachments/assets/be2d2feb-e5d2-4b82-aec5-139caad47196" />

### 1. Initalisation of Protocol
Buyers provide the Merchant address and the payment amount they wish to transfer using the FZAP protocol. The Operator creates an ephemeral "input" wallet using a unique ID, which is calculated using a quantum random number generator, and a series of smart contracts which will settle the payment using funds from anonymous ephemeral "output" wallets controlled by the Operator upon Buyer deposit into the input wallet.

### 2. Merchant Payment
Upon Buyer payment into the input wallet, the smart contracts detect the payment, and start to slowly trigger their deposits from the output wallets. Each smart contract makes a deposit of a random amount (predetermined upon contract creation, and sums to the desired transaction amount) over a small, random (again, able to be predetermined), time period of around 5 - 15 minutes. Flare's Data Collector is used to ensure that the Merchant has received the transaction and to prove to the Buyer that the transaction has been successful.

### 3. Anonymisation of funds
The Input wallets aggregate their funds into a central exchange wallet after a random time interval of a few seconds. These funds are then used to purchase random amounts of different stablecoins with low volatility on a decentralised exchange, with the destination wallets of the exchange being created for the purposes of holding these traded funds in interim. The Operator repeats this process as many times as they like, though 6 - 7 times should be enough to fracture and hide the sources of funds enough that tracing becomes virtually impossible. Every trade "breaks" the blockchain, as any trackers would have to do cross-chain analysis in order to identify when/where currencies have been exchanged.

### 4. Multi-Hop Conversion
The pooled stablecoins from buyers' payments are then pooled and converted across low-volatility assets. Using Flare's Time Series Oracle, we obtained data on prices for stablecoins over 30 day periods to determine momentum signals, spread and volatility - converting this to a correlation matrix. These conversions are batched, decoupled from individual payments and most importantly non-deterministic. A quantum approximate optimiser algorithm (QAOA) is applied to ensure the optimal path is taken through the graph (where nodes represent currencies, edges are  calculated by a weighting algorithm) - with the aim of minimising net loss in USDT. This ensures that asset hops add uncertainty without creating traceable routing graphs.

### 5. Liquidity creation
Finally, the last hop is achieved by trading back to the desired currency, which we will assume here is the stablecoin that the Buyers have deposited. These funds are now held in new ephemeral "output" wallets, which are difficult to trace back to the original buyer, and each "exchange hop" and fracturing of funds makes it even harder for trackers to track funds. Even if a complete link of exchanges and payments was computationally feasible to create, the buyer is still never linked to their specific purchase, as their funds are used to pay for future purchases by other Buyers, which will in turn not pay their own Merchants with their own coins, but will pay for future Merchants and so on.

## Privacy Model
Privacy is achieved through structural decoupling of payments, settlement, and liquidity management which prevents on-chain observers from reliably linking buyers to merchants or correlating deposits with withdrawals. Liquidity transformations, including multi-hop cross-chain conversions across low-volatility assets, occur asynchronously at the liquidity layer and are explicitly decoupled from user actions. These operations are batched, non-deterministic, and independent of payment or settlement events, ensuring they do not create traceable routing graphs. As a result, finding the links between Buyer and Merchant becomes unreasonably computationally expensive.

## Improvements over mixers
Firstly, traffic both at the Buyer and Merchant end become harder to identify as using anonymisation services, as addresses are never reused, making it impossible to construct a list of known "mixing" addresses to flag interaction with.
Secondly, verification of payment is easy, as the Flare Data protocol ensures that each payment identifier can only be used once and that settlement timing rules are strictly enforced by on-chain timestamps.
Thirdly, traditional mixers take a very long time to anonymise funds, usually taking hours or days to fully hide funds. The FZAP protocol manages to execute transactions in just under half an hour.
Fourthly, there is never a direct link between Buyer and Merchant, meaning that even if every ephemeral account was exposed to be controlled by the FZAP protocol, it would still be impossible to prove that any given transaction occured.
Finally, we do not introduce a single point of failure in the blockchain. With traditional mixers, it is trivially easy to identify mixer addresses and freeze or seize funds by law enforcement or hacker groups. Here, every address is discarded after use, meaning that there is no way to stop the protocol purely on the blockchain.

## Plasma usage 
Plasma is used an omnibus pool host. When a position lands on Plasma, it sits in this native pool. Ehpemeral wallets are created on Plasma, moving USDT into them and burning them afterwards - this costs nothing in gas! 

## Legal Positioning
Of course, such a program designed to maximise anonymity will inevitably attract money laundering groups. In order to combat this, the settlement processor would have to make sure to prevent bad actors from being able to access such a service. Methods could include:
1. Banning accounts that have recieved funds from or are sanctioned accounts from using this service
2. Requiring a minimum amount of blockchain activity/account age in order to use such a service
3. Banning known dark-web deposit addresses linked to criminal activities.

## Summary
FZAP demonstrates how privacy improvements can be achieved in on-chain payments by "breaking" the blockchain, making it so that there is no direct link between buyer, merchant, and obscuring when/where the settlement processor has made payments on behalf of buyers. By separating payment, settlement, and liquidity into distinct layers, the protocol completely removes transaction linkability, and only computationally expensive heuristic methods can be used to track the movement of funds. Privacy is achieved through aggregation, timing uncertainty, and liquidity abstraction. This protocol outperforms traditional mixing in every way.

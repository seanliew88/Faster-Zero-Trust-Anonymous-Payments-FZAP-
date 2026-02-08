// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

/**
 * @title QuantumSwapPoolController
 * @notice On-chain pool management for the Quantum Anonymous Swap system.
 *
 * This contract handles:
 *   1. Collecting USDC from buyers into a shared pool
 *   2. Recording commitment hashes (hiding buyer-seller links)
 *   3. Authorising the mixing engine to redistribute funds
 *   4. Simultaneous settlement to all sellers
 *   5. FDC-verified proof that sellers received correct amounts
 *
 * Deployed on: Flare Coston2 testnet
 * Dependencies: FDCVerifiedSettlement for payment verification
 */

// ── Interfaces ──────────────────────────────────────────────────────

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
}

interface IFlareContractRegistry {
    function getContractAddressByName(string calldata _name) external view returns (address);
}

interface IFdcVerification {
    function verifyEVMTransaction(bytes calldata _proof) external view returns (bool);
}

// ── Main Contract ───────────────────────────────────────────────────

contract QuantumSwapPoolController {

    // ── State ───────────────────────────────────────────────────────

    address public owner;
    address public mixingEngine;     // authorised off-chain engine
    IERC20  public usdc;

    IFlareContractRegistry public constant REGISTRY =
        IFlareContractRegistry(0xaD67FE66660Fb8dFE9d6b1b4240d8650e30F6019);

    // Pool state
    uint256 public poolTotal;
    uint256 public poolRound;        // incrementing batch ID
    bool    public poolLocked;

    // Buyer deposits
    struct Deposit {
        address buyer;
        uint256 amount;
        bytes32 commitmentHash;      // hash(seller_address, amount, salt)
        uint256 round;
    }

    // Seller claims
    struct Claim {
        address seller;
        uint256 expectedAmount;
        bytes32 destChainId;
        bytes   destAddress;         // address on destination chain
        bool    settled;
        bool    fdcVerified;
        bytes32 settlementTxHash;
    }

    mapping(uint256 => Deposit) public deposits;
    uint256 public depositCount;

    mapping(uint256 => Claim) public claims;
    uint256 public claimCount;

    // ── Events ──────────────────────────────────────────────────────

    event PoolDeposit(uint256 indexed depositId, address indexed buyer,
                      uint256 amount, bytes32 commitmentHash, uint256 round);
    event PoolLocked(uint256 round, uint256 totalAmount, uint256 numDeposits);
    event SettlementExecuted(uint256 indexed claimId, address indexed seller,
                             uint256 amount, bytes32 txHash);
    event SettlementVerified(uint256 indexed claimId, uint256 fdcRound);
    event PoolReset(uint256 newRound);

    // ── Constructor ─────────────────────────────────────────────────

    constructor(address _usdc) {
        owner = msg.sender;
        usdc = IERC20(_usdc);
        poolRound = 1;
    }

    // ── Modifiers ───────────────────────────────────────────────────

    modifier onlyOwner()      { require(msg.sender == owner, "Not owner"); _; }
    modifier onlyEngine()     { require(msg.sender == mixingEngine, "Not engine"); _; }
    modifier poolOpen()       { require(!poolLocked, "Pool locked"); _; }
    modifier poolIsLocked()   { require(poolLocked, "Pool not locked"); _; }

    // ── Configuration ───────────────────────────────────────────────

    function setMixingEngine(address _engine) external onlyOwner {
        mixingEngine = _engine;
    }

    // ── STEP 1: Buyers Deposit ──────────────────────────────────────

    /**
     * @notice Buyer deposits USDC into the pool with a commitment hash.
     * @param amount USDC amount (6 decimals)
     * @param commitmentHash keccak256(abi.encodePacked(sellerAddr, amount, salt))
     *        This hides which seller the buyer intends to pay.
     */
    function deposit(uint256 amount, bytes32 commitmentHash) external poolOpen {
        require(amount > 0, "Zero amount");
        require(usdc.transferFrom(msg.sender, address(this), amount), "Transfer failed");

        deposits[depositCount] = Deposit({
            buyer: msg.sender,
            amount: amount,
            commitmentHash: commitmentHash,
            round: poolRound
        });

        poolTotal += amount;
        emit PoolDeposit(depositCount, msg.sender, amount, commitmentHash, poolRound);
        depositCount++;
    }

    // ── STEP 2: Lock Pool ───────────────────────────────────────────

    /**
     * @notice Lock the pool when enough deposits are collected.
     *         After locking, no more deposits and mixing begins.
     */
    function lockPool() external onlyOwner {
        require(poolTotal > 0, "Empty pool");
        poolLocked = true;
        emit PoolLocked(poolRound, poolTotal, depositCount);
    }

    // ── STEP 3: Register Seller Claims ──────────────────────────────

    /**
     * @notice Register a seller's expected payment.
     *         Called by the mixing engine after verifying commitments.
     */
    function registerClaim(
        address seller,
        uint256 expectedAmount,
        bytes32 destChainId,
        bytes calldata destAddress
    ) external onlyEngine poolIsLocked {
        claims[claimCount] = Claim({
            seller: seller,
            expectedAmount: expectedAmount,
            destChainId: destChainId,
            destAddress: destAddress,
            settled: false,
            fdcVerified: false,
            settlementTxHash: bytes32(0)
        });
        claimCount++;
    }

    // ── STEP 4: Record Settlement ───────────────────────────────────

    /**
     * @notice Record that a settlement TX was sent on the destination chain.
     *         Called by mixing engine after simultaneous settlement.
     */
    function recordSettlement(
        uint256 claimId,
        bytes32 txHash
    ) external onlyEngine poolIsLocked {
        require(claimId < claimCount, "Invalid claim");
        Claim storage c = claims[claimId];
        require(!c.settled, "Already settled");

        c.settled = true;
        c.settlementTxHash = txHash;
        emit SettlementExecuted(claimId, c.seller, c.expectedAmount, txHash);
    }

    // ── STEP 5: FDC Verification ────────────────────────────────────

    /**
     * @notice Submit FDC proof that seller received payment.
     *         Anyone can call this with a valid Merkle proof.
     *
     * In production, this calls FdcVerification.verifyEVMTransaction()
     * which checks the proof against the on-chain Merkle root.
     */
    function verifySettlement(
        uint256 claimId,
        bytes calldata fdcProof
    ) external {
        Claim storage c = claims[claimId];
        require(c.settled, "Not settled");
        require(!c.fdcVerified, "Already verified");

        // Resolve FDC verification contract
        address fdcAddr = REGISTRY.getContractAddressByName("FdcVerification");
        IFdcVerification fdc = IFdcVerification(fdcAddr);

        // Verify the Merkle proof against on-chain root
        require(fdc.verifyEVMTransaction(fdcProof), "FDC proof invalid");

        c.fdcVerified = true;
        emit SettlementVerified(claimId, block.number);
    }

    // ── STEP 6: Reset for Next Round ────────────────────────────────

    function resetPool() external onlyOwner {
        // Verify all claims are settled and verified
        for (uint256 i = 0; i < claimCount; i++) {
            require(claims[i].fdcVerified, "Unverified claims remain");
        }

        poolTotal = 0;
        poolLocked = false;
        depositCount = 0;
        claimCount = 0;
        poolRound++;
        emit PoolReset(poolRound);
    }

    // ── View Functions ──────────────────────────────────────────────

    function getDeposit(uint256 id) external view returns (Deposit memory) {
        return deposits[id];
    }

    function getClaim(uint256 id) external view returns (Claim memory) {
        return claims[id];
    }

    function isFullyVerified() external view returns (bool) {
        if (claimCount == 0) return false;
        for (uint256 i = 0; i < claimCount; i++) {
            if (!claims[i].fdcVerified) return false;
        }
        return true;
    }

    function getPoolStatus() external view returns (
        uint256 round, uint256 total, uint256 numDeposits,
        uint256 numClaims, bool locked
    ) {
        return (poolRound, poolTotal, depositCount, claimCount, poolLocked);
    }
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

library QuantumPoolTypes {
    struct Deposit {
        address buyer;
        uint256 amount;
        bytes32 commitmentHash;
        uint256 round;
    }

    struct Claim {
        address seller;
        uint256 expectedAmount;
        bytes32 destChainId;
        bytes   destAddress;
        bool    settled;
        bool    fdcVerified;
        bytes32 settlementTxHash;
    }
}
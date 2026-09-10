"""
sepolia_anchor.py — Ethereum Sepolia testnet anchor for the audit chain
VERIDEX | blockchain | Anubhav

Periodically reads the latest hash_chain root from chain.py and submits
it as a zero-value transaction to the Ethereum Sepolia testnet via Infura.
The transaction data field carries:

    b"VERIDEX:CHK-<checkpoint_id>:seq_<from>-<to>:<chain_root_hash>"

This gives the audit log a publicly verifiable, immutable timestamp on a
real blockchain without needing a custom smart contract.

Design decisions
----------------
- No smart contract: the chain root is stored in tx.input (calldata).
  Any Etherscan query can retrieve and verify it.
- Offline-first: if Infura is unreachable the anchor is retried on the
  next call. Unanchored records accumulate and are all covered by the
  next successful anchor.
- Gas price: uses eth_maxPriorityFeePerGas + basefee (EIP-1559) to avoid
  overpaying on testnet.

Environment variables required (from .env):
    INFURA_URL       https://sepolia.infura.io/v3/YOUR_PROJECT_ID
    WALLET_ADDRESS   0xYOUR_WALLET_ADDRESS
    PRIVATE_KEY      YOUR_PRIVATE_KEY_NEVER_COMMIT_THIS

Public API
----------
    anchor_now(db)          -> AnchorResult   (submit an anchor transaction)
    check_pending(db)       -> list[AnchorResult] (confirm pending txns)
    get_anchor_history(db)  -> list[BlockchainAnchor]
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.config.settings import get_settings
from backend.db.models import BlockchainAnchor
from backend.db.session import SessionLocal, init_db
from backend.audit.chain import get_chain_root, get_records, mark_anchored

logger = logging.getLogger(__name__)
_cfg   = get_settings()

# Minimum unanchored records before triggering an anchor
ANCHOR_THRESHOLD = 10


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class AnchorResult:
    success:        bool
    tx_hash:        str | None = None
    block_number:   int | None = None
    chain_root:     str | None = None
    seq_from:       int | None = None
    seq_to:         int | None = None
    gas_used:       int | None = None
    error:          str | None = None


# ── Public API ────────────────────────────────────────────────────────────────

def anchor_now(db: Session | None = None) -> AnchorResult:
    """
    Anchor the current chain root to the Sepolia testnet.

    Steps:
    1. Query unanchored AuditRecords from the local DB.
    2. Get the chain root (latest hash_chain value).
    3. Build calldata payload.
    4. Send an EIP-1559 transaction via Infura.
    5. Wait for 1 confirmation.
    6. Mark records as anchored and persist a BlockchainAnchor row.

    Returns an AnchorResult. On failure, returns AnchorResult(success=False)
    without raising — the caller decides whether to retry.
    """
    _own_session = db is None
    if _own_session:
        init_db()
        db = SessionLocal()

    try:
        # ── 1. Check there is something to anchor ─────────────────────────────
        unanchored = get_records(db, limit=1000, unanchored_only=True)
        if not unanchored:
            logger.info("No unanchored records — nothing to anchor.")
            return AnchorResult(success=True, error="No unanchored records.")

        seq_from    = unanchored[-1].seq   # oldest (records are newest-first)
        seq_to      = unanchored[0].seq    # newest
        chain_root  = get_chain_root(db)

        # ── 2. Validate config ────────────────────────────────────────────────
        missing = _check_config()
        if missing:
            err = f"Blockchain config missing: {missing}"
            logger.error(err)
            return AnchorResult(success=False, error=err)

        # ── 3. Connect to Sepolia ─────────────────────────────────────────────
        w3 = _get_web3()
        if not w3.is_connected():
            err = "Could not connect to Sepolia via Infura."
            logger.error(err)
            return AnchorResult(success=False, error=err)

        # ── 4. Build and send transaction ─────────────────────────────────────
        calldata = _build_calldata(chain_root, seq_from, seq_to)
        result   = _send_transaction(w3, calldata)

        if not result.success:
            return result

        # ── 5. Mark records as anchored ───────────────────────────────────────
        mark_anchored(db, seq_to, result.tx_hash, result.block_number)

        # ── 6. Persist anchor log ─────────────────────────────────────────────
        anchor_row = BlockchainAnchor(
            chain_root_hash = chain_root,
            seq_from        = seq_from,
            seq_to          = seq_to,
            tx_hash         = result.tx_hash,
            block_number    = result.block_number,
            gas_used        = result.gas_used,
            status          = "CONFIRMED",
            confirmed_at    = datetime.datetime.utcnow(),
        )
        db.add(anchor_row)
        db.commit()

        logger.info(
            "Anchored seq %d-%d to Sepolia. tx=%s block=%s",
            seq_from, seq_to, result.tx_hash, result.block_number,
        )
        return AnchorResult(
            success      = True,
            tx_hash      = result.tx_hash,
            block_number = result.block_number,
            chain_root   = chain_root,
            seq_from     = seq_from,
            seq_to       = seq_to,
            gas_used     = result.gas_used,
        )

    except Exception as exc:
        db.rollback()
        logger.exception("Unexpected error during anchor: %s", exc)
        return AnchorResult(success=False, error=str(exc))
    finally:
        if _own_session:
            db.close()


def check_pending(db: Session | None = None) -> list[AnchorResult]:
    """
    Check the confirmation status of all PENDING anchor transactions.
    Updates their status in the DB.
    Returns a list of AnchorResult for each checked transaction.
    """
    _own_session = db is None
    if _own_session:
        db = SessionLocal()

    results = []
    try:
        pending = (
            db.query(BlockchainAnchor)
            .filter(BlockchainAnchor.status == "PENDING")
            .all()
        )
        if not pending:
            return []

        w3 = _get_web3()
        for row in pending:
            try:
                receipt = w3.eth.get_transaction_receipt(row.tx_hash)
                if receipt is None:
                    results.append(AnchorResult(
                        success=False, tx_hash=row.tx_hash,
                        error="Still pending (not mined yet).",
                    ))
                    continue

                if receipt.status == 1:
                    row.status       = "CONFIRMED"
                    row.block_number = receipt.blockNumber
                    row.gas_used     = receipt.gasUsed
                    row.confirmed_at = datetime.datetime.utcnow()
                    mark_anchored(db, row.seq_to, row.tx_hash, receipt.blockNumber)
                    results.append(AnchorResult(
                        success      = True,
                        tx_hash      = row.tx_hash,
                        block_number = receipt.blockNumber,
                        gas_used     = receipt.gasUsed,
                    ))
                else:
                    row.status = "FAILED"
                    results.append(AnchorResult(
                        success=False, tx_hash=row.tx_hash,
                        error="Transaction reverted on-chain.",
                    ))
            except Exception as exc:
                logger.warning("Could not check tx %s: %s", row.tx_hash, exc)
                results.append(AnchorResult(
                    success=False, tx_hash=row.tx_hash, error=str(exc),
                ))

        db.commit()
        return results
    finally:
        if _own_session:
            db.close()


def get_anchor_history(
    db: Session,
    limit: int = 20,
) -> list[BlockchainAnchor]:
    """Return the most recent blockchain anchor entries (newest first)."""
    return (
        db.query(BlockchainAnchor)
        .order_by(BlockchainAnchor.submitted_at.desc())
        .limit(limit)
        .all()
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_config() -> list[str]:
    """Return a list of missing required config keys."""
    missing = []
    if not _cfg.infura_url:
        missing.append("INFURA_URL")
    if not _cfg.wallet_address:
        missing.append("WALLET_ADDRESS")
    if not _cfg.private_key:
        missing.append("PRIVATE_KEY")
    return missing


def _get_web3():
    """Create and return a Web3 instance connected to Sepolia via Infura."""
    from web3 import Web3
    return Web3(Web3.HTTPProvider(_cfg.infura_url))


def _build_calldata(chain_root: str, seq_from: int, seq_to: int) -> bytes:
    """
    Build the transaction calldata string that is stored on-chain.

    Format:
        VERIDEX:CHK-<checkpoint_id>:seq_<from>-<to>:<chain_root_hash>

    This is human-readable on Etherscan and verifiable by anyone with the
    local audit DB.
    """
    tag = (
        f"VERIDEX:"
        f"CHK-{_cfg.checkpoint_id}:"
        f"seq_{seq_from}-{seq_to}:"
        f"{chain_root}"
    )
    return tag.encode("utf-8")


def _send_transaction(w3, calldata: bytes) -> AnchorResult:
    """
    Build, sign, and broadcast an EIP-1559 transaction containing *calldata*.
    Waits for 1 block confirmation before returning.
    Sends 0 ETH — the calldata is the only meaningful content.
    """
    from web3 import Web3

    try:
        account  = w3.eth.account.from_key(_cfg.private_key)
        nonce    = w3.eth.get_transaction_count(account.address)

        # EIP-1559 fee estimation
        base_fee    = w3.eth.get_block("latest")["baseFeePerGas"]
        priority    = w3.eth.max_priority_fee
        max_fee     = base_fee * 2 + priority   # generous cap for testnet

        tx = {
            "chainId":             11155111,   # Sepolia chain ID
            "nonce":               nonce,
            "to":                  account.address,  # self-send (no contract)
            "value":               0,
            "data":                calldata,
            "gas":                 60_000,
            "maxFeePerGas":        max_fee,
            "maxPriorityFeePerGas": priority,
            "type":                "0x2",
        }

        signed  = w3.eth.account.sign_transaction(tx, _cfg.private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        logger.info("Tx submitted to Sepolia: %s — waiting for receipt...", tx_hash.hex())

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt.status != 1:
            return AnchorResult(
                success=False,
                tx_hash=tx_hash.hex(),
                error="Transaction reverted on-chain.",
            )

        return AnchorResult(
            success      = True,
            tx_hash      = tx_hash.hex(),
            block_number = receipt.blockNumber,
            gas_used     = receipt.gasUsed,
        )

    except Exception as exc:
        logger.error("Transaction failed: %s", exc)
        return AnchorResult(success=False, error=str(exc))

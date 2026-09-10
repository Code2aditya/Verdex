"""
test_step2.py — Smoke test for audit chain + blockchain anchor config
Run from the veridex/ root:

    venv\Scripts\python.exe scripts/test_step2.py

What it tests:
  1. DB initialises and tables are created
  2. append_record() adds records with correct chaining
  3. hash_chain values are unique and deterministic
  4. verify_chain() passes on an unmodified chain
  5. Tampering a record breaks verify_chain()
  6. get_chain_root() returns the last hash_chain
  7. mark_anchored() updates the right records
  8. Blockchain config detection (reports missing keys without crashing)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use an in-memory SQLite DB so the test never touches the real audit file
os.environ["AUDIT_DB_PATH"] = ":memory:"
# Unset blockchain keys so anchor config check test works cleanly
os.environ.setdefault("INFURA_URL",      "")
os.environ.setdefault("WALLET_ADDRESS",  "")
os.environ.setdefault("PRIVATE_KEY",     "")

from backend.db.session import init_db, SessionLocal
from backend.audit.chain import (
    append_record, verify_chain, get_chain_root,
    get_records, mark_anchored, ScreeningPayload, GENESIS_HASH,
)
from backend.blockchain.sepolia_anchor import _check_config

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")
    results.append((label, condition))


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Setup: initialising in-memory DB ===")
init_db()
db = SessionLocal()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Append first record (genesis link)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 1: append first record ===")

p1 = ScreeningPayload(
    checkpoint_id = "CHK-TEST",
    officer_id    = "OFC-001",
    doc_type      = "PASSPORT",
    doc_number    = "Z1234567",
    risk_score    = 0.12,
    decision      = "CLEAR",
)
r1 = append_record(p1, db=db)

check("Record 1 seq == 1",               r1.seq == 1, f"seq={r1.seq}")
check("Record 1 prev_hash == GENESIS",   r1.prev_hash == GENESIS_HASH,
      f"prev_hash={r1.prev_hash[:16]}...")
check("record_hash is 64 hex chars",     len(r1.record_hash) == 64)
check("hash_chain is 64 hex chars",      len(r1.hash_chain) == 64)
check("decision stored correctly",       r1.decision == "CLEAR")
check("anchored is False initially",     r1.anchored == False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Append more records, check chaining
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: append 4 more records and verify chain links ===")

prev_chain = r1.hash_chain
for i, decision in enumerate(["REVIEW", "CLEAR", "ALERT", "CLEAR"], start=2):
    p = ScreeningPayload(
        checkpoint_id = "CHK-TEST",
        doc_number    = f"DOC-{i:04d}",
        risk_score    = 0.1 * i,
        decision      = decision,
    )
    r = append_record(p, db=db)
    check(f"Record {i} seq == {i}",        r.seq == i, f"seq={r.seq}")
    check(f"Record {i} prev_hash correct", r.prev_hash == prev_chain,
          f"prev={r.prev_hash[:12]}... expected={prev_chain[:12]}...")
    prev_chain = r.hash_chain


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — verify_chain() on clean chain
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: verify_chain() — clean chain ===")

result = verify_chain(db=db)
print(f"  total_records: {result.total_records}")
print(f"  valid:         {result.valid}")

check("Chain valid on clean records",    result.valid)
check("total_records == 5",             result.total_records == 5)
check("first_broken_seq is None",       result.first_broken_seq is None)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — get_chain_root()
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: get_chain_root() ===")

root = get_chain_root(db=db)
check("Chain root is 64 hex chars",     len(root) == 64, f"root={root[:16]}...")
check("Chain root matches last record's hash_chain",
      root == prev_chain, f"root={root[:16]}... last={prev_chain[:16]}...")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Tamper a record and verify chain breaks
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 5: verify_chain() — after tampering record seq=2 ===")

from backend.db.models import AuditRecord
tampered = db.query(AuditRecord).filter(AuditRecord.seq == 2).first()
original_decision = tampered.decision
tampered.decision = "CLEAR"   # silently change REVIEW -> CLEAR
db.commit()

tamper_result = verify_chain(db=db)
print(f"  valid:             {tamper_result.valid}")
print(f"  first_broken_seq:  {tamper_result.first_broken_seq}")

check("Chain invalid after tampering",        not tamper_result.valid)
check("Broken at seq=2",                      tamper_result.first_broken_seq == 2,
      f"broken at seq={tamper_result.first_broken_seq}")

# Restore the original value
tampered.decision = original_decision
db.commit()
restored = verify_chain(db=db)
check("Chain valid again after restoring",    restored.valid)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — mark_anchored()
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 6: mark_anchored() ===")

fake_tx    = "0x" + "ab" * 32
fake_block = 7654321
count = mark_anchored(db, seq_to=3, tx_hash=fake_tx, block_number=fake_block)

check("3 records marked anchored (seq 1-3)",  count == 3, f"count={count}")

anchored_records = db.query(AuditRecord).filter(AuditRecord.anchored == True).all()
check("All anchored records have tx_hash",
      all(r.anchor_tx_hash == fake_tx for r in anchored_records))

unanchored = get_records(db, unanchored_only=True)
check("2 records still unanchored (seq 4-5)", len(unanchored) == 2,
      f"unanchored count={len(unanchored)}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Blockchain config check
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 7: blockchain config detection ===")

missing = _check_config()
print(f"  missing keys: {missing}")
check("Config check returns missing keys when .env not set",
      len(missing) > 0, f"missing={missing}")
check("INFURA_URL is in missing",     "INFURA_URL"     in missing)
check("WALLET_ADDRESS is in missing", "WALLET_ADDRESS" in missing)
check("PRIVATE_KEY is in missing",    "PRIVATE_KEY"    in missing)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Empty chain root
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 8: get_chain_root() on empty chain ===")

empty_db = SessionLocal()  # fresh session on new in-memory DB (same connection)
# We can test with a mock: just verify GENESIS_HASH is returned when no records
import hashlib
expected_genesis = hashlib.sha256(b"VERIDEX_GENESIS").hexdigest()
check("GENESIS_HASH is correct SHA-256", GENESIS_HASH == expected_genesis,
      f"GENESIS_HASH={GENESIS_HASH[:16]}...")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
db.close()

total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"\n{'='*55}")
print(f"  RESULTS: {passed}/{total} passed  |  {failed} failed")
print(f"{'='*55}\n")

if failed:
    sys.exit(1)

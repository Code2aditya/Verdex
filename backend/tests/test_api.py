"""API tests — auth, screen→audit→tamper-demo, and the multipart verify contract."""

from __future__ import annotations

OFFICER = {"username": "officer", "password": "officer123"}
SUPERVISOR = {"username": "supervisor", "password": "super123"}


def _token(client, creds: dict[str, str]) -> str:
    r = client.post("/api/auth/login", json=creds)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "access_token" in body and "officer_id" in body
    return body["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_login_me_and_presets(client) -> None:
    token = _token(client, OFFICER)
    me = client.get("/api/auth/me", headers=_auth(token))
    assert me.status_code == 200
    assert me.json()["officer_id"] == "SSB-OP-1042"

    presets = client.get("/api/screen/presets", headers=_auth(token))
    assert presets.status_code == 200
    assert presets.json()["count"] == 5


def test_bad_login_rejected(client) -> None:
    r = client.post("/api/auth/login", json={"username": "officer", "password": "wrong"})
    assert r.status_code in (401, 403)
    assert "access_token" not in r.json()


def test_screen_preset_appends_and_chain_intact(client) -> None:
    token = _token(client, OFFICER)

    screen = client.post("/api/screen", json={"preset": "D"}, headers=_auth(token))
    assert screen.status_code == 200, screen.text
    body = screen.json()
    assert body["level"] == "CRITICAL"
    assert body["audit"]["seq"] is not None

    audit = client.get("/api/audit/verify", headers=_auth(token))
    assert audit.status_code == 200
    report = audit.json()
    assert report["intact"] is True
    assert report["records"] >= 1
    assert report["chain_root"] not in (None, "")


def test_verify_multipart_contract_and_tamper_demo(client, sample_document_bytes) -> None:
    officer = _token(client, OFFICER)
    heads = _auth(officer)

    # 1. Multipart document verification writes an audit record.
    r = client.post(
        "/api/v1/verify-document",
        files={"file": ("doc.jpg", sample_document_bytes, "image/jpeg")},
        data={"document_type": "aadhaar", "officer_id": "SSB-OP-1042", "terminal_node": "TEST-NODE"},
        headers=heads,
    )
    assert r.status_code == 200, r.text
    contract = r.json()
    for key in ("risk_score", "risk_tier", "ocr", "validation", "tamper", "face", "liveness", "blockchain"):
        assert key in contract, key
    assert isinstance(contract["risk_score"], int)
    assert "audit" not in contract  # the Flutter contract drops the audit envelope

    # The screening event did land in the tamper-evident ledger.
    log = client.get("/api/audit/log?limit=5", headers=heads).json()
    assert log["total"] >= 1
    tail = log["records"][0]
    assert tail["doc_type"] == "aadhaar"
    assert tail["hash_chain"] not in (None, "")

    # 2. Flutter AppConstants endpoint works identically.
    r2 = client.post(
        "/api/verify",
        files={"file": ("doc.jpg", sample_document_bytes, "image/jpeg")},
        data={"document_type": "pan_card"},
        headers=heads,
    )
    assert r2.status_code == 200, r2.text
    assert "risk_score" in r2.json()

    # 3. Chain intact after verifications.
    assert client.get("/api/audit/verify", headers=heads).json()["intact"] is True

    # 4. Supervisor triggers the ledger-tamper demo → chain breaks.
    super_token = _token(client, SUPERVISOR)
    demo = client.post("/api/audit/tamper-demo", headers=_auth(super_token))
    assert demo.status_code == 200, demo.text
    assert demo.json()["verification"]["intact"] is False

    broken = client.get("/api/audit/verify", headers=heads).json()
    assert broken["intact"] is False
    assert broken["first_broken_seq"] == 1


def test_screen_offline_queues_encrypted(client) -> None:
    token = _token(client, OFFICER)
    r = client.post("/api/screen", json={"preset": "E", "offline": True}, headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued_offline"]["encrypted"] is True
    assert body["audit"] is None


def test_blockchain_status_and_anchor_simulated(client) -> None:
    officer = _token(client, OFFICER)
    super_token = _token(client, SUPERVISOR)
    officer_heads = _auth(officer)

    # Build a chain first.
    client.post("/api/screen", json={"preset": "B"}, headers=officer_heads)

    status = client.get("/api/blockchain/status", headers=officer_heads)
    assert status.status_code == 200

    anchor = client.post("/api/audit/anchor", headers=_auth(super_token))
    assert anchor.status_code == 200, anchor.text
    assert anchor.json()["success"] is True
    assert anchor.json()["mode"] == "simulated"

    history = client.get("/api/blockchain/history", headers=officer_heads)
    assert history.status_code == 200
    assert len(history.json()["anchors"]) >= 1
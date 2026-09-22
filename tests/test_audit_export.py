"""Tests for resumable snapshot-cursor audit export.

Covers: legacy no-argument compatibility, snapshot pinning, contiguous page
walks, stability under events appended during the export, resumption from an
interrupted cursor, restart resumption (signing key persisted), uniform
``invalid_cursor`` errors for tampered/cross-session/forged cursors, and
``audit_chain_tampered`` when the stored rows inside the snapshot window break
the hash chain.
"""

import importlib
import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DR_DB_PATH", str(tmp_path / "export.db"))
    import app.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def create(client, secret="export-secret", t=2, n=3, **extra):
    body = {"secret": secret, "encoding": "utf-8", "threshold": t, "total": n}
    body.update(extra)
    r = client.post("/api/v1/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def build_chain(client, sid, shares, *, partials=1, recover=True):
    """Append audit events: `partials` pending submissions, then recovery.

    Event count afterwards: 1 (genesis) + partials + (2 if recovered else 0),
    since a successful recovery appends both ``shares_submitted`` and
    ``session_recovered`` in the same transaction.
    """
    for i in range(partials):
        r = client.post(
            f"/api/v1/sessions/{sid}/shares", json={"shares": [shares[i]]}
        )
        assert r.status_code == 200
    if recover:
        r = client.post(
            f"/api/v1/sessions/{sid}/shares", json={"shares": shares[:2]}
        )
        assert r.status_code == 200
    return 1 + partials + (2 if recover else 0)


def walk_export(client, sid, page_limit, **params):
    """Follow next_cursor until the end; return (all_events, first_page)."""
    collected = []
    cursor = params.pop("cursor", None)
    first = None
    while True:
        kwargs = {"cursor": cursor} if cursor is not None else {}
        if first is None and "limit" not in kwargs:
            kwargs["limit"] = page_limit
        elif cursor is not None:
            kwargs["limit"] = page_limit
        r = client.get(f"/api/v1/sessions/{sid}/audit", params=kwargs)
        assert r.status_code == 200, r.text
        page = r.json()
        if first is None:
            first = page
        collected.extend(page["events"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    return collected, first


# --------------------------------------------------------------------------
# Legacy compatibility
# --------------------------------------------------------------------------

def test_no_arguments_keeps_legacy_shape(client):
    data = create(client)
    body = client.get(f"/api/v1/sessions/{data['session_id']}/audit").json()
    assert set(body.keys()) == {"session_id", "events", "chain"}
    assert body["chain"]["valid"] is True
    assert body["chain"]["event_count"] == 1
    assert "snapshot_seq" not in body and "next_cursor" not in body


def test_legacy_chain_reports_tampering_as_before(client):
    data = create(client)
    sid = data["session_id"]
    conn = sqlite3.connect(os.environ["DR_DB_PATH"])
    conn.execute(
        "UPDATE audit_events SET result='revoked' WHERE session_id=? AND seq=1",
        (sid,),
    )
    conn.commit()
    body = client.get(f"/api/v1/sessions/{sid}/audit").json()
    assert body["chain"]["valid"] is False
    assert body["chain"]["first_invalid_seq"] == 1


# --------------------------------------------------------------------------
# First page pins the snapshot
# --------------------------------------------------------------------------

def test_first_page_pins_snapshot_head(client):
    data = create(client)
    sid = data["session_id"]
    total = build_chain(client, sid, data["shares"], partials=2)  # 5 events
    page = client.get(f"/api/v1/sessions/{sid}/audit?limit=2").json()
    assert page["snapshot_seq"] == total
    assert len(page["events"]) == 2
    assert [e["seq"] for e in page["events"]] == [1, 2]
    assert page["next_cursor"]
    assert page["limit"] == 2
    # Snapshot hash is the hash of the chain head at request time.
    conn = sqlite3.connect(os.environ["DR_DB_PATH"])
    head = conn.execute(
        "SELECT event_hash FROM audit_events WHERE session_id=? AND seq=?",
        (sid, total),
    ).fetchone()
    assert page["snapshot_hash"] == head[0]


def test_single_page_export_has_null_cursor(client):
    data = create(client)  # only the genesis event
    page = client.get(
        f"/api/v1/sessions/{data['session_id']}/audit?limit=100"
    ).json()
    assert page["snapshot_seq"] == 1
    assert len(page["events"]) == 1
    assert page["next_cursor"] is None
    assert page["events"][0]["event_hash"] == page["snapshot_hash"]


def test_events_have_legacy_shape_and_are_valid(client):
    data = create(client)
    page = client.get(
        f"/api/v1/sessions/{data['session_id']}/audit?limit=10"
    ).json()
    ev = page["events"][0]
    assert set(ev.keys()) == {
        "seq", "type", "timestamp", "result", "details",
        "prev_hash", "event_hash", "valid",
    }
    assert ev["valid"] is True and ev["prev_hash"] == ""


# --------------------------------------------------------------------------
# Full walks: contiguous pages, tail == snapshot hash
# --------------------------------------------------------------------------

@pytest.mark.parametrize("page_limit", [1, 2, 3, 4, 5, 7, 100])
def test_walk_all_page_sizes_is_contiguous_and_complete(client, page_limit):
    data = create(client)
    sid = data["session_id"]
    total = build_chain(client, sid, data["shares"], partials=2)  # 5 events
    events, first = walk_export(client, sid, page_limit)
    assert [e["seq"] for e in events] == list(range(1, total + 1))
    # prev_hash linkage is contiguous across page boundaries.
    prev = ""
    for e in events:
        assert e["prev_hash"] == prev
        assert e["valid"] is True
        prev = e["event_hash"]
    assert first["snapshot_seq"] == total
    assert events[-1]["event_hash"] == first["snapshot_hash"]


def test_default_limit_applies_with_cursor_only_first_call(client):
    data = create(client)
    sid = data["session_id"]
    total = build_chain(client, sid, data["shares"], partials=1)  # 4 events
    page = client.get(f"/api/v1/sessions/{sid}/audit?limit=2").json()
    assert page["limit"] == 2
    nxt = client.get(
        f"/api/v1/sessions/{sid}/audit", params={"cursor": page["next_cursor"]}
    ).json()
    assert nxt["limit"] == 100  # default on subsequent requests
    assert [e["seq"] for e in nxt["events"]] == [3, 4]
    assert nxt["next_cursor"] is None


# --------------------------------------------------------------------------
# Snapshot stability: events appended during export never enter the pages
# --------------------------------------------------------------------------

def test_events_appended_during_export_do_not_change_pages(client):
    data = create(client)
    sid = data["session_id"]
    # Genesis + one pending event -> snapshot of 2.
    client.post(
        f"/api/v1/sessions/{sid}/shares", json={"shares": [data["shares"][0]]}
    )
    first = client.get(f"/api/v1/sessions/{sid}/audit?limit=1").json()
    assert first["snapshot_seq"] == 2
    snapshot_seq = first["snapshot_seq"]
    snapshot_hash = first["snapshot_hash"]

    # Recovery appends events 3 and 4 AFTER the snapshot was taken.
    client.post(
        f"/api/v1/sessions/{sid}/shares", json={"shares": data["shares"][:2]}
    )
    page2 = client.get(
        f"/api/v1/sessions/{sid}/audit",
        params={"cursor": first["next_cursor"]},
    ).json()
    assert [e["seq"] for e in page2["events"]] == [2]
    assert page2["next_cursor"] is None  # end of the SNAPSHOT, not of the chain
    assert page2["snapshot_seq"] == snapshot_seq
    assert page2["snapshot_hash"] == snapshot_hash
    # Final chain head of the export equals the first snapshot's head.
    assert page2["events"][-1]["event_hash"] == snapshot_hash

    # A fresh export sees the newer head.
    fresh = client.get(f"/api/v1/sessions/{sid}/audit?limit=100").json()
    assert fresh["snapshot_seq"] == 4
    assert fresh["snapshot_hash"] != snapshot_hash


def test_resume_from_interrupted_cursor(client):
    data = create(client)
    sid = data["session_id"]
    total = build_chain(client, sid, data["shares"], partials=2)  # 5 events
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=1").json()
    # Simulate an interruption: the same cursor is retried any number of times.
    for _ in range(3):
        again = client.get(
            f"/api/v1/sessions/{sid}/audit",
            params={"cursor": p1["next_cursor"], "limit": 1},
        ).json()
        assert [e["seq"] for e in again["events"]] == [2]
        assert again["snapshot_seq"] == p1["snapshot_seq"]
    # Then continue to the end one event at a time.
    pages = []
    cursor = again["next_cursor"]
    while cursor is not None:
        nxt = client.get(
            f"/api/v1/sessions/{sid}/audit",
            params={"cursor": cursor, "limit": 1},
        ).json()
        pages.extend(nxt["events"])
        cursor = nxt["next_cursor"]
    assert [e["seq"] for e in pages] == [3, 4, 5]
    assert pages[-1]["event_hash"] == p1["snapshot_hash"]


def test_replaying_final_cursor_returns_empty_page_with_null_cursor(client):
    data = create(client)
    sid = data["session_id"]
    total = build_chain(client, sid, data["shares"], partials=1)  # 4 events
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=2").json()
    final = client.get(
        f"/api/v1/sessions/{sid}/audit",
        params={"cursor": p1["next_cursor"], "limit": 2},
    ).json()
    assert [e["seq"] for e in final["events"]] == [3, 4]
    assert final["next_cursor"] is None
    # Export already finished: replaying the previous cursor is idempotent.
    replay = client.get(
        f"/api/v1/sessions/{sid}/audit",
        params={"cursor": p1["next_cursor"], "limit": 2},
    ).json()
    assert [e["seq"] for e in replay["events"]] == [3, 4]
    assert replay["next_cursor"] is None


def test_cursor_survives_service_restart(tmp_path, monkeypatch):
    """The HMAC signing key is persisted, so cursors resume after restart."""
    db_path = str(tmp_path / "restart.db")
    monkeypatch.setenv("DR_DB_PATH", db_path)
    import app.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        data = create(c)
        sid = data["session_id"]
        total = build_chain(c, sid, data["shares"], partials=2)  # 5 events
        p1 = c.get(f"/api/v1/sessions/{sid}/audit?limit=2").json()
        assert p1["next_cursor"]
        cursor = p1["next_cursor"]
        snapshot_seq = p1["snapshot_seq"]
        snapshot_hash = p1["snapshot_hash"]

    # Restart: rebuild the app/database against the same file.
    importlib.reload(main)
    with TestClient(main.app) as c:
        p2 = c.get(
            f"/api/v1/sessions/{sid}/audit",
            params={"cursor": cursor, "limit": 10},
        ).json()
        assert [e["seq"] for e in p2["events"]] == list(range(3, total + 1))
        assert p2["next_cursor"] is None
        assert p2["snapshot_seq"] == snapshot_seq
        assert p2["snapshot_hash"] == snapshot_hash


# --------------------------------------------------------------------------
# Invalid cursors: uniform error for every tampering/reuse shape
# --------------------------------------------------------------------------

def _b64(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def test_garbage_cursor_rejected_uniformly(client):
    data = create(client)
    for token in ("garbage", "a.b.c", "", ".", "abc.", ".abc", "@@@.@@@"):
        r = client.get(
            f"/api/v1/sessions/{data['session_id']}/audit",
            params={"cursor": token},
        )
        assert r.status_code == 400, token
        assert r.json()["error"]["code"] == "invalid_cursor", token


def test_oversized_cursor_rejected_as_invalid_not_validation_error(client):
    data = create(client)
    r = client.get(
        f"/api/v1/sessions/{data['session_id']}/audit",
        params={"cursor": "x." + "b" * 2000},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


def test_cursor_with_bad_signature_rejected(client):
    data = create(client)
    sid = data["session_id"]
    build_chain(client, sid, data["shares"], partials=1)
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=1").json()
    body, signature = p1["next_cursor"].split(".")
    import base64

    raw_sig = bytearray(base64.urlsafe_b64decode(signature + "=="))
    raw_sig[0] ^= 0x01
    forged = body + "." + _b64(bytes(raw_sig))
    r = client.get(
        f"/api/v1/sessions/{sid}/audit", params={"cursor": forged, "limit": 1}
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


def test_cursor_payload_tampering_rejected(client):
    """Rewriting signed payload fields (snapshot seq/hash, boundary) fails MAC."""
    data = create(client)
    sid = data["session_id"]
    build_chain(client, sid, data["shares"], partials=1)
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=1").json()
    body_b64, sig_b64 = p1["next_cursor"].split(".")
    import base64

    payload = json.loads(base64.urlsafe_b64decode(body_b64 + "=="))
    for mutated in (
        {**payload, "sseq": payload["sseq"] + 1},
        {**payload, "lseq": payload["lseq"] + 1},
        {**payload, "shash": "0" * 64},
        {**payload, "lhash": "f" * 64},
        {**payload, "v": 2},
    ):
        new_body = _b64(
            json.dumps(mutated, separators=(",", ":")).encode()
        )
        r = client.get(
            f"/api/v1/sessions/{sid}/audit",
            params={"cursor": new_body + "." + sig_b64, "limit": 1},
        )
        assert r.status_code == 400, mutated
        assert r.json()["error"]["code"] == "invalid_cursor"


def test_cursor_bound_to_other_session_rejected(client):
    a = create(client, secret="aaa")
    build_chain(client, a["session_id"], a["shares"], partials=1)
    b = create(client, secret="bbb")
    p1 = client.get(
        f"/api/v1/sessions/{a['session_id']}/audit?limit=1"
    ).json()
    r = client.get(
        f"/api/v1/sessions/{b['session_id']}/audit",
        params={"cursor": p1["next_cursor"]},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


def test_cursor_signed_by_another_server_rejected(client):
    data = create(client)
    sid = data["session_id"]
    build_chain(client, sid, data["shares"], partials=1)
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=1").json()
    # Rotate the server's signing key (e.g. rebuilt server with another DB):
    # old cursors must stop validating.
    conn = sqlite3.connect(os.environ["DR_DB_PATH"])
    conn.execute(
        "UPDATE app_meta SET value=? WHERE key='cursor_signing_key'",
        ("11" * 32,),
    )
    conn.commit()
    conn.close()
    client.app.state.db._cursor_signing_key = None
    r = client.get(
        f"/api/v1/sessions/{sid}/audit",
        params={"cursor": p1["next_cursor"], "limit": 1},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


# --------------------------------------------------------------------------
# Database boundary tampering after a cursor was issued
# --------------------------------------------------------------------------

def _sql(db_path, statement, *args):
    conn = sqlite3.connect(db_path)
    conn.execute(statement, args)
    conn.commit()
    conn.close()


def test_tampering_snapshot_head_after_issue_invalidates_cursor(client):
    data = create(client)
    sid = data["session_id"]
    total = build_chain(client, sid, data["shares"], partials=2)
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=2").json()
    assert p1["snapshot_seq"] == total
    # Rewrite the snapshot head's stored hash.
    _sql(
        os.environ["DR_DB_PATH"],
        "UPDATE audit_events SET event_hash=? WHERE session_id=? AND seq=?",
        "0" * 64, sid, total,
    )
    r = client.get(
        f"/api/v1/sessions/{sid}/audit",
        params={"cursor": p1["next_cursor"], "limit": 2},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


def test_tampering_delivered_boundary_invalidates_cursor(client):
    data = create(client)
    sid = data["session_id"]
    build_chain(client, sid, data["shares"], partials=2)  # 4 events
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=2").json()
    # The cursor's boundary is event 2's hash; rewrite that event's hash.
    _sql(
        os.environ["DR_DB_PATH"],
        "UPDATE audit_events SET event_hash=? WHERE session_id=? AND seq=2",
        "a" * 64, sid,
    )
    r = client.get(
        f"/api/v1/sessions/{sid}/audit",
        params={"cursor": p1["next_cursor"], "limit": 2},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


def test_tampering_remaining_window_reports_chain_error(client):
    data = create(client)
    sid = data["session_id"]
    build_chain(client, sid, data["shares"], partials=2)  # 5 events
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=2").json()
    # Tamper content of an event still inside the unread window (seq 3) while
    # leaving the snapshot head row untouched: page verification must fail.
    _sql(
        os.environ["DR_DB_PATH"],
        "UPDATE audit_events SET result='revoked' WHERE session_id=? AND seq=3",
        sid,
    )
    r = client.get(
        f"/api/v1/sessions/{sid}/audit",
        params={"cursor": p1["next_cursor"], "limit": 2},
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "audit_chain_tampered"
    assert err["details"]["first_invalid_seq"] == 3
    assert err["details"]["failure_reason"] == "event_content_tampered"


def test_deleted_event_inside_window_reports_gap(client):
    data = create(client)
    sid = data["session_id"]
    build_chain(client, sid, data["shares"], partials=2)  # 5 events
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=2").json()
    _sql(
        os.environ["DR_DB_PATH"],
        "DELETE FROM audit_events WHERE session_id=? AND seq=3",
        sid,
    )
    r = client.get(
        f"/api/v1/sessions/{sid}/audit",
        params={"cursor": p1["next_cursor"], "limit": 2},
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "audit_chain_tampered"
    assert err["details"]["failure_reason"] == "gap_or_duplicate_sequence"


def test_first_page_detects_already_broken_chain(client):
    data = create(client)
    sid = data["session_id"]
    build_chain(client, sid, data["shares"], partials=1)  # 3 events
    _sql(
        os.environ["DR_DB_PATH"],
        "UPDATE audit_events SET prev_hash=? WHERE session_id=? AND seq=2",
        "f" * 64, sid,
    )
    r = client.get(f"/api/v1/sessions/{sid}/audit?limit=10")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "audit_chain_tampered"
    assert err["details"]["first_invalid_seq"] == 2
    assert err["details"]["failure_reason"] == "previous_hash_mismatch"


def test_truncated_snapshot_head_rejects_cursor(client):
    data = create(client)
    sid = data["session_id"]
    total = build_chain(client, sid, data["shares"], partials=2)
    p1 = client.get(f"/api/v1/sessions/{sid}/audit?limit=2").json()
    # Delete the snapshot head itself.
    _sql(
        os.environ["DR_DB_PATH"],
        "DELETE FROM audit_events WHERE session_id=? AND seq=?",
        sid, total,
    )
    r = client.get(
        f"/api/v1/sessions/{sid}/audit",
        params={"cursor": p1["next_cursor"], "limit": 2},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_cursor"


# --------------------------------------------------------------------------
# Parameter handling and edge cases
# --------------------------------------------------------------------------

def test_limit_validation(client):
    data = create(client)
    sid = data["session_id"]
    assert client.get(f"/api/v1/sessions/{sid}/audit?limit=0").status_code == 422
    assert client.get(f"/api/v1/sessions/{sid}/audit?limit=-1").status_code == 422
    assert client.get(
        f"/api/v1/sessions/{sid}/audit?limit=1001"
    ).status_code == 422
    assert client.get(f"/api/v1/sessions/{sid}/audit?limit=abc").status_code == 422


def test_paged_export_for_unknown_session_is_404(client):
    r = client.get("/api/v1/sessions/deadbeef/audit?limit=10")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "session_not_found"
    r = client.get(
        "/api/v1/sessions/deadbeef/audit", params={"cursor": "x.y"}
    )
    assert r.status_code == 404


def test_session_without_any_events_exports_empty_snapshot(client, tmp_path):
    """A legacy/migrated session row with zero audit events: snapshot seq 0."""
    from app.database import Database

    db: Database = client.app.state.db
    conn = sqlite3.connect(os.environ["DR_DB_PATH"])
    conn.execute(
        "INSERT INTO sessions (id, threshold, total_shares, key_digest, "
        "key_encoding, key_length, status, created_at) "
        "VALUES ('legacy', 2, 3, 'd', 'utf-8', 1, 'active', 'now')"
    )
    conn.commit()
    conn.close()
    r = client.get("/api/v1/sessions/legacy/audit?limit=10")
    assert r.status_code == 200, r.text
    page = r.json()
    assert page["events"] == []
    assert page["snapshot_seq"] == 0
    assert page["snapshot_hash"] == ""
    assert page["next_cursor"] is None


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

class TestExportApiKeyAuth:
    @pytest.fixture()
    def authed_client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DR_DB_PATH", str(tmp_path / "auth-export.db"))
        monkeypatch.setenv("DR_API_KEY", "export-token")
        import app.main as main

        importlib.reload(main)
        with TestClient(main.app) as c:
            yield c

    def test_paged_export_requires_api_key(self, authed_client):
        r = authed_client.get("/api/v1/sessions/whatever/audit?limit=10")
        assert r.status_code == 401

    def test_paged_export_works_with_api_key(self, authed_client):
        r = authed_client.post(
            "/api/v1/sessions",
            json={"secret": "x", "threshold": 2, "total": 3},
            headers={"X-API-Key": "export-token"},
        )
        sid = r.json()["session_id"]
        r = authed_client.get(
            f"/api/v1/sessions/{sid}/audit?limit=10",
            headers={"X-API-Key": "export-token"},
        )
        assert r.status_code == 200
        assert r.json()["snapshot_seq"] == 1


# --------------------------------------------------------------------------
# Cursor codec unit tests
# --------------------------------------------------------------------------

class TestCursorCodec:
    def _key(self):
        return b"0" * 64

    def test_roundtrip(self):
        from app import audit_cursor

        token = audit_cursor.make_cursor(
            session_id="sid", snapshot_seq=5, snapshot_hash="a" * 64,
            last_seq=2, last_hash="b" * 64, signing_key=self._key(),
        )
        payload = audit_cursor.decode_cursor(token, self._key(), "sid")
        assert payload == {
            "session_id": "sid",
            "snapshot_seq": 5,
            "snapshot_hash": "a" * 64,
            "last_seq": 2,
            "last_hash": "b" * 64,
        }

    def test_genesis_boundary_roundtrip(self):
        from app import audit_cursor

        token = audit_cursor.make_cursor(
            session_id="sid", snapshot_seq=0, snapshot_hash="",
            last_seq=0, last_hash="", signing_key=self._key(),
        )
        payload = audit_cursor.decode_cursor(token, self._key(), "sid")
        assert payload["snapshot_seq"] == 0 and payload["last_seq"] == 0

    def test_wrong_key_rejected(self):
        from app import audit_cursor

        token = audit_cursor.make_cursor(
            session_id="sid", snapshot_seq=1, snapshot_hash="a" * 64,
            last_seq=0, last_hash="", signing_key=self._key(),
        )
        with pytest.raises(audit_cursor.InvalidCursor):
            audit_cursor.decode_cursor(token, b"1" * 64, "sid")

    def test_cross_session_rejected(self):
        from app import audit_cursor

        token = audit_cursor.make_cursor(
            session_id="sid-a", snapshot_seq=1, snapshot_hash="a" * 64,
            last_seq=0, last_hash="", signing_key=self._key(),
        )
        with pytest.raises(audit_cursor.InvalidCursor):
            audit_cursor.decode_cursor(token, self._key(), "sid-b")

    @pytest.mark.parametrize(
        "token",
        [
            "",
            "nodot",
            "a.b.c",
            "!!!.###",
            "eyJ2IjoxfQ.aaaa",  # structurally b64 but bad signature/JSON
        ],
    )
    def test_malformed_tokens(self, token):
        from app import audit_cursor

        with pytest.raises(audit_cursor.InvalidCursor):
            audit_cursor.decode_cursor(token, self._key(), "sid")

"""Stateless, resumable cursors for snapshot audit exports.

An export cursor pins one page position inside an immutable *snapshot* of a
session's audit chain::

    snapshot_seq / snapshot_hash
        head of the chain at the time of the first paged request; later pages
        only ever read events with ``seq <= snapshot_seq``.
    last_seq / last_hash
        last event already delivered ("boundary"); the next page starts at
        ``last_seq + 1`` and must link to ``last_hash``.
    session_id
        the cursor is valid for this session only.

The cursor is a signed, self-contained token -- the server keeps no export
state::

    base64url(canonical_json_payload) + "." + base64url(HMAC-SHA256(payload, key))

The HMAC key is generated once and persisted in ``app_meta`` (see
:mod:`app.database`), so interrupted exports remain resumable after a restart.
Any structural problem, signature mismatch, bound-session mismatch or hash
that no longer matches the stored database boundary is reported as
:class:`InvalidCursor`; the endpoint maps every such case to one uniform
``invalid_cursor`` error, so tampering probes and cross-session reuse cannot be
distinguished and learn nothing.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import Any

CURSOR_VERSION = 1
MAX_CURSOR_CHARS = 512


class InvalidCursor(Exception):
    """The cursor token is malformed, unsigned for this server, tampered with,
    bound to another session, or otherwise unusable."""


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    try:
        padding = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + padding)
    except (binascii.Error, ValueError) as exc:
        raise InvalidCursor("cursor is not valid base64url") from exc


def encode_cursor(payload: dict[str, Any], signing_key: bytes) -> str:
    """Sign and serialise a cursor payload (the canonical form is signed)."""
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    signature = hmac.new(signing_key, body, hashlib.sha256).digest()
    return _b64url_encode(body) + "." + _b64url_encode(signature)


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; it is not a valid sequence number.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def decode_cursor(
    token: str, signing_key: bytes, expected_session_id: str
) -> dict[str, Any]:
    """Verify and parse a cursor token for ``expected_session_id``.

    Performs the structural and signature checks; snapshot/boundary agreement
    with the current database rows is the caller's responsibility (it needs the
    persistence layer). Raises :class:`InvalidCursor` on any problem.
    """
    if not isinstance(token, str) or not token or len(token) > MAX_CURSOR_CHARS:
        raise InvalidCursor("cursor missing or too long")
    parts = token.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise InvalidCursor("cursor has the wrong structure")

    body_b64, signature_b64 = parts
    body = _b64url_decode(body_b64)
    provided_signature = _b64url_decode(signature_b64)
    expected_signature = hmac.new(
        signing_key, body, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(provided_signature, expected_signature):
        raise InvalidCursor("cursor signature does not match")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidCursor("cursor payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidCursor("cursor payload is not an object")

    v = payload.get("v")
    if not _is_int(v) or v != CURSOR_VERSION:
        raise InvalidCursor("unsupported cursor version")
    session_id = payload.get("sid")
    if not isinstance(session_id, str) or not 1 <= len(session_id) <= 64:
        raise InvalidCursor("cursor has no bound session")
    # Constant-time: never reveal whether the cursor belongs to another session.
    if not hmac.compare_digest(session_id, expected_session_id):
        raise InvalidCursor("cursor is bound to another session")

    snap_seq = payload.get("sseq")
    last_seq = payload.get("lseq")
    if not _is_int(snap_seq) or not _is_int(last_seq):
        raise InvalidCursor("cursor sequence fields are not integers")
    if snap_seq < 0 or last_seq < 0 or last_seq > snap_seq:
        raise InvalidCursor("cursor sequence fields are out of range")

    snap_hash = payload.get("shash")
    last_hash = payload.get("lhash")
    # The genesis boundary is the empty string; every other boundary hash is
    # 64 lowercase hex chars.
    if not isinstance(snap_hash, str) or not (
        snap_hash == "" or _is_hex64(snap_hash)
    ):
        raise InvalidCursor("cursor snapshot hash is invalid")
    if not isinstance(last_hash, str) or not (
        last_hash == "" or _is_hex64(last_hash)
    ):
        raise InvalidCursor("cursor boundary hash is invalid")
    if snap_seq == 0 and snap_hash != "":
        raise InvalidCursor("empty snapshot must pin the genesis hash")
    if last_seq == 0 and last_hash != "":
        raise InvalidCursor("genesis boundary must use the empty hash")

    return {
        "session_id": session_id,
        "snapshot_seq": snap_seq,
        "snapshot_hash": snap_hash,
        "last_seq": last_seq,
        "last_hash": last_hash,
    }


def make_cursor(
    *,
    session_id: str,
    snapshot_seq: int,
    snapshot_hash: str,
    last_seq: int,
    last_hash: str,
    signing_key: bytes,
) -> str:
    """Build a signed cursor advancing the export to ``(last_seq, last_hash)``."""
    return encode_cursor(
        {
            "v": CURSOR_VERSION,
            "sid": session_id,
            "sseq": snapshot_seq,
            "shash": snapshot_hash,
            "lseq": last_seq,
            "lhash": last_hash,
        },
        signing_key,
    )

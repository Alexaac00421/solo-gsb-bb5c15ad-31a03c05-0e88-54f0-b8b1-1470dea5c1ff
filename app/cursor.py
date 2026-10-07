"""Opaque, recoverable snapshot cursors for paged audit export.

A cursor binds together:

* ``session_id``  -- the only session the cursor may be used with,
* ``snapshot_seq`` -- the chain head fixed by the *first* export request,
* ``snapshot_hash`` -- the ``event_hash`` of that chain head,
* ``after_seq`` -- the highest sequence already delivered to the client,
* ``after_hash`` -- the ``event_hash`` delivered at ``after_seq`` (the
  predecessor hash of the next page), used as a database-boundary check.

Only events with ``seq <= snapshot_seq`` are ever read through a cursor, so an
export is a stable, replayable snapshot even while new audit events keep being
appended. Cursors are stateless on the server: the payload is authenticated
with an HMAC-SHA256 key persisted in the database (see
:meth:`app.database.Database.get_export_key`), so cursors remain resumable
across service restarts while every edit, truncation or cross-session reuse is
rejected as an invalid cursor.

Wire format (URL-safe, no padding)::

    base64url(payload_json) + "." + base64url(hmac_sha256(key, payload))
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import Any, Mapping

# Cursor payload fields. Keep the document tiny and canonical (sorted keys on
# decode for comparison-independent parsing).
FIELD_SESSION = "session_id"
FIELD_SNAPSHOT_SEQ = "snapshot_seq"
FIELD_SNAPSHOT_HASH = "snapshot_hash"
FIELD_AFTER_SEQ = "after_seq"
FIELD_AFTER_HASH = "after_hash"

# A signed cursor for a few hundred events is well under this bound.
MAX_CURSOR_BYTES = 2048

# Bounds shared with the API layer.
MIN_LIMIT = 1
MAX_LIMIT = 1000
DEFAULT_LIMIT = 200


class InvalidCursor(Exception):
    """The cursor token is malformed, forged or has a bad signature."""


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    try:
        # Padding is stripped on encode; re-add it before decoding.
        padded = text + "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise InvalidCursor("cursor is not valid base64") from exc


def encode_cursor(
    key: bytes,
    *,
    session_id: str,
    snapshot_seq: int,
    snapshot_hash: str,
    after_seq: int,
    after_hash: str,
) -> str:
    """Build an HMAC-authenticated opaque cursor token."""
    payload = {
        FIELD_SESSION: session_id,
        FIELD_SNAPSHOT_SEQ: snapshot_seq,
        FIELD_SNAPSHOT_HASH: snapshot_hash,
        FIELD_AFTER_SEQ: after_seq,
        FIELD_AFTER_HASH: after_hash,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    signature = hmac.new(key, body, hashlib.sha256).digest()
    return _b64encode(body) + "." + _b64encode(signature)


def decode_cursor(key: bytes, token: Any) -> dict[str, Any]:
    """Validate an opaque cursor token and return its verified payload.

    Raises :class:`InvalidCursor` for any malformed input, signature mismatch
    or non-canonical payload; callers map it onto a single uniform error.
    """
    if not isinstance(token, str) or not token or len(token) > MAX_CURSOR_BYTES:
        raise InvalidCursor("malformed cursor")
    body_text, sep, sig_text = token.partition(".")
    if not sep or not body_text or not sig_text:
        raise InvalidCursor("malformed cursor")

    body = _b64decode(body_text)
    signature = _b64decode(sig_text)
    expected = hmac.new(key, body, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, signature):
        raise InvalidCursor("cursor signature mismatch")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursor("cursor payload is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise InvalidCursor("cursor payload is not an object")
    # A re-serialised payload must reproduce the signed bytes exactly, so no
    # duplicate keys / odd whitespace / unexpected encoding can sneak through.
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if canonical != body:
        raise InvalidCursor("cursor payload is not canonical")

    _require(payload, FIELD_SESSION, str)
    _require(payload, FIELD_SNAPSHOT_SEQ, int)
    _require(payload, FIELD_SNAPSHOT_HASH, str)
    _require(payload, FIELD_AFTER_SEQ, int)
    _require(payload, FIELD_AFTER_HASH, str)
    if set(payload) != {
        FIELD_SESSION,
        FIELD_SNAPSHOT_SEQ,
        FIELD_SNAPSHOT_HASH,
        FIELD_AFTER_SEQ,
        FIELD_AFTER_HASH,
    }:
        raise InvalidCursor("cursor payload has unexpected fields")
    if not payload[FIELD_SESSION]:
        raise InvalidCursor("cursor session is empty")
    if payload[FIELD_SNAPSHOT_SEQ] < 1:
        raise InvalidCursor("cursor snapshot_seq out of range")
    if payload[FIELD_AFTER_SEQ] < 0:
        raise InvalidCursor("cursor after_seq out of range")
    if payload[FIELD_AFTER_SEQ] > payload[FIELD_SNAPSHOT_SEQ]:
        raise InvalidCursor("cursor after_seq exceeds snapshot")
    # Both hashes are lowercase hex SHA-256 digests; after_seq == 0 anchors on
    # the genesis boundary, whose hash is the empty string.
    if not _is_sha256_hex(payload[FIELD_SNAPSHOT_HASH]):
        raise InvalidCursor("cursor snapshot_hash is invalid")
    if payload[FIELD_AFTER_SEQ] == 0:
        if payload[FIELD_AFTER_HASH] != "":
            raise InvalidCursor("cursor after_hash is invalid")
    elif not _is_sha256_hex(payload[FIELD_AFTER_HASH]):
        raise InvalidCursor("cursor after_hash is invalid")
    return dict(payload)


def _require(payload: Mapping[str, Any], field: str, expected_type: type) -> None:
    value = payload.get(field)
    # ``bool`` is a subclass of ``int``; never accept it for sequence numbers.
    if not isinstance(value, expected_type) or (
        expected_type is int and isinstance(value, bool)
    ):
        raise InvalidCursor(f"cursor field {field} has the wrong type")


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()

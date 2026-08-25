"""Pure-Python policy primitives for SafeKV namespace and public-prefix control."""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple


class Visibility(str, Enum):
    """Visibility state of a SafeKV cache variant."""

    PRIVATE = "private"
    CANDIDATE = "candidate"
    BUDGETED_SHARED = "budgeted_shared"
    EXHAUSTED_PRIVATE = "exhausted_private"
    VERIFIED_PUBLIC = "verified_public"


@dataclass(frozen=True)
class PublicAuthorization:
    """Immutable operator authorization for one exact public prefix."""

    public_object_id: str
    issuer: str
    model_id: str
    tokenizer_version: str
    prefix_token_length: int
    prefix_fingerprint: str
    policy_epoch: int
    expires_at: float
    revoked: bool
    mac: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PublicAuthorization":
        """Parse an authorization received over a JSON-compatible API."""

        return cls(
            public_object_id=str(value["public_object_id"]),
            issuer=str(value["issuer"]),
            model_id=str(value["model_id"]),
            tokenizer_version=str(value["tokenizer_version"]),
            prefix_token_length=int(value["prefix_token_length"]),
            prefix_fingerprint=str(value["prefix_fingerprint"]),
            policy_epoch=int(value["policy_epoch"]),
            expires_at=float(value["expires_at"]),
            revoked=bool(value["revoked"]),
            mac=str(value["mac"]),
        )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "public_object_id": self.public_object_id,
            "issuer": self.issuer,
            "model_id": self.model_id,
            "tokenizer_version": self.tokenizer_version,
            "prefix_token_length": self.prefix_token_length,
            "prefix_fingerprint": self.prefix_fingerprint,
            "policy_epoch": self.policy_epoch,
            "expires_at": self.expires_at,
            "revoked": self.revoked,
            "mac": self.mac,
        }


@dataclass(frozen=True)
class VerificationResult:
    """Result of authorization verification with a stable reason code."""

    valid: bool
    reason: str

    def __bool__(self) -> bool:
        """Return whether verification succeeded."""

        return self.valid

    def __iter__(self) -> Iterator[object]:
        """Allow compatibility-friendly ``valid, reason = result`` unpacking."""

        yield self.valid
        yield self.reason


@dataclass(frozen=True)
class NamespaceKey:
    """Cache namespace key that separates private and public identities."""

    visibility: Visibility
    identity: str
    fingerprint: str

    @classmethod
    def private(cls, principal: str, fingerprint: str) -> "NamespaceKey":
        """Create a private key scoped to a principal."""

        return cls(Visibility.PRIVATE, principal, fingerprint)

    @classmethod
    def public(cls, object_id: str, fingerprint: str) -> "NamespaceKey":
        """Create a public key scoped to an authorized object."""

        return cls(Visibility.VERIFIED_PUBLIC, object_id, fingerprint)

    @classmethod
    def Private(cls, principal: str, fingerprint: str) -> "NamespaceKey":
        """Create a private key using the policy notation."""

        return cls.private(principal, fingerprint)

    @classmethod
    def Public(cls, object_id: str, fingerprint: str) -> "NamespaceKey":
        """Create a public key using the policy notation."""

        return cls.public(object_id, fingerprint)


@dataclass(frozen=True)
class VariantMetadata:
    """Policy metadata attached to a cache variant."""

    namespace_key: NamespaceKey
    visibility: Visibility
    authorization: Optional[PublicAuthorization] = None


@dataclass(frozen=True)
class SafeKVEvent:
    """Immutable structured SafeKV policy event."""

    name: str
    timestamp: float
    attributes: Mapping[str, object]


class SafeKVMetrics:
    """Thread-safe SafeKV counters and structured event storage."""

    COUNTER_NAMES: Tuple[str, ...] = (
        "unauth_public_promotions",
        "victim_node_relabels",
        "private_address_aliases",
        "cross_tenant_private_hits",
        "public_object_created",
        # Balanced-mode counters
        "cross_tenant_balanced_hits",   # hits served on BUDGETED_SHARED nodes
        "budget_exhausted_nodes",       # nodes demoted BUDGETED_SHARED → EXHAUSTED_PRIVATE
    )

    def __init__(self) -> None:
        """Initialize empty counters and event storage."""

        self._lock = threading.RLock()
        self._counters: Dict[str, int] = {
            name: 0 for name in self.COUNTER_NAMES
        }
        self._events: List[SafeKVEvent] = []

    def increment(
        self, counter: str, amount: int = 1, **attributes: object
    ) -> int:
        """Increment a known counter and optionally emit a matching event."""

        if counter not in self.COUNTER_NAMES:
            raise ValueError("unknown SafeKV counter: %s" % counter)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError("amount must be a non-negative integer")
        with self._lock:
            self._counters[counter] += amount
            value = self._counters[counter]
            if attributes:
                self._events.append(
                    SafeKVEvent(
                        counter,
                        time.time(),
                        MappingProxyType(dict(attributes)),
                    )
                )
            return value

    def record_event(self, name: str, **attributes: object) -> SafeKVEvent:
        """Append a structured event without changing a counter."""

        event = SafeKVEvent(
            name, time.time(), MappingProxyType(dict(attributes))
        )
        with self._lock:
            self._events.append(event)
        return event

    def events(self) -> List[SafeKVEvent]:
        """Return a stable copy of recorded events."""

        with self._lock:
            return list(self._events)

    def snapshot(self) -> Mapping[str, object]:
        """Return an immutable point-in-time metrics snapshot."""

        with self._lock:
            return MappingProxyType(
                {
                    "counters": MappingProxyType(dict(self._counters)),
                    "events": tuple(self._events),
                }
            )

    def reset(self) -> None:
        """Reset all counters and discard all events atomically."""

        with self._lock:
            for name in self._counters:
                self._counters[name] = 0
            self._events.clear()


class DurableLedger:
    """Persist cross-tenant access budgets across eviction and process restarts.

    Each entry maps a fingerprint (SHA-256 hex of model + tokenizer + tokens)
    to the cumulative number of cross-tenant hits served across *all* residencies
    within the current accounting epoch.  When a node is evicted, its hit total
    is flushed here.  On reinsertion the node restores from the ledger so the
    lifetime budget cannot be reset by eviction.

    Thread-safety: an RLock guards all reads and writes.
    Durability: changes are fsynced to the backing JSON file on every flush.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """Create a ledger optionally backed by *path* on disk."""
        self._path: Optional[str] = path
        self._lock = threading.RLock()
        self._store: Dict[str, int] = {}
        self._available = True
        self._ready = True
        self._load_error: Optional[str] = None
        if path is not None:
            self._load()

    def _load(self) -> None:
        try:
            import json as _json
            with open(self._path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            with self._lock:
                self._store = {k: int(v) for k, v in data.items()}
                self._ready = True
                self._load_error = None
        except FileNotFoundError:
            self._ready = True
            self._load_error = None
        except Exception as exc:
            # Unknown exposure state must never be interpreted as an empty budget.
            self._ready = False
            self._load_error = str(exc)

    @property
    def is_operational(self) -> bool:
        """Return whether reservations can be served safely."""
        with self._lock:
            return self._available and self._ready

    @property
    def load_error(self) -> Optional[str]:
        """Return the most recent recovery error, if any."""
        with self._lock:
            return self._load_error

    def set_available(self, available: bool) -> None:
        """Set backend availability (used by health handling and fault tests)."""
        with self._lock:
            self._available = bool(available)

    @contextmanager
    def _process_lock(self):
        """Serialize disk-backed reservations across local worker processes."""
        if self._path is None:
            yield
            return

        import fcntl

        lock_path = self._path + ".lock"
        with open(lock_path, "a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    def _reload_locked(self) -> bool:
        """Refresh state from disk while holding thread and process locks."""
        if self._path is None:
            return True
        try:
            import json as _json

            with open(self._path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            self._store = {k: int(v) for k, v in data.items()}
            self._ready = True
            self._load_error = None
            return True
        except FileNotFoundError:
            self._store = {}
            self._ready = True
            self._load_error = None
            return True
        except Exception as exc:
            self._ready = False
            self._load_error = str(exc)
            return False

    def _save_locked(self) -> None:
        if self._path is None:
            return
        import json as _json

        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(self._store, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._path)

    def reserve_hit(
        self, fingerprint: str, budget: int
    ) -> Tuple[bool, int, str]:
        """Atomically reserve one hit before serving it.

        Returns ``(accepted, charged_hits, reason)``. Disk-backed instances use
        an advisory process lock and refresh the ledger while holding it, so
        local replicas sharing one ledger path draw from one cumulative budget.
        Any unavailable, corrupt, or failed persistence state rejects the hit.
        """
        if budget <= 0:
            raise ValueError("budget must be positive")
        with self._lock:
            if not self._available:
                return False, self._store.get(fingerprint, 0), "unavailable"
            if not self._ready:
                return False, self._store.get(fingerprint, 0), "recovery_incomplete"

            with self._process_lock():
                if not self._reload_locked():
                    return (
                        False,
                        self._store.get(fingerprint, 0),
                        "recovery_incomplete",
                    )
                current = self._store.get(fingerprint, 0)
                if current >= budget:
                    return False, current, "exhausted"

                self._store[fingerprint] = current + 1
                try:
                    self._save_locked()
                except Exception as exc:
                    self._store[fingerprint] = current
                    self._ready = False
                    self._load_error = str(exc)
                    return False, current, "persist_failed"
                return True, current + 1, "ok"

    def charged_hits(self, fingerprint: str) -> int:
        """Return cumulative hits already charged for *fingerprint*."""
        with self._lock:
            if self._path is not None and self._available and self._ready:
                with self._process_lock():
                    self._reload_locked()
            return self._store.get(fingerprint, 0)

    def add_hits(self, fingerprint: str, hits: int, persist: bool = False) -> int:
        """Add *hits* to the fingerprint's ledger entry and return new total.

        If *persist* is True the change is fsynced to disk immediately.
        """
        if hits < 0:
            raise ValueError("hits must be non-negative")
        with self._lock:
            new_total = self._store.get(fingerprint, 0) + hits
            self._store[fingerprint] = new_total
            if persist:
                with self._process_lock():
                    self._save_locked()
            return new_total

    def flush(self) -> None:
        """Sync all pending changes to the backing file (no-op if in-memory)."""
        with self._lock:
            if not self._available or not self._ready:
                return
            with self._process_lock():
                pending = dict(self._store)
                if not self._reload_locked():
                    return
                for fingerprint, hits in pending.items():
                    self._store[fingerprint] = max(
                        hits, self._store.get(fingerprint, 0)
                    )
                self._save_locked()

    def reset(self, epoch: Optional[int] = None) -> None:
        """Clear all ledger entries (call on policy-epoch rotation)."""
        with self._lock:
            with self._process_lock():
                self._store.clear()
                self._ready = True
                self._load_error = None
                self._save_locked()

    def snapshot(self) -> Mapping[str, object]:
        """Return an immutable copy of current ledger state."""
        with self._lock:
            if self._path is not None and self._available and self._ready:
                with self._process_lock():
                    self._reload_locked()
            return MappingProxyType(dict(self._store))


class PublicRegistry:
    """Thread-safe registry for HMAC-authorized exact public prefixes."""

    OK = "ok"
    INVALID_MAC = "invalid_mac"
    WRONG_MODEL = "wrong_model"
    WRONG_TOKENIZER = "wrong_tokenizer"
    WRONG_LENGTH = "wrong_length"
    WRONG_FINGERPRINT = "wrong_fingerprint"
    STALE_EPOCH = "stale_epoch"
    EXPIRED = "expired"
    REVOKED = "revoked"
    UNKNOWN_OBJECT = "unknown_object"
    OBJECT_MISMATCH = "object_mismatch"

    def __init__(self, operator_key: bytes, policy_epoch: int = 0) -> None:
        """Create a registry using a non-empty operator key."""

        if not isinstance(operator_key, bytes) or not operator_key:
            raise ValueError("operator_key must be non-empty bytes")
        if not isinstance(policy_epoch, int) or isinstance(policy_epoch, bool):
            raise ValueError("policy_epoch must be an integer")
        self._operator_key = operator_key
        self._policy_epoch = policy_epoch
        self._authorizations: Dict[str, PublicAuthorization] = {}
        self._revoked_ids: set[str] = set()
        self._lock = threading.RLock()

    @property
    def policy_epoch(self) -> int:
        """Return the currently accepted policy epoch."""

        with self._lock:
            return self._policy_epoch

    @staticmethod
    def _length_prefixed(value: bytes) -> bytes:
        return struct.pack(">Q", len(value)) + value

    @classmethod
    def _prefix_payload(
        cls,
        model_id: str,
        tokenizer_version: str,
        token_ids: Iterable[int],
    ) -> Tuple[bytes, Tuple[int, ...]]:
        tokens = tuple(token_ids)
        encoded_tokens: List[bytes] = []
        for token_id in tokens:
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                or token_id > 0xFFFFFFFFFFFFFFFF
            ):
                raise ValueError("token IDs must be unsigned 64-bit integers")
            encoded_tokens.append(struct.pack(">Q", token_id))
        model_bytes = model_id.encode("utf-8")
        tokenizer_bytes = tokenizer_version.encode("utf-8")
        payload = b"".join(
            (
                cls._length_prefixed(model_bytes),
                cls._length_prefixed(tokenizer_bytes),
                struct.pack(">Q", len(tokens)),
                b"".join(encoded_tokens),
            )
        )
        return payload, tokens

    @staticmethod
    def _authorization_payload(
        public_object_id: str,
        issuer: str,
        prefix_payload: bytes,
        policy_epoch: int,
        expires_at: float,
        revoked: bool,
    ) -> bytes:
        fields = (
            public_object_id.encode("utf-8"),
            issuer.encode("utf-8"),
            prefix_payload,
            struct.pack(">q", policy_epoch),
            struct.pack(">d", expires_at),
            b"\x01" if revoked else b"\x00",
        )
        return b"".join(
            struct.pack(">Q", len(field)) + field for field in fields
        )

    @classmethod
    def fingerprint(
        cls,
        model_id: str,
        tokenizer_version: str,
        token_ids: Iterable[int],
    ) -> str:
        """Return the SHA-256 fingerprint of an exact prefix identity."""

        payload, _ = cls._prefix_payload(
            model_id, tokenizer_version, token_ids
        )
        return hashlib.sha256(payload).hexdigest()

    def issue(
        self,
        public_object_id: str,
        issuer: str,
        model_id: str,
        tokenizer_version: str,
        token_ids: Iterable[int],
        expires_at: float,
        policy_epoch: Optional[int] = None,
        revoked: bool = False,
    ) -> PublicAuthorization:
        """Issue an authorization and install it unless already revoked."""

        if not public_object_id or not issuer or not model_id:
            raise ValueError("object ID, issuer, and model ID must be non-empty")
        payload, tokens = self._prefix_payload(
            model_id, tokenizer_version, token_ids
        )
        with self._lock:
            epoch = self._policy_epoch if policy_epoch is None else policy_epoch
            if epoch != self._policy_epoch:
                raise ValueError(self.STALE_EPOCH)
            if expires_at <= time.time():
                raise ValueError(self.EXPIRED)
            fingerprint = hashlib.sha256(payload).hexdigest()
            auth_payload = self._authorization_payload(
                public_object_id,
                issuer,
                payload,
                epoch,
                float(expires_at),
                revoked,
            )
            authorization = PublicAuthorization(
                public_object_id=public_object_id,
                issuer=issuer,
                model_id=model_id,
                tokenizer_version=tokenizer_version,
                prefix_token_length=len(tokens),
                prefix_fingerprint=fingerprint,
                policy_epoch=epoch,
                expires_at=float(expires_at),
                revoked=revoked,
                mac=hmac.new(
                    self._operator_key, auth_payload, hashlib.sha256
                ).hexdigest(),
            )
            if revoked:
                self._revoked_ids.add(public_object_id)
            else:
                self._authorizations[public_object_id] = authorization
                self._revoked_ids.discard(public_object_id)
            return authorization

    def _verify_locked(
        self,
        authorization: PublicAuthorization,
        model_id: str,
        tokenizer_version: str,
        token_ids: Iterable[int],
        now: Optional[float],
        require_installed: bool,
    ) -> VerificationResult:
        payload, tokens = self._prefix_payload(
            model_id, tokenizer_version, token_ids
        )
        fingerprint = hashlib.sha256(payload).hexdigest()
        auth_payload = self._authorization_payload(
            authorization.public_object_id,
            authorization.issuer,
            payload,
            authorization.policy_epoch,
            float(authorization.expires_at),
            authorization.revoked,
        )
        expected_mac = hmac.new(
            self._operator_key, auth_payload, hashlib.sha256
        ).hexdigest()
        mac_matches = hmac.compare_digest(expected_mac, authorization.mac)
        if authorization.model_id != model_id:
            return VerificationResult(False, self.WRONG_MODEL)
        if authorization.tokenizer_version != tokenizer_version:
            return VerificationResult(False, self.WRONG_TOKENIZER)
        if authorization.prefix_token_length != len(tokens):
            return VerificationResult(False, self.WRONG_LENGTH)
        if not hmac.compare_digest(
            authorization.prefix_fingerprint, fingerprint
        ):
            return VerificationResult(False, self.WRONG_FINGERPRINT)
        if not mac_matches:
            return VerificationResult(False, self.INVALID_MAC)
        if authorization.policy_epoch != self._policy_epoch:
            return VerificationResult(False, self.STALE_EPOCH)
        current_time = time.time() if now is None else now
        if authorization.expires_at <= current_time:
            return VerificationResult(False, self.EXPIRED)
        if authorization.revoked or (
            authorization.public_object_id in self._revoked_ids
        ):
            return VerificationResult(False, self.REVOKED)
        if require_installed:
            installed = self._authorizations.get(
                authorization.public_object_id
            )
            if installed is None:
                return VerificationResult(False, self.UNKNOWN_OBJECT)
            if installed != authorization:
                return VerificationResult(False, self.OBJECT_MISMATCH)
        return VerificationResult(True, self.OK)

    def verify(
        self,
        authorization: PublicAuthorization,
        model_id: str,
        tokenizer_version: str,
        token_ids: Iterable[int],
        now: Optional[float] = None,
    ) -> VerificationResult:
        """Verify an installed authorization and return a stable reason code."""

        with self._lock:
            return self._verify_locked(
                authorization,
                model_id,
                tokenizer_version,
                token_ids,
                now,
                True,
            )

    def install(
        self,
        authorization: PublicAuthorization,
        token_ids: Iterable[int],
        model_id: Optional[str] = None,
        tokenizer_version: Optional[str] = None,
        now: Optional[float] = None,
    ) -> VerificationResult:
        """Verify and atomically install an externally issued authorization."""

        checked_model = authorization.model_id if model_id is None else model_id
        checked_tokenizer = (
            authorization.tokenizer_version
            if tokenizer_version is None
            else tokenizer_version
        )
        with self._lock:
            result = self._verify_locked(
                authorization,
                checked_model,
                checked_tokenizer,
                token_ids,
                now,
                False,
            )
            if result.valid:
                self._authorizations[
                    authorization.public_object_id
                ] = authorization
            elif result.reason == self.REVOKED and authorization.revoked:
                # A correctly authenticated revocation is itself a control-plane
                # update.  Remember it so a previously issued valid manifest
                # cannot reinstall the object.
                self._revoked_ids.add(authorization.public_object_id)
                self._authorizations.pop(
                    authorization.public_object_id, None
                )
            return result

    def revoke(self, public_object_id: str) -> bool:
        """Revoke an object ID, including against concurrent installations."""

        with self._lock:
            existed = public_object_id in self._authorizations
            self._revoked_ids.add(public_object_id)
            return existed

    def is_revoked(self, public_object_id: str) -> bool:
        """Return whether the object is currently revoked."""
        with self._lock:
            return public_object_id in self._revoked_ids

    def reset(self, policy_epoch: Optional[int] = None) -> None:
        """Clear registry state and optionally replace the accepted epoch."""

        with self._lock:
            self._authorizations.clear()
            self._revoked_ids.clear()
            if policy_epoch is not None:
                if not isinstance(policy_epoch, int) or isinstance(
                    policy_epoch, bool
                ):
                    raise ValueError("policy_epoch must be an integer")
                self._policy_epoch = policy_epoch

    def snapshot(self) -> Mapping[str, object]:
        """Return an immutable point-in-time registry snapshot."""

        with self._lock:
            return MappingProxyType(
                {
                    "policy_epoch": self._policy_epoch,
                    "authorizations": MappingProxyType(
                        dict(self._authorizations)
                    ),
                    "revoked_object_ids": frozenset(self._revoked_ids),
                }
            )

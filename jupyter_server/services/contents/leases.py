"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import threading
import time
import typing as t
import uuid
from contextlib import contextmanager
from datetime import datetime

from jupyter_core.paths import jupyter_runtime_dir
from tornado.web import HTTPError
from traitlets import Float, Integer, Unicode
from traitlets.config.configurable import LoggingConfigurable

from jupyter_server import _tz as tz

# Recoverable conflict codes returned to callers in the 409 payload.
LEASE_REQUIRED = "lease_required"
LEASE_UNKNOWN = "lease_unknown"
LEASE_HELD = "lease_held"
LEASE_EXPIRED = "lease_expired"
LEASE_INVALIDATED = "lease_invalidated"
STALE_GENERATION = "stale_generation"
EXTERNAL_MODIFICATION = "external_modification"
DUPLICATE_REQUEST = "duplicate_request"

# Recovery hints handed back to clients so they can resolve conflicts
# without support staff comparing modification times.
RECOVERY_OPEN = "open"  # acquire a fresh lease (after reloading content)
RECOVERY_RELOAD = "reload"  # re-read the current content, then re-open a lease
RECOVERY_RETRY = "retry"  # wait for the current lease to expire, then retry
RECOVERY_TAKEOVER = "takeover"  # an administrator may take the lease over
RECOVERY_NEW_REQUEST_ID = "new_request_id"  # retry with a fresh request id

_STATE_VERSION = 1
_MAX_TRACKED_REQUESTS = 32
_MAX_INVALIDATIONS = 100

# Sentinel distinguishing "no fingerprint supplied" (skip the external
# modification check) from "fingerprint checked and the file is gone".
_UNSET: t.Any = object()


def _isoformat(timestamp: float) -> str:
    """项目内部接口说明。"""
    return tz.isoformat(datetime.fromtimestamp(timestamp, tz.UTC))


def supports_lease_kwarg(method: t.Any) -> bool:
    """项目内部接口说明。"""
    try:
        params = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD or p.name == "lease" for p in params)


class LeaseConflictError(HTTPError):
    """项目内部接口说明。"""

    def __init__(self, code: str, message: str, *, path: str, recovery: str, **details: t.Any):
        super().__init__(409, message)
        self.reason = code
        self.conflict: dict[str, t.Any] = {
            "code": code,
            "path": path,
            "recovery": recovery,
        }
        self.conflict.update(details)

    @classmethod
    def required(cls, path: str, op: str) -> LeaseConflictError:
        """项目内部接口说明。"""
        return cls(
            LEASE_REQUIRED,
            f"Operation {op!r} on {path!r} requires an active edit lease",
            path=path,
            operation=op,
            recovery=RECOVERY_OPEN,
        )

    @classmethod
    def unknown(cls, path: str, op: str) -> LeaseConflictError:
        """项目内部接口说明。"""
        return cls(
            LEASE_UNKNOWN,
            f"No edit lease is recorded for {path!r}; acquire a lease before {op!r}",
            path=path,
            operation=op,
            recovery=RECOVERY_OPEN,
        )

    @classmethod
    def held(cls, path: str, lease: dict[str, t.Any]) -> LeaseConflictError:
        """项目内部接口说明。"""
        return cls(
            LEASE_HELD,
            "{} is being edited by {!r} until {}".format(
                path, lease["holder"], _isoformat(lease["expires_at"])
            ),
            path=path,
            recovery=RECOVERY_RETRY,
            current_generation=lease["generation"],
            holder=lease["holder"],
            expires_at=_isoformat(lease["expires_at"]),
        )

    @classmethod
    def expired(cls, path: str, lease: dict[str, t.Any]) -> LeaseConflictError:
        """项目内部接口说明。"""
        return cls(
            LEASE_EXPIRED,
            "Edit lease for {} held by {!r} expired at {}".format(
                path, lease["holder"], _isoformat(lease["expires_at"])
            ),
            path=path,
            recovery=RECOVERY_OPEN,
            current_generation=lease["generation"],
            holder=lease["holder"],
            expires_at=_isoformat(lease["expires_at"]),
        )

    @classmethod
    def invalidated(cls, path: str, entry: dict[str, t.Any]) -> LeaseConflictError:
        """项目内部接口说明。"""
        message = "Edit lease for {} held by {!r} was ended ({})".format(
            path, entry["holder"], entry["ended_by"]
        )
        if entry.get("reason"):
            message += ": {}".format(entry["reason"])
        return cls(
            LEASE_INVALIDATED,
            message,
            path=path,
            recovery=RECOVERY_OPEN,
            current_generation=entry["generation"],
            holder=entry["holder"],
            ended_by=entry["ended_by"],
            ended_reason=entry.get("reason"),
            ended_at=_isoformat(entry["at"]),
        )

    @classmethod
    def stale(cls, path: str, lease: dict[str, t.Any], presented: t.Any) -> LeaseConflictError:
        """项目内部接口说明。"""
        return cls(
            STALE_GENERATION,
            "Stale fencing token {!r} for {}; current generation is {}".format(
                presented, path, lease["generation"]
            ),
            path=path,
            recovery=RECOVERY_RELOAD,
            expected_generation=presented,
            current_generation=lease["generation"],
            holder=lease["holder"],
            expires_at=_isoformat(lease["expires_at"]),
        )

    @classmethod
    def external_modification(cls, path: str, lease: dict[str, t.Any]) -> LeaseConflictError:
        """项目内部接口说明。"""
        return cls(
            EXTERNAL_MODIFICATION,
            f"{path} was modified on disk outside the current edit lease",
            path=path,
            recovery=RECOVERY_RELOAD,
            current_generation=lease["generation"],
            holder=lease["holder"],
            expires_at=_isoformat(lease["expires_at"]),
        )

    @classmethod
    def duplicate_request(
        cls, path: str, request_id: str, prior: dict[str, t.Any]
    ) -> LeaseConflictError:
        """项目内部接口说明。"""
        return cls(
            DUPLICATE_REQUEST,
            "Request id {!r} was already applied to {} as {!r}".format(
                request_id, path, prior["op"]
            ),
            path=path,
            recovery=RECOVERY_NEW_REQUEST_ID,
            request_id=request_id,
            prior_operation=prior["op"],
        )


class LeaseManager(LoggingConfigurable):
    """项目内部接口说明。"""

    lease_ttl = Float(
        300.0,
        config=True,
        help="Default edit lease duration in seconds.",
    )

    max_lease_ttl = Float(
        3600.0,
        config=True,
        help="Maximum edit lease duration in seconds; requested TTLs are clamped to this.",
    )

    lease_state_dir = Unicode(
        "",
        config=True,
        help="""Directory holding the persisted lease state file.

        If empty (default), a per-contents-root file is created inside the
        Jupyter runtime directory so that fencing tokens survive a server
        restart. Set this explicitly when the runtime directory is volatile.
        """,
    )

    def __init__(self, **kwargs: t.Any):
        super().__init__(**kwargs)
        self._lock = threading.RLock()
        self._loaded = False
        self._paths: dict[str, dict[str, t.Any]] = {}
        self._mono: dict[str, float] = {}
        self._path_locks: dict[str, threading.RLock] = {}
        self._path_alocks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # State persistence (survives service restarts)
    # ------------------------------------------------------------------

    def _root_dir(self) -> str:
        """项目内部接口说明。"""
        return str(getattr(self.parent, "root_dir", "") or os.getcwd())

    def _state_path(self) -> str:
        """项目内部接口说明。"""
        if self.lease_state_dir:
            state_dir = self.lease_state_dir
        else:
            state_dir = os.path.join(jupyter_runtime_dir(), "contents_leases")
        os.makedirs(state_dir, exist_ok=True)
        key = hashlib.sha256(os.path.abspath(self._root_dir()).encode("utf-8")).hexdigest()[:16]
        return os.path.join(state_dir, f"leases-{key}.json")

    def _ensure_loaded(self) -> None:
        """项目内部接口说明。"""
        if self._loaded:
            return
        self._loaded = True
        try:
            state_path = self._state_path()
        except OSError:
            # No usable state directory: fencing still works in-memory,
            # only restart durability is lost.
            self.log.warning("Lease state directory is not usable", exc_info=True)
            return
        try:
            with open(state_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as e:
            # A corrupt or unreadable state file must fail closed: starting
            # empty means old tokens are rejected as unknown, never accepted.
            self.log.error("Unable to load lease state, starting empty: %s", e)
            try:
                os.replace(state_path, state_path + ".corrupt")
            except OSError:
                pass
            return
        if not isinstance(data, dict) or data.get("version") != _STATE_VERSION:
            self.log.warning("Ignoring lease state with unsupported format")
            return
        paths = data.get("paths", {})
        if not isinstance(paths, dict):
            return
        self._paths = paths
        # Re-anchor expiry on this process's monotonic clock.  If the wall
        # clock jumped backwards, remaining time is capped at the lease TTL;
        # if it jumped forwards, the lease simply expires early.  Both
        # directions fail closed.
        now_wall = time.time()
        now_mono = time.monotonic()
        for path, rec in self._paths.items():
            lease = rec.get("lease")
            if not lease:
                continue
            remaining = lease.get("expires_at", 0.0) - now_wall
            ttl = lease.get("ttl", self.lease_ttl)
            if remaining <= 0:
                self._end_lease(path, rec, ended_by="expired")
            else:
                self._mono[path] = now_mono + min(remaining, ttl)

    def _persist(self) -> None:
        """项目内部接口说明。"""
        data = {"version": _STATE_VERSION, "paths": self._paths}
        try:
            state_path = self._state_path()
            tmp_path = state_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, state_path)
        except OSError:
            # Fencing still works in-memory; only restart durability degrades.
            self.log.warning("Unable to persist lease state", exc_info=True)

    # ------------------------------------------------------------------
    # Internal record helpers (called with self._lock held)
    # ------------------------------------------------------------------

    def _record(self, path: str) -> dict[str, t.Any]:
        """项目内部接口说明。"""
        rec = self._paths.get(path)
        if rec is None:
            rec = {
                "generation": 0,
                "lease": None,
                "invalidations": [],
                "requests": {},
            }
            self._paths[path] = rec
        rec.setdefault("invalidations", [])
        rec.setdefault("requests", {})
        return rec

    def _expired(self, path: str, rec: dict[str, t.Any]) -> bool:
        """项目内部接口说明。"""
        lease = rec.get("lease")
        if not lease:
            return False
        mono = self._mono.get(path)
        if mono is not None:
            return time.monotonic() >= mono
        expires_at: float = lease["expires_at"]
        return time.time() >= expires_at

    def _end_lease(
        self,
        path: str,
        rec: dict[str, t.Any],
        *,
        ended_by: str,
        reason: str | None = None,
        by: str | None = None,
    ) -> None:
        """项目内部接口说明。"""
        lease = rec.get("lease")
        if not lease:
            return
        rec["invalidations"].append(
            {
                "lease_id": lease["lease_id"],
                "generation": lease["generation"],
                "holder": lease["holder"],
                "ended_by": ended_by,
                "reason": reason,
                "by": by,
                "at": time.time(),
            }
        )
        del rec["invalidations"][:-_MAX_INVALIDATIONS]
        rec["lease"] = None
        self._mono.pop(path, None)

    def _find_invalidation(self, rec: dict[str, t.Any], lease_id: str) -> dict[str, t.Any] | None:
        """项目内部接口说明。"""
        invalidations: list[dict[str, t.Any]] = rec.get("invalidations", [])
        found: dict[str, t.Any] | None = None
        for entry in reversed(invalidations):
            if entry["lease_id"] == lease_id:
                found = entry
                break
        return found

    def _clamp_ttl(self, ttl: float | None) -> float:
        """项目内部接口说明。"""
        if ttl is None:
            return self.lease_ttl
        try:
            ttl = float(ttl)
        except (TypeError, ValueError):
            raise HTTPError(400, "Lease ttl must be a number of seconds") from None
        if ttl <= 0:
            raise HTTPError(400, "Lease ttl must be positive")
        return min(ttl, self.max_lease_ttl)

    def _public_lease(self, path: str, rec: dict[str, t.Any]) -> dict[str, t.Any]:
        """项目内部接口说明。"""
        lease = rec["lease"]
        return {
            "path": path,
            "lease_id": lease["lease_id"],
            "generation": lease["generation"],
            "holder": lease["holder"],
            "created_at": _isoformat(lease["created_at"]),
            "expires_at": _isoformat(lease["expires_at"]),
            "ttl": lease["ttl"],
        }

    # ------------------------------------------------------------------
    # Lease lifecycle
    # ------------------------------------------------------------------

    def acquire(
        self,
        path: str,
        holder: str,
        ttl: float | None = None,
        file_state: dict[str, t.Any] | None = None,
    ) -> tuple[dict[str, t.Any], bool]:
        """项目内部接口说明。"""
        if not holder:
            raise HTTPError(400, "A lease holder is required")
        ttl = self._clamp_ttl(ttl)
        with self._lock:
            self._ensure_loaded()
            rec = self._paths.get(path)
            if rec and rec.get("lease"):
                lease = rec["lease"]
                if not self._expired(path, rec):
                    if lease["holder"] == holder:
                        # A reconnecting client re-acquires its own lease,
                        # unless the file changed on disk in the meantime:
                        # then it must reload, release, and acquire afresh
                        # rather than silently re-baselining the fingerprint.
                        if (
                            file_state is not None or lease.get("file_state") is not None
                        ) and lease.get("file_state") != file_state:
                            raise LeaseConflictError.external_modification(path, lease)
                        return self._public_lease(path, rec), False
                    raise LeaseConflictError.held(path, lease)
                self._end_lease(path, rec, ended_by="expired")
            rec = self._record(path)
            rec["generation"] += 1
            now = time.time()
            rec["lease"] = {
                "lease_id": uuid.uuid4().hex,
                "generation": rec["generation"],
                "holder": holder,
                "created_at": now,
                "expires_at": now + ttl,
                "ttl": ttl,
                "file_state": file_state,
            }
            self._mono[path] = time.monotonic() + ttl
            self._persist()
            return self._public_lease(path, rec), True

    def renew(
        self,
        path: str,
        lease_id: str,
        generation: int,
        ttl: float | None = None,
    ) -> dict[str, t.Any]:
        """项目内部接口说明。"""
        ttl = self._clamp_ttl(ttl)
        with self._lock:
            self._ensure_loaded()
            self._validate_locked(path, lease_id, generation, op="renew")
            rec = self._paths[path]
            lease = rec["lease"]
            now = time.time()
            lease["expires_at"] = now + ttl
            lease["ttl"] = ttl
            self._mono[path] = time.monotonic() + ttl
            self._persist()
            return self._public_lease(path, rec)

    def release(self, path: str, lease_id: str, generation: int) -> None:
        """项目内部接口说明。"""
        with self._lock:
            self._ensure_loaded()
            self._validate_locked(path, lease_id, generation, op="release")
            rec = self._paths[path]
            self._end_lease(path, rec, ended_by="released")
            self._persist()

    def takeover(
        self,
        path: str,
        holder: str,
        reason: str,
        ttl: float | None = None,
        by: str | None = None,
        file_state: dict[str, t.Any] | None = None,
    ) -> dict[str, t.Any]:
        """项目内部接口说明。"""
        if not holder:
            raise HTTPError(400, "A lease holder is required")
        if not reason or not str(reason).strip():
            raise HTTPError(400, "A takeover reason is required")
        ttl = self._clamp_ttl(ttl)
        with self._lock:
            self._ensure_loaded()
            rec = self._record(path)
            if rec.get("lease"):
                self._end_lease(path, rec, ended_by="takeover", reason=reason, by=by)
                self.log.warning(
                    "Lease on %s held by %r was taken over by %r: %s",
                    path,
                    rec["invalidations"][-1]["holder"],
                    by or holder,
                    reason,
                )
            rec["generation"] += 1
            now = time.time()
            rec["lease"] = {
                "lease_id": uuid.uuid4().hex,
                "generation": rec["generation"],
                "holder": holder,
                "created_at": now,
                "expires_at": now + ttl,
                "ttl": ttl,
                "file_state": file_state,
            }
            self._mono[path] = time.monotonic() + ttl
            self._persist()
            return self._public_lease(path, rec)

    def inspect(self, path: str) -> dict[str, t.Any]:
        """项目内部接口说明。"""
        with self._lock:
            self._ensure_loaded()
            rec = self._paths.get(path)
            view: dict[str, t.Any] = {
                "path": path,
                "active": False,
                "generation": rec["generation"] if rec else 0,
                "holder": None,
                "created_at": None,
                "expires_at": None,
                "invalidations": [],
            }
            if rec:
                lease = rec.get("lease")
                if lease and not self._expired(path, rec):
                    view.update(
                        active=True,
                        holder=lease["holder"],
                        created_at=_isoformat(lease["created_at"]),
                        expires_at=_isoformat(lease["expires_at"]),
                    )
                view["invalidations"] = [
                    {
                        "generation": entry["generation"],
                        "holder": entry["holder"],
                        "ended_by": entry["ended_by"],
                        "reason": entry.get("reason"),
                        "by": entry.get("by"),
                        "at": _isoformat(entry["at"]),
                    }
                    for entry in rec.get("invalidations", [])
                ]
            return view

    # ------------------------------------------------------------------
    # Fencing-token validation
    # ------------------------------------------------------------------

    def _validate_locked(
        self,
        path: str,
        lease_id: str,
        generation: int,
        op: str,
        file_state: t.Any = _UNSET,
    ) -> None:
        """项目内部接口说明。"""
        rec = self._paths.get(path)
        if rec is None:
            raise LeaseConflictError.unknown(path, op)
        lease = rec.get("lease")
        if lease is None:
            entry = self._find_invalidation(rec, lease_id)
            if entry is not None:
                raise LeaseConflictError.invalidated(path, entry)
            raise LeaseConflictError.unknown(path, op)
        if lease["lease_id"] != lease_id:
            entry = self._find_invalidation(rec, lease_id)
            if entry is not None:
                raise LeaseConflictError.invalidated(path, entry)
            raise LeaseConflictError.held(path, lease)
        if generation != lease["generation"]:
            raise LeaseConflictError.stale(path, lease, generation)
        if self._expired(path, rec):
            raise LeaseConflictError.expired(path, lease)
        if file_state is not _UNSET:
            expected = lease.get("file_state")
            if expected != file_state:
                raise LeaseConflictError.external_modification(path, lease)

    def validate(
        self,
        path: str,
        lease_id: str,
        generation: int,
        op: str,
        file_state: t.Any = _UNSET,
    ) -> None:
        """项目内部接口说明。"""
        with self._lock:
            self._ensure_loaded()
            self._validate_locked(path, lease_id, generation, op, file_state=file_state)

    def note_file_state(self, path: str, file_state: dict[str, t.Any] | None) -> None:
        """项目内部接口说明。"""
        if file_state is None:
            return
        with self._lock:
            self._ensure_loaded()
            rec = self._paths.get(path)
            if rec and rec.get("lease"):
                rec["lease"]["file_state"] = file_state
                self._persist()

    # ------------------------------------------------------------------
    # Lease bookkeeping across renames and deletes
    # ------------------------------------------------------------------

    def validate_transfer(self, old_path: str, new_path: str) -> None:
        """项目内部接口说明。"""
        if old_path == new_path:
            return
        with self._lock:
            self._ensure_loaded()
            rec = self._paths.get(new_path)
            if rec and rec.get("lease") and not self._expired(new_path, rec):
                raise LeaseConflictError.held(new_path, rec["lease"])

    def transfer(self, old_path: str, new_path: str) -> None:
        """项目内部接口说明。"""
        if old_path == new_path:
            return
        with self._lock:
            self._ensure_loaded()
            rec = self._paths.get(old_path)
            if rec is None:
                return
            self.validate_transfer(old_path, new_path)
            # Keep a tombstone at the old path so that replayed requests and
            # stale tokens still resolve instead of looking like a fresh path.
            self._paths[old_path] = {
                "generation": rec["generation"],
                "lease": None,
                "invalidations": rec["invalidations"],
                "requests": rec["requests"],
                "moved_to": new_path,
            }
            self._paths[new_path] = rec
            mono = self._mono.pop(old_path, None)
            if mono is not None:
                self._mono[new_path] = mono
            self._persist()

    def void(self, path: str, ended_by: str = "delete") -> None:
        """项目内部接口说明。"""
        with self._lock:
            self._ensure_loaded()
            rec = self._paths.get(path)
            if rec is None:
                return
            self._end_lease(path, rec, ended_by=ended_by)
            self._persist()

    # ------------------------------------------------------------------
    # Idempotent request replay (duplicate client submissions)
    # ------------------------------------------------------------------

    def lookup_request(self, path: str, request_id: str) -> dict[str, t.Any] | None:
        """项目内部接口说明。"""
        with self._lock:
            self._ensure_loaded()
            rec = self._paths.get(path)
            if not rec:
                return None
            requests: dict[str, dict[str, t.Any]] = rec.get("requests", {})
            return requests.get(request_id)

    def record_request(self, path: str, request_id: str, summary: dict[str, t.Any]) -> None:
        """项目内部接口说明。"""
        with self._lock:
            self._ensure_loaded()
            rec = self._record(path)
            requests = rec.setdefault("requests", {})
            summary = dict(summary)
            summary["at"] = time.time()
            requests[request_id] = summary
            while len(requests) > _MAX_TRACKED_REQUESTS:
                requests.pop(next(iter(requests)))
            self._persist()

    # ------------------------------------------------------------------
    # Per-path write serialization
    # ------------------------------------------------------------------

    @contextmanager
    def guard(self, *paths: str) -> t.Iterator[None]:
        """项目内部接口说明。"""
        locks = [
            self._path_locks.setdefault(path, threading.RLock()) for path in sorted(set(paths))
        ]
        for lock in locks:
            lock.acquire()
        try:
            yield
        finally:
            for lock in reversed(locks):
                lock.release()

    async def _async_guard_acquire(self, paths: tuple[str, ...]) -> list[asyncio.Lock]:
        """项目内部接口说明。"""
        locks = [self._path_alocks.setdefault(path, asyncio.Lock()) for path in paths]
        for lock in locks:
            await lock.acquire()
        return locks

    def aguard(self, *paths: str) -> t.AsyncContextManager[None]:
        """项目内部接口说明。"""
        return _AsyncPathGuard(self, tuple(sorted(set(paths))))


class _AsyncPathGuard:
    """项目内部接口说明。"""

    def __init__(self, manager: LeaseManager, paths: tuple[str, ...]):
        self._manager = manager
        self._paths = paths
        self._locks: list[asyncio.Lock] = []

    async def __aenter__(self) -> None:
        self._locks = await self._manager._async_guard_acquire(self._paths)

    async def __aexit__(self, *exc_info: object) -> None:
        for lock in reversed(self._locks):
            lock.release()

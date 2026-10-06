"""Fencing-token edit leases for the contents service.

A lease guards one document while a client has it open for editing.  It
combines two well-known optimistic-concurrency primitives:

* a **fencing token**: an opaque, unguessable secret returned when the lease
  is opened.  The token must be presented on every mutating request (save,
  rename, delete, ...).  When a lease is superseded -- because another client
  opened the document, or an administrator took it over -- the old token is
  invalidated *permanently* and can never be used again, even after a
  reconnect;
* a **generation**: a monotonically increasing counter shared by every lease
  ever issued for a path.  "Open, renew, save, rename and delete all verify
  the same generation" -- a request carrying a generation that is not the
  current one is rejected with a recoverable conflict instead of silently
  overwriting a newer version.

The store is deliberately in-memory.  That is sufficient to protect against
the documented threats (duplicate submissions, reconnects of stale clients,
administrator takeover) and it means state is rebuilt deterministically when
the service restarts: *every* issued token dies on restart, so no stale
client can fence-write after a restart -- callers receive a conflict and must
re-open the document.  Expiry uses a monotonic clock plus a configurable
grace period so ordinary client clock drift never causes a lease to be
considered expired prematurely.
"""

# Copyright (c) Jupyter Development Team.
from __future__ import annotations

import secrets
import time
import typing as t
from dataclasses import dataclass, field
from enum import Enum


class LeaseConflictReason(str, Enum):
    """Machine-readable reasons for a rejected lease operation.

    The string values are part of the API payload returned to callers so
    that clients can branch on them to recover (re-open, merge, refresh).
    """

    #: No matching lease exists (never opened, or service restarted).
    not_found = "lease_not_found"
    #: The presented fencing token was valid once, but has been superseded.
    #: It is dead forever; retrying with it can never succeed.
    stale_token = "stale_token"
    #: A different, live token currently holds the lease.
    held_by_other = "lease_held_by_other"
    #: The presented generation does not match the current generation.
    generation_mismatch = "generation_mismatch"
    #: The lease has passed its expiry time plus the clock-skew grace.
    expired = "lease_expired"
    #: A lease for the path is already open by this (or another) client.
    already_open = "lease_already_open"
    #: An administrative takeover without a recorded reason was attempted.
    reason_required = "takeover_reason_required"
    #: The operation is not allowed for the presented lease state
    #: (e.g. a checkpoint trying to extend write permission).
    not_permitted = "operation_not_permitted"
    #: The file on disk changed underneath the lease (external writer or a
    #: restore); the save must not silently overwrite it.
    external_modified = "external_modification"
    #: The same write was already committed under this lease; a re-submission
    #: with different content at the same version is a client bug, not a
    #: reason to overwrite.
    duplicate_write = "duplicate_write"


class LeaseError(Exception):
    """Raised when a lease check fails.

    Carries a structured :class:`LeaseConflict` so the HTTP layer (or any
    other caller) can turn the failure into a recoverable conflict response
    without parsing message strings.
    """

    def __init__(self, conflict: LeaseConflict):
        self.conflict = conflict
        super().__init__(conflict.message)


@dataclass
class LeaseConflict:
    """Structured, recoverable information about a rejected lease op.

    ``recoverable`` tells the client whether retrying the same request could
    ever succeed.  ``stale_token`` conflicts are *not* recoverable with the
    same credentials: the client must re-open the document and merge its
    edits against the current version.
    """

    reason: LeaseConflictReason
    path: str
    message: str
    expected_generation: int | None = None
    actual_generation: int | None = None
    held_by: str | None = None
    recoverable: bool = True
    retry_after: float | None = None

    def to_dict(self) -> dict[str, t.Any]:
        data: dict[str, t.Any] = {
            "error": "lease_conflict",
            "reason": self.reason.value,
            "message": self.message,
            "path": self.path,
            "recoverable": self.recoverable,
        }
        if self.expected_generation is not None:
            data["expected_generation"] = self.expected_generation
        if self.actual_generation is not None:
            data["current_generation"] = self.actual_generation
        if self.held_by is not None:
            data["held_by"] = self.held_by
        if self.retry_after is not None:
            data["retry_after"] = self.retry_after
        return data


@dataclass(frozen=True)
class LeaseCredentials:
    """Proof of a lease, presented by a client on a mutating request.

    ``token`` is the fencing token handed out at open/renew/takeover;
    ``generation`` is the generation the client edited against;
    ``lease_version`` optionally fences duplicate submissions against the
    exact write version (idempotency for retried saves).
    """

    token: str | None = None
    generation: int | None = None
    lease_version: int | None = None

    @classmethod
    def from_parts(
        cls,
        token: str | None,
        generation: int | None = None,
        lease_version: int | None = None,
    ) -> LeaseCredentials | None:
        """Build credentials, returning ``None`` when no token was supplied."""
        if token is None:
            return None
        return cls(token=token, generation=generation, lease_version=lease_version)


@dataclass
class LeaseRecord:
    """A single outstanding lease."""

    token: str
    generation: int
    owner: str
    issued_at: float
    expires_at: float
    #: Monotonic version bumped on every successful write made *under* this
    #: lease.  Checkpoints may record it, but creating one never bumps it.
    lease_version: int = 0
    #: Free-form client label (e.g. session id / hostname), purely advisory.
    owner_label: str | None = None
    #: Set when the lease ended through administrative takeover.
    takeover_reason: str | None = None
    #: Last renewal time (monotonic), purely informational.
    renewed_at: float | None = None
    #: Opaque, manager-supplied fingerprint of the file when the lease
    #: opened / after the last lease-authorised write.  Used to detect
    #: modifications made outside the API (external editors, restores);
    #: ``None`` means the manager cannot fingerprint the storage.
    baseline_key: t.Any = None
    #: Hash of the content written by the most recent lease-authorised
    #: write, for duplicate-submission detection.
    last_write_hash: str | None = None

    def is_live(self, now: float) -> bool:
        """项目内部接口说明。"""
        return now < self.expires_at


@dataclass
class _Tombstone:
    """A permanently dead token.

    Tombstones remember the token and its generation forever (bounded by the
    number of distinct documents ever edited in a process) so that a
    reconnecting stale client gets ``stale_token`` rather than silently
    acquiring a fresh lease for the same credentials.
    """

    token: str
    generation: int
    owner: str
    ended_at: float
    takeover_reason: str | None = None


@dataclass
class _LeaseEntry:
    """All lease state for one path."""

    next_generation: int = 1
    current: LeaseRecord | None = None
    history: list[_Tombstone] = field(default_factory=list)

    @property
    def generation(self) -> int:
        """The latest generation handed out for this path (live or dead)."""
        return self.next_generation - 1


def new_token() -> str:
    """Generate an opaque fencing token (URL-safe, 256 bits of entropy)."""
    return secrets.token_urlsafe(32)


class LeaseStore:
    """Process-wide store of edit leases.

    Synchronous and :class:`AsyncFileContentsManager` style callers share one
    store; an :class:`asyncio.Lock` serialises mutating operations so the
    check-then-act sequences (open/renew/save checks) are atomic with respect
    to each other.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        clock_skew_grace_seconds: float = 30.0,
        max_history_per_path: int = 100,
    ):
        if ttl_seconds <= 0:
            msg = "ttl_seconds must be positive"
            raise ValueError(msg)
        self.ttl_seconds = float(ttl_seconds)
        # A lease is only treated as expired this many seconds after its
        # nominal deadline, absorbing modest clock differences between the
        # client's renewal timer and the server clock.
        self.clock_skew_grace_seconds = float(clock_skew_grace_seconds)
        self.max_history_per_path = max_history_per_path
        self._entries: dict[str, _LeaseEntry] = {}

    # -- helpers -----------------------------------------------------------

    def _now(self) -> float:
        return time.monotonic()

    def _entry(self, path: str) -> _LeaseEntry:
        return self._entries.setdefault(path, _LeaseEntry())

    def _expire_if_due(self, entry: _LeaseEntry, path: str, now: float) -> None:
        """Move the current lease to history once ttl plus grace has elapsed.

        The lease is given ``clock_skew_grace_seconds`` of slack *past* its
        nominal deadline: only when ``now >= expires_at + grace`` is it
        considered dead, so a renewal timer running slightly behind the
        server clock cannot lose a live lease.
        """
        cur = entry.current
        if cur is not None and now >= cur.expires_at + self.clock_skew_grace_seconds:
            entry.history.append(
                _Tombstone(
                    token=cur.token,
                    generation=cur.generation,
                    owner=cur.owner,
                    ended_at=now,
                )
            )
            entry.current = None
            self._trim(entry)

    def _trim(self, entry: _LeaseEntry) -> None:
        if len(entry.history) > self.max_history_per_path:
            del entry.history[: -self.max_history_per_path]

    def _find_tombstone(self, entry: _LeaseEntry, token: str) -> _Tombstone | None:
        for tomb in entry.history:
            if secrets.compare_digest(tomb.token, token):
                return tomb
        return None

    # -- operations --------------------------------------------------------

    def acquire(
        self,
        path: str,
        owner: str,
        *,
        known_generation: int | None = None,
        owner_label: str | None = None,
    ) -> LeaseRecord:
        """Open a lease on ``path`` (a.k.a. "open for editing").

        If the caller already believes it knows the current generation
        (e.g. it fetched the document model and saw ``lease.generation``),
        passing ``known_generation`` turns the open into a compare-and-set:
        opening a document that was replaced in the meantime fails with a
        generation conflict instead of quietly bumping the writer.
        """
        now = self._now()
        entry = self._entry(path)
        self._expire_if_due(entry, path, now)

        if known_generation is not None and known_generation != entry.generation:
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.generation_mismatch,
                    path=path,
                    message=(
                        f"Document generation changed: client knew {known_generation}, "
                        f"current is {entry.generation}"
                    ),
                    expected_generation=known_generation,
                    actual_generation=entry.generation,
                    held_by=entry.current.owner if entry.current else None,
                )
            )

        if entry.current is not None:
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.held_by_other,
                    path=path,
                    message=f"Document is already open for editing by {entry.current.owner!r}",
                    actual_generation=entry.generation,
                    held_by=entry.current.owner,
                    retry_after=max(0.0, entry.current.expires_at - now),
                )
            )

        generation = entry.next_generation
        entry.next_generation += 1
        record = LeaseRecord(
            token=new_token(),
            generation=generation,
            owner=owner,
            issued_at=now,
            expires_at=now + self.ttl_seconds,
            owner_label=owner_label,
        )
        entry.current = record
        return record

    def renew(
        self,
        path: str,
        token: str,
        *,
        expected_generation: int | None = None,
    ) -> LeaseRecord:
        """Extend the TTL of a live lease.

        Renewal never changes the generation or the token: a reconnecting
        client keeps fencing with the same values.  A dead token stays dead.
        """
        now = self._now()
        entry = self._entry(path)
        self._expire_if_due(entry, path, now)

        dead = self._find_tombstone(entry, token)
        if dead is not None:
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.stale_token,
                    path=path,
                    message="Edit token has been superseded or revoked; reopen the document",
                    expected_generation=expected_generation,
                    actual_generation=entry.generation,
                    held_by=entry.current.owner if entry.current else None,
                    recoverable=False,
                )
            )

        cur = entry.current
        if cur is None or not secrets.compare_digest(cur.token, token):
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.not_found,
                    path=path,
                    message="No active edit lease for this token; reopen the document",
                    expected_generation=expected_generation,
                    actual_generation=entry.generation,
                )
            )

        if expected_generation is not None and expected_generation != cur.generation:
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.generation_mismatch,
                    path=path,
                    message=(
                        f"Generation mismatch on renew: expected {expected_generation}, "
                        f"current is {cur.generation}"
                    ),
                    expected_generation=expected_generation,
                    actual_generation=entry.generation,
                    held_by=cur.owner,
                )
            )

        cur.expires_at = now + self.ttl_seconds
        cur.renewed_at = now
        return cur

    def check(
        self,
        path: str,
        token: str | None,
        *,
        expected_generation: int | None = None,
        permit_when_absent: bool = False,
    ) -> LeaseRecord | None:
        """Validate a token + generation pair for a mutating operation.

        Returns the live :class:`LeaseRecord` on success.

        ``permit_when_absent`` preserves the legacy, lease-free API: when no
        token is presented and no lease exists on the path, the operation is
        allowed (contents clients that never adopted leases keep working).

        The important asymmetry: once *any* token is presented it must be a
        live token of the current generation, and if a lease is open, an
        operation without the token is rejected -- so external tooling can
        never silently clobber a document that is being edited.
        """
        now = self._now()
        entry = self._entries.get(path)
        presented = token is not None

        if entry is None:
            # Nothing was ever opened here (or the process restarted).
            if not presented:
                # Generation-only fencing is still meaningful even without a
                # lease record: generation 0 means "never edited".
                if permit_when_absent and expected_generation not in (None, 0):
                    raise LeaseError(
                        LeaseConflict(
                            reason=LeaseConflictReason.generation_mismatch,
                            path=path,
                            message=(
                                f"Generation mismatch: expected {expected_generation}, current is 0"
                            ),
                            expected_generation=expected_generation,
                            actual_generation=0,
                        )
                    )
                return None
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.not_found,
                    path=path,
                    message="No active edit lease for this token; reopen the document",
                    expected_generation=expected_generation,
                    actual_generation=0,
                )
            )

        self._expire_if_due(entry, path, now)

        dead = self._find_tombstone(entry, token) if token is not None else None
        if dead is not None:
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.stale_token,
                    path=path,
                    message=(
                        "Edit token has been superseded"
                        + (
                            f": {dead.takeover_reason!r}"
                            if dead.takeover_reason
                            else "; the document was opened by a newer client or the service restarted"
                        )
                    ),
                    expected_generation=expected_generation,
                    actual_generation=entry.generation,
                    held_by=entry.current.owner if entry.current else None,
                    recoverable=False,
                )
            )

        cur = entry.current

        if not presented:
            if cur is not None:
                raise LeaseError(
                    LeaseConflict(
                        reason=LeaseConflictReason.held_by_other,
                        path=path,
                        message=(
                            "Document is open for editing; the edit token is required "
                            "for this change"
                        ),
                        expected_generation=expected_generation,
                        actual_generation=entry.generation,
                        held_by=cur.owner,
                        retry_after=max(0.0, cur.expires_at - now),
                    )
                )
            if expected_generation is not None and permit_when_absent:
                # A client fencing purely with a generation (no open lease)
                # is still protected against newer versions.
                if expected_generation != entry.generation:
                    raise LeaseError(
                        LeaseConflict(
                            reason=LeaseConflictReason.generation_mismatch,
                            path=path,
                            message=(
                                f"Generation mismatch: expected {expected_generation}, "
                                f"current is {entry.generation}"
                            ),
                            expected_generation=expected_generation,
                            actual_generation=entry.generation,
                        )
                    )
            return None

        if cur is None or not secrets.compare_digest(cur.token, token or ""):
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.not_found,
                    path=path,
                    message="No active edit lease for this token; reopen the document",
                    expected_generation=expected_generation,
                    actual_generation=entry.generation,
                )
            )

        if expected_generation is not None and expected_generation != cur.generation:
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.generation_mismatch,
                    path=path,
                    message=(
                        f"Generation mismatch: client edited generation "
                        f"{expected_generation}, current is {cur.generation}"
                    ),
                    expected_generation=expected_generation,
                    actual_generation=entry.generation,
                    held_by=cur.owner,
                )
            )

        return cur

    def note_write(self, path: str, token: str) -> int:
        """Record a successful write under a lease; return the new version.

        Bumped by saves/renames/deletes only.  Checkpoint creation explicitly
        does not call this: a checkpoint can be *associated* with a version
        but must never extend write permission.
        """
        now = self._now()
        entry = self._entry(path)
        self._expire_if_due(entry, path, now)
        cur = entry.current
        if cur is None or not secrets.compare_digest(cur.token, token):
            # Should be unreachable after check(), but fail closed regardless.
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.not_found,
                    path=path,
                    message="Lease vanished before write could be recorded",
                    actual_generation=entry.generation,
                )
            )
        cur.lease_version += 1
        return cur.lease_version

    def set_baseline(self, path: str, token: str, key: t.Any) -> None:
        """Remember the file fingerprint captured when the lease opened.

        The key is opaque to the store; managers use (mtime_ns, size) or a
        content hash.  It is the reference against which later writes detect
        external modification of the file.
        """
        now = self._now()
        entry = self._entry(path)
        self._expire_if_due(entry, path, now)
        cur = entry.current
        if cur is None or not secrets.compare_digest(cur.token, token):
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.not_found,
                    path=path,
                    message="Cannot set baseline without an active lease",
                    actual_generation=entry.generation,
                )
            )
        cur.baseline_key = key

    def prepare_write(
        self,
        path: str,
        creds: LeaseCredentials,
        *,
        current_key: t.Any = None,
        content_hash: str | None = None,
    ) -> tuple[LeaseRecord, bool]:
        """Authorise (or reject) a content write before it touches storage.

        Returns ``(record, already_committed)``; when ``already_committed``
        is true the exact same content already produced the current write
        version, so the caller should answer the duplicate request with the
        existing model instead of writing again (idempotent retry).

        Rejects with:

        * ``generation_mismatch`` / ``stale_token`` -- fencing failure;
        * ``external_modification`` -- the file fingerprint differs from the
          baseline the lease last knew (edited out-of-band or restored from a
          checkpoint under a different write);
        * ``duplicate_write`` -- the client's expected write version is stale
          and the new content differs from what was committed there.
        """
        record = self.check(
            path,
            creds.token,
            expected_generation=creds.generation,
        )
        assert record is not None  # a token was presented; check() guarantees

        # Duplicate / reordered submission: the client fences against the
        # write version it last observed.
        if creds.lease_version is not None and creds.lease_version != record.lease_version:
            if content_hash is not None and record.last_write_hash == content_hash:
                return record, True
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.duplicate_write,
                    path=path,
                    message=(
                        f"Write version conflict: client expected version "
                        f"{creds.lease_version}, current is {record.lease_version}"
                    ),
                    expected_generation=creds.generation,
                    actual_generation=record.generation,
                    held_by=record.owner,
                )
            )

        # A naive retry without a version but with byte-identical content is
        # also an idempotent no-op.
        if (
            creds.lease_version is None
            and record.lease_version > 0
            and content_hash is not None
            and record.last_write_hash == content_hash
        ):
            return record, True

        # Out-of-band modification detection.
        if record.baseline_key is not None and current_key != record.baseline_key:
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.external_modified,
                    path=path,
                    message=(
                        "File was modified outside this edit lease since it was "
                        "opened; merge or re-open before saving"
                    ),
                    expected_generation=creds.generation,
                    actual_generation=record.generation,
                    held_by=record.owner,
                )
            )

        return record, False

    def commit_write(
        self,
        path: str,
        token: str,
        *,
        new_key: t.Any = None,
        content_hash: str | None = None,
    ) -> int:
        """Bookkeeping after a storage write has succeeded.

        Bumps the write version and advances the baseline fingerprint, so the
        next save compares against the file just written.  TTL is untouched
        (writing does not extend a lease -- only explicit renew does).
        """
        entry = self._entries.get(path)
        cur = entry.current if entry is not None else None
        if cur is None or not secrets.compare_digest(cur.token, token):
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.not_found,
                    path=path,
                    message="Lease vanished during write",
                    actual_generation=entry.generation if entry is not None else 0,
                )
            )
        cur.lease_version += 1
        cur.baseline_key = new_key
        if content_hash is not None:
            cur.last_write_hash = content_hash
        return cur.lease_version

    def release(self, path: str, token: str) -> None:
        """Close a lease voluntarily (client closed the document)."""
        now = self._now()
        entry = self._entry(path)
        self._expire_if_due(entry, path, now)

        dead = self._find_tombstone(entry, token)
        if dead is not None:
            # Idempotent: releasing an already-dead token is not an error so
            # duplicate close requests from the client are harmless.
            return

        cur = entry.current
        if cur is None or not secrets.compare_digest(cur.token, token):
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.not_found,
                    path=path,
                    message="No active edit lease for this token",
                    actual_generation=entry.generation,
                )
            )

        entry.history.append(_Tombstone(cur.token, cur.generation, cur.owner, now))
        entry.current = None
        self._trim(entry)

    def takeover(
        self,
        path: str,
        *,
        admin: str,
        reason: str,
        owner_label: str | None = None,
    ) -> LeaseRecord:
        """Administratively revoke the current lease and take it over.

        A non-empty ``reason`` is mandatory and is recorded against the
        tombstone so the evicted owner can see why their token died.  The old
        token is invalidated permanently.
        """
        if not reason or not reason.strip():
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.reason_required,
                    path=path,
                    message="Administrative takeover requires a non-empty reason",
                )
            )

        now = self._now()
        entry = self._entry(path)
        self._expire_if_due(entry, path, now)
        old = entry.current
        if old is not None:
            entry.history.append(
                _Tombstone(
                    token=old.token,
                    generation=old.generation,
                    owner=old.owner,
                    ended_at=now,
                    takeover_reason=reason.strip(),
                )
            )
        record = LeaseRecord(
            token=new_token(),
            generation=entry.next_generation,
            owner=admin,
            issued_at=now,
            expires_at=now + self.ttl_seconds,
            owner_label=owner_label or f"admin-takeover: {reason.strip()}",
            takeover_reason=reason.strip(),
        )
        entry.next_generation += 1
        entry.current = record
        self._trim(entry)
        return record

    def move(self, old_path: str, new_path: str) -> None:
        """Carry lease state over a rename.

        Called *after* a rename has been authorised (token + generation
        checked against ``old_path``): the live lease and tombstone history
        move to the new path, keeping old tokens dead and the generation
        continuous.  Leftover state at the destination (a deleted file that
        was once edited) is merged rather than clobbered; an *active* lease
        there is refused up-front by :meth:`check_rename_destination`.
        """
        if old_path == new_path:
            return
        entry = self._entries.pop(old_path, None)
        if entry is None:
            return
        dest = self._entries.get(new_path)
        if dest is None:
            self._entries[new_path] = entry
            return
        # Merge: dead tokens of both names stay dead, generation counter
        # never goes backwards.  If the vacated name has a higher generation
        # than the moving lease, the surviving lease is re-stamped with a new
        # generation (its token is unchanged) so two leases never share a
        # generation; the client learns the new generation from the response.
        entry.history.extend(dest.history)
        new_next = max(entry.next_generation, dest.next_generation)
        if entry.current is not None and entry.current.generation < new_next:
            entry.current.generation = new_next
            entry.next_generation = new_next + 1
        else:
            entry.next_generation = new_next
        self._trim(entry)
        self._entries[new_path] = entry

    def check_rename_destination(self, new_path: str) -> None:
        """Reject renaming onto a path that currently has a live lease."""
        entry = self.get(new_path)
        if entry is not None and entry.current is not None:
            cur = entry.current
            raise LeaseError(
                LeaseConflict(
                    reason=LeaseConflictReason.held_by_other,
                    path=new_path,
                    message=f"Destination is already open for editing by {cur.owner!r}",
                    actual_generation=entry.generation,
                    held_by=cur.owner,
                    retry_after=max(0.0, cur.expires_at - self._now()),
                )
            )

    def get(self, path: str) -> _LeaseEntry | None:
        """Return (lazily-expiring) state for inspection, or ``None``.

        Never creates an entry: listing a directory must not populate the
        store with empty per-file records.
        """
        entry = self._entries.get(path)
        if entry is None:
            return None
        self._expire_if_due(entry, path, self._now())
        return entry

    def current_generation(self, path: str) -> int:
        """项目内部接口说明。"""
        entry = self.get(path)
        return entry.generation if entry is not None else 0

    def model_for(self, path: str) -> dict[str, t.Any] | None:
        """Public lease info for a contents model, or ``None`` if no lease."""
        entry = self.get(path)
        if entry is None:
            return None
        cur = entry.current
        data: dict[str, t.Any] = {"generation": entry.generation}
        if cur is not None:
            data.update(
                token=cur.token,
                owner=cur.owner,
                lease_version=cur.lease_version,
                issued_at=cur.issued_at,
                expires_in=max(0.0, cur.expires_at - self._now()),
            )
            return data
        data["open"] = False
        return data

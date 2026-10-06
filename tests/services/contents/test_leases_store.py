"""Unit tests for the fencing-token edit lease store."""

import time

import pytest

from jupyter_server.services.contents.leases import (
    LeaseConflictReason,
    LeaseCredentials,
    LeaseError,
    LeaseStore,
)


@pytest.fixture
def store():
    return LeaseStore(ttl_seconds=10.0, clock_skew_grace_seconds=0.5)


def test_open_returns_token_and_generation(store):
    lease = store.acquire("nb.ipynb", "alice")
    assert lease.token
    assert lease.generation == 1
    assert store.check("nb.ipynb", lease.token, expected_generation=1) is lease


def test_second_open_is_rejected_without_taking_over(store):
    first = store.acquire("nb.ipynb", "alice")
    with pytest.raises(LeaseError) as exc:
        store.acquire("nb.ipynb", "bob")
    conflict = exc.value.conflict
    assert conflict.reason is LeaseConflictReason.held_by_other
    assert conflict.held_by == "alice"
    assert conflict.recoverable
    assert conflict.retry_after is not None
    # the original lease is untouched
    assert store.check("nb.ipynb", first.token, expected_generation=1) is first


def test_open_with_known_generation_is_compare_and_set(store):
    store.acquire("nb.ipynb", "alice")
    # A stale client that read generation 0 after another client opened gen 1
    # must fail to acquire instead of fencing the newer writer.
    with pytest.raises(LeaseError) as exc:
        store.acquire("nb.ipynb", "carol", known_generation=0)
    assert exc.value.conflict.reason is LeaseConflictReason.generation_mismatch
    assert exc.value.conflict.actual_generation == 1


def test_takeover_kills_old_token_permanently_and_requires_reason(store):
    first = store.acquire("nb.ipynb", "alice")

    with pytest.raises(LeaseError) as exc:
        store.takeover("nb.ipynb", admin="root", reason="   ")
    assert exc.value.conflict.reason is LeaseConflictReason.reason_required

    second = store.takeover("nb.ipynb", admin="root", reason="incident-42")
    assert second.generation == 2

    # The old token is dead forever: it can never fence again, even after the
    # new lease itself is released.
    with pytest.raises(LeaseError) as exc:
        store.check("nb.ipynb", first.token, expected_generation=1)
    dead = exc.value.conflict
    assert dead.reason is LeaseConflictReason.stale_token
    assert dead.recoverable is False

    store.release("nb.ipynb", second.token)
    with pytest.raises(LeaseError) as exc:
        store.check("nb.ipynb", first.token, expected_generation=1)
    assert exc.value.conflict.reason is LeaseConflictReason.stale_token


def test_renew_keeps_token_and_generation(store):
    lease = store.acquire("nb.ipynb", "alice")
    renewed = store.renew("nb.ipynb", lease.token, expected_generation=1)
    assert renewed.token == lease.token
    assert renewed.generation == 1
    assert renewed.expires_at >= lease.expires_at


def test_renew_with_wrong_generation_is_rejected(store):
    lease = store.acquire("nb.ipynb", "alice")
    store.takeover("nb.ipynb", admin="root", reason="x")
    with pytest.raises(LeaseError) as exc:
        store.renew("nb.ipynb", lease.token, expected_generation=1)
    assert exc.value.conflict.reason is LeaseConflictReason.stale_token


def test_release_is_idempotent(store):
    lease = store.acquire("nb.ipynb", "alice")
    store.release("nb.ipynb", lease.token)
    store.release("nb.ipynb", lease.token)  # duplicate close is harmless


def test_generation_mismatch_blocks_write(store):
    lease = store.acquire("nb.ipynb", "alice")
    with pytest.raises(LeaseError) as exc:
        store.check("nb.ipynb", lease.token, expected_generation=99)
    assert exc.value.conflict.reason is LeaseConflictReason.generation_mismatch


def test_tokenless_write_allowed_without_lease_but_blocked_with_one(store):
    # Legacy, lease-free API keeps working on untouched documents.
    assert store.check("free.ipynb", None) is None
    lease = store.acquire("open.ipynb", "alice")
    with pytest.raises(LeaseError) as exc:
        store.check("open.ipynb", None)
    assert exc.value.conflict.reason is LeaseConflictReason.held_by_other


def test_clock_skew_grace_keeps_lease_alive_past_deadline():
    # ttl 0.05s + 0.5s grace: a renewal timer slightly behind the server
    # clock must not lose a live lease.
    store = LeaseStore(ttl_seconds=0.05, clock_skew_grace_seconds=0.5)
    lease = store.acquire("nb.ipynb", "alice")
    time.sleep(0.1)
    renewed = store.renew("nb.ipynb", lease.token)
    assert renewed.token == lease.token


def test_expired_lease_cannot_be_renewed():
    store = LeaseStore(ttl_seconds=0.01, clock_skew_grace_seconds=0.0)
    lease = store.acquire("nb.ipynb", "alice")
    time.sleep(0.05)
    with pytest.raises(LeaseError) as exc:
        store.renew("nb.ipynb", lease.token)
    assert exc.value.conflict.reason is LeaseConflictReason.stale_token


def test_rename_carries_lease_and_keeps_tokens_dead(store):
    first = store.acquire("old.ipynb", "alice")
    store.takeover("old.ipynb", admin="root", reason="reassign")
    admin_token = store.get("old.ipynb").current.token
    store.move("old.ipynb", "new.ipynb")
    # live lease follows the document at a continuous generation
    assert store.check("new.ipynb", admin_token, expected_generation=2) is not None
    # old token stays dead at the new path
    with pytest.raises(LeaseError) as exc:
        store.check("new.ipynb", first.token, expected_generation=1)
    assert exc.value.conflict.reason is LeaseConflictReason.stale_token
    # nothing left at the old path
    assert store.check("old.ipynb", None) is None


def test_prepare_write_detects_duplicate_and_external_change(store):
    lease = store.acquire("nb.ipynb", "alice")
    store.set_baseline("nb.ipynb", lease.token, key=("mtime-1", 10))
    creds = LeaseCredentials(lease.token, generation=1, lease_version=0)

    # First write against the baseline is fine.
    record, already = store.prepare_write(
        "nb.ipynb", creds, current_key=("mtime-1", 10), content_hash="h1"
    )
    assert already is False
    store.commit_write("nb.ipynb", record.token, new_key=("mtime-2", 20), content_hash="h1")

    # Byte-identical resubmission is an idempotent no-op.
    same = LeaseCredentials(lease.token, generation=1, lease_version=0)
    _, already = store.prepare_write(
        "nb.ipynb", same, current_key=("mtime-2", 20), content_hash="h1"
    )
    assert already is True

    # Different content presented against the stale write version is rejected.
    changed = LeaseCredentials(lease.token, generation=1, lease_version=0)
    with pytest.raises(LeaseError) as exc:
        store.prepare_write("nb.ipynb", changed, current_key=("mtime-2", 20), content_hash="h2")
    assert exc.value.conflict.reason is LeaseConflictReason.duplicate_write

    # File touched out of band (fingerprint changed) blocks the next save.
    fresh = LeaseCredentials(lease.token, generation=1, lease_version=1)
    with pytest.raises(LeaseError) as exc:
        store.prepare_write("nb.ipynb", fresh, current_key=("external", 99), content_hash="h3")
    assert exc.value.conflict.reason is LeaseConflictReason.external_modified


def test_commit_write_does_not_extend_expiry(store):
    lease = store.acquire("nb.ipynb", "alice")
    deadline = lease.expires_at
    store.commit_write("nb.ipynb", lease.token, new_key="k")
    assert store.get("nb.ipynb").current.expires_at == deadline


def test_rename_merges_leftover_destination_history(store):
    # Destination name has a dead lease from a previously deleted file.
    old = store.acquire("dst.ipynb", "zoe")
    store.release("dst.ipynb", old.token)

    live = store.acquire("src.ipynb", "alice")
    store.move("src.ipynb", "dst.ipynb")

    # both old tokens are dead, the live one survives, generation continuous
    with pytest.raises(LeaseError) as exc:
        store.check("dst.ipynb", old.token, expected_generation=1)
    assert exc.value.conflict.reason is LeaseConflictReason.stale_token
    assert store.check("dst.ipynb", live.token, expected_generation=2) is not None


def test_rename_into_live_destination_is_rejected(store):
    store.acquire("dst.ipynb", "bob")
    store.acquire("src.ipynb", "alice")
    with pytest.raises(LeaseError) as exc:
        store.check_rename_destination("dst.ipynb")
    assert exc.value.conflict.reason is LeaseConflictReason.held_by_other
    assert exc.value.conflict.held_by == "bob"


def test_restart_of_process_invalidates_all_tokens():
    # A brand-new store models a restart: it has no memory of issued tokens.
    old = LeaseStore()
    lease = old.acquire("nb.ipynb", "alice")
    fresh = LeaseStore()
    with pytest.raises(LeaseError) as exc:
        fresh.check("nb.ipynb", lease.token, expected_generation=1)
    conflict = exc.value.conflict
    assert conflict.reason is LeaseConflictReason.not_found
    assert conflict.recoverable  # client can re-open

"""Integration tests for edit leases at the ContentsManager layer."""

import os
import time

import pytest
from jupyter_core.utils import ensure_async
from nbformat import from_dict
from nbformat.v4 import new_markdown_cell, new_notebook
from tornado.web import HTTPError

from jupyter_server.services.contents.filemanager import (
    AsyncFileContentsManager,
    FileContentsManager,
)
from jupyter_server.services.contents.leases import LeaseCredentials
from jupyter_server.services.contents.manager import LeaseHTTPError

from ...utils import expected_http_error


@pytest.fixture(params=["sync", "async"])
def cm(request, contents):
    root_dir = str(contents["contents_dir"])
    if request.param == "sync":
        manager = FileContentsManager(root_dir=root_dir)
    else:
        manager = AsyncFileContentsManager(root_dir=root_dir)
    yield manager


async def _new_nb(manager, path):
    model = {"type": "notebook", "content": new_notebook()}
    await ensure_async(manager.save(model, path))


async def _open(manager, path, owner="alice", **kwargs):
    return await ensure_async(manager.open_lease(path, owner, **kwargs))


def _creds(lease, **overrides):
    data = dict(token=lease.token, generation=lease.generation)
    data.update(overrides)
    return LeaseCredentials(**data)


def _nb_model(source):
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(source))
    return {"type": "notebook", "content": nb}


async def test_open_then_save_requires_matching_generation(cm, contents):
    path = "foo/a.ipynb"
    lease = await _open(cm, path)
    model = _nb_model("edited with lease")

    # no token while a lease is open -> rejected, nothing written
    with pytest.raises(HTTPError) as exc:
        await ensure_async(cm.save(model, path))
    assert expected_http_error(exc, 409)
    on_disk = from_dict((await ensure_async(cm.get(path)))["content"])
    assert on_disk.cells == [] or all(c.source != "edited with lease" for c in on_disk.cells)

    # wrong generation -> rejected
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.save(model, path, credentials=_creds(lease, generation=99)))
    assert exc.value.lease_conflict["reason"] == "generation_mismatch"

    # correct token + generation -> accepted, version bumped
    saved = await ensure_async(cm.save(model, path, credentials=_creds(lease)))
    assert saved["lease"]["generation"] == 1
    assert saved["lease"]["lease_version"] == 1


async def test_stale_client_after_takeover_cannot_overwrite(cm):
    """The reported incident: an early client reconnecting must not win."""
    path = "foo/a.ipynb"
    early = await _open(cm, path, owner="early-client")

    # Support takes over while the early client is disconnected.
    admin = await ensure_async(
        cm.takeover_lease(path, admin="support", reason="user reported lost edits")
    )
    await ensure_async(cm.save(_nb_model("support version"), path, credentials=_creds(admin)))

    # The early client reconnects with its (now dead) token and tries to save.
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.save(_nb_model("stale overwrite"), path, credentials=_creds(early)))
    conflict = exc.value.lease_conflict
    assert conflict["reason"] == "stale_token"
    assert conflict["recoverable"] is False
    assert conflict["current_generation"] == 2

    # ...and the confirmed newer content is still on disk.
    current = from_dict((await ensure_async(cm.get(path)))["content"])
    assert current.cells[0].source == "support version"

    # renewing the dead token is impossible too
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.renew_lease(path, _creds(early)))
    assert exc.value.lease_conflict["reason"] == "stale_token"


async def test_takeover_requires_reason(cm):
    path = "foo/a.ipynb"
    await _open(cm, path)
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.takeover_lease(path, admin="support", reason=""))
    assert exc.value.lease_conflict["reason"] == "takeover_reason_required"


async def test_duplicate_submission_is_idempotent_then_conflicts(cm):
    path = "foo/a.ipynb"
    lease = await _open(cm, path)
    model = _nb_model("v1")
    creds_v0 = _creds(lease, lease_version=0)

    first = await ensure_async(cm.save(model, path, credentials=creds_v0))
    assert first["lease"]["lease_version"] == 1

    # identical retried request with the same version -> no rewrite, no error
    duplicate = await ensure_async(cm.save(model, path, credentials=creds_v0))
    assert duplicate["lease"]["lease_version"] == 1

    # a genuinely different save carrying the old write version -> conflict
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.save(_nb_model("v2"), path, credentials=creds_v0))
    assert exc.value.lease_conflict["reason"] == "duplicate_write"

    # advancing the client's known version lets the save through
    saved = await ensure_async(
        cm.save(_nb_model("v2"), path, credentials=_creds(lease, lease_version=1))
    )
    assert saved["lease"]["lease_version"] == 2


async def test_external_modification_blocks_save(cm, contents):
    path = "foo/a.ipynb"
    lease = await _open(cm, path)

    # Someone (another process, an external editor) changes the file on disk.
    os_path = cm._get_os_path(path)
    with open(os_path, "w", encoding="utf-8") as f:
        f.write(
            '{"cells": [], "metadata": {"externally": true}, "nbformat": 4, "nbformat_minor": 5}'
        )
    # force an mtime strictly newer than the baseline
    future = time.time() + 5
    os.utime(os_path, (future, future))

    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.save(_nb_model("would clobber"), path, credentials=_creds(lease)))
    assert exc.value.lease_conflict["reason"] == "external_modification"

    # re-opening resets the baseline and allows saving again
    await ensure_async(cm.release_lease(path, _creds(lease)))
    new_lease = await _open(cm, path, owner="alice")
    saved = await ensure_async(cm.save(_nb_model("merged"), path, credentials=_creds(new_lease)))
    assert saved["lease"]["lease_version"] == 1


async def test_rename_validates_lease_and_carries_it(cm):
    path = "foo/a.ipynb"
    new_path = "foo/a-renamed.ipynb"
    lease = await _open(cm, path)

    # rename without the token is blocked
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.rename(path, new_path))
    assert exc.value.lease_conflict["reason"] == "lease_held_by_other"
    assert await ensure_async(cm.file_exists(path))

    # rename with the fence moves the lease with the file
    await ensure_async(cm.rename(path, new_path, credentials=_creds(lease)))
    assert not await ensure_async(cm.file_exists(path))
    assert await ensure_async(cm.file_exists(new_path))

    # the same token/generation now fences at the new path
    saved = await ensure_async(
        cm.save(_nb_model("at new name"), new_path, credentials=_creds(lease))
    )
    assert saved["path"] == new_path
    assert saved["lease"]["generation"] == 1


async def test_delete_validates_lease_and_kills_token(cm):
    path = "foo/b.ipynb"
    await _new_nb(cm, path)
    lease = await _open(cm, path)

    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.delete(path))
    assert exc.value.lease_conflict["reason"] == "lease_held_by_other"
    assert await ensure_async(cm.file_exists(path))

    await ensure_async(cm.delete(path, credentials=_creds(lease)))
    assert not await ensure_async(cm.file_exists(path))
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.save(_nb_model("ghost"), path, credentials=_creds(lease)))
    assert exc.value.lease_conflict["reason"] == "stale_token"


async def test_checkpoint_associates_version_without_extending_lease(cm):
    path = "foo/a.ipynb"
    lease = await _open(cm, path)
    await ensure_async(cm.save(_nb_model("v1"), path, credentials=_creds(lease, lease_version=0)))
    deadline = cm.lease_store.get(path).current.expires_at

    cp = await ensure_async(cm.create_checkpoint(path))
    assert cp["lease_generation"] == 1
    assert cp["lease_version"] == 1
    # creating a checkpoint does not extend write permission
    assert cm.lease_store.get(path).current.expires_at == deadline

    # restore is a write: it is fenced like save
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.restore_checkpoint(cp["id"], path))
    assert exc.value.lease_conflict["reason"] == "lease_held_by_other"

    await ensure_async(cm.save(_nb_model("v2"), path, credentials=_creds(lease, lease_version=1)))
    await ensure_async(
        cm.restore_checkpoint(cp["id"], path, credentials=_creds(lease, lease_version=2))
    )
    current = from_dict((await ensure_async(cm.get(path)))["content"])
    assert current.cells[0].source == "v1"
    assert cm.lease_store.get(path).current.lease_version == 3


async def test_restart_like_fresh_manager_rejects_old_token(contents):
    root_dir = str(contents["contents_dir"])
    first = FileContentsManager(root_dir=root_dir)
    path = "foo/a.ipynb"
    lease = first.open_lease(path, "alice")

    # A new manager instance models a service restart: in-memory lease state
    # is gone, so the old token cannot fence anymore.
    restarted = FileContentsManager(root_dir=root_dir)
    with pytest.raises(LeaseHTTPError) as exc:
        restarted.save(_nb_model("after restart"), path, credentials=_creds(lease))
    conflict = exc.value.lease_conflict
    assert conflict["reason"] == "lease_not_found"
    assert conflict["recoverable"] is True


async def test_create_new_file_under_lease(cm):
    path = "foo/brand-new.ipynb"
    assert not await ensure_async(cm.file_exists(path))
    lease = await _open(cm, path)

    # a tokenless create of that name is blocked while the lease is open
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.new({"type": "notebook", "content": new_notebook()}, path))
    assert exc.value.lease_conflict["reason"] == "lease_held_by_other"

    # creation with the fence succeeds
    saved = await ensure_async(
        cm.new(
            {"type": "notebook", "content": new_notebook()},
            path,
            credentials=_creds(lease),
        )
    )
    assert saved["path"] == path
    assert await ensure_async(cm.file_exists(path))


async def test_rename_toward_open_name_is_rejected(cm):
    # two existing files, each with an open lease
    src = "foo/a.ipynb"
    dst = "foo/b.ipynb"
    src_lease = await _open(cm, src, owner="alice")
    await _open(cm, dst, owner="bob")
    with pytest.raises(LeaseHTTPError) as exc:
        await ensure_async(cm.rename(src, dst, credentials=_creds(src_lease)))
    assert exc.value.lease_conflict["reason"] == "lease_held_by_other"
    assert await ensure_async(cm.file_exists(src))


async def test_expired_lease_is_rejected_after_grace(contents):
    root_dir = str(contents["contents_dir"])
    manager = FileContentsManager(
        root_dir=root_dir, lease_ttl_seconds=0.01, lease_clock_skew_grace_seconds=0.0
    )
    path = "foo/a.ipynb"
    lease = manager.open_lease(path, "alice")
    time.sleep(0.05)
    with pytest.raises(LeaseHTTPError) as exc:
        manager.save(_nb_model("late"), path, credentials=_creds(lease))
    assert exc.value.lease_conflict["reason"] == "stale_token"

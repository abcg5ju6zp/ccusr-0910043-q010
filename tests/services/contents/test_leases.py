"""项目内部接口说明。"""

import asyncio
import json
import os
import time
import warnings

import pytest
import tornado
from jupyter_core.utils import ensure_async
from nbformat.v4 import new_notebook

from jupyter_server.services.contents.filemanager import (
    AsyncFileContentsManager,
    FileContentsManager,
)
from jupyter_server.services.contents.leases import LeaseConflictError

from ...utils import expected_http_error


@pytest.fixture(autouse=True)
def suppress_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The synchronous ContentsManager",
            category=DeprecationWarning,
        )
        yield


@pytest.fixture(params=[FileContentsManager, AsyncFileContentsManager])
def jp_contents_manager(request, tmp_path):
    return request.param(root_dir=str(tmp_path))


@pytest.fixture(params=[FileContentsManager, AsyncFileContentsManager])
def jp_file_contents_manager_class(request):
    return request.param


# -------------- Helpers ----------------------------


async def _make_file(cm, path, text="initial"):
    model = {"type": "file", "content": text, "format": "text"}
    return await ensure_async(cm.save(model, path))


async def _save_text(cm, path, text, lease=None):
    model = {"type": "file", "content": text, "format": "text"}
    return await ensure_async(cm.save(model, path, lease=lease))


async def _read_text(cm, path):
    model = await ensure_async(cm.get(path))
    return model["content"]


async def _acquire(cm, path, holder="analyst-a", ttl=None):
    lease, created = await ensure_async(cm.open_lease(path, holder=holder, ttl=ttl))
    assert created
    return lease


def _token(lease, request_id=None):
    token = {"lease_id": lease["lease_id"], "generation": lease["generation"]}
    if request_id is not None:
        token["request_id"] = request_id
    return token


def _assert_conflict(excinfo, code):
    error = excinfo.value
    assert isinstance(error, LeaseConflictError)
    assert error.status_code == 409
    conflict = error.conflict
    assert conflict["code"] == code
    assert conflict["recovery"]
    return conflict


# -------------- Lease lifecycle ----------------------------


async def test_acquire_save_roundtrip(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt")
    assert lease["generation"] == 1
    assert lease["holder"] == "analyst-a"
    await _save_text(cm, "a.txt", "edited", lease=_token(lease))
    assert await _read_text(cm, "a.txt") == "edited"


async def test_second_holder_rejected_and_reconnect_recovers(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt", holder="analyst-a")

    # A second analyst cannot acquire while the lease is live.
    with pytest.raises(LeaseConflictError) as excinfo:
        await ensure_async(cm.open_lease("a.txt", holder="analyst-b"))
    conflict = _assert_conflict(excinfo, "lease_held")
    assert conflict["holder"] == "analyst-a"
    assert conflict["recovery"] == "retry"

    # The same holder re-acquires its own lease after a reconnect.
    again, created = await ensure_async(cm.open_lease("a.txt", holder="analyst-a"))
    assert not created
    assert again["lease_id"] == lease["lease_id"]
    assert again["generation"] == lease["generation"]


async def test_renew_and_release(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt", ttl=60)
    token = _token(lease)

    renewed = await ensure_async(
        cm.renew_lease("a.txt", token["lease_id"], token["generation"], ttl=120)
    )
    assert renewed["generation"] == lease["generation"]
    assert renewed["expires_at"] > lease["expires_at"]

    # Renewing with a stale generation is rejected.
    with pytest.raises(LeaseConflictError) as excinfo:
        await ensure_async(
            cm.renew_lease("a.txt", token["lease_id"], token["generation"] + 1, ttl=120)
        )
    _assert_conflict(excinfo, "stale_generation")

    await ensure_async(cm.release_lease("a.txt", token["lease_id"], token["generation"]))
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "x", lease=token)
    conflict = _assert_conflict(excinfo, "lease_invalidated")
    assert conflict["ended_by"] == "released"

    # A fresh acquire starts a new generation.
    successor = await _acquire(cm, "a.txt", holder="analyst-b")
    assert successor["generation"] == lease["generation"] + 1


async def test_lease_expiry(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt", ttl=0.2)
    token = _token(lease)
    await asyncio.sleep(0.4)
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "late", lease=token)
    _assert_conflict(excinfo, "lease_expired")

    # Another holder can acquire once the lease has expired.
    successor = await _acquire(cm, "a.txt", holder="analyst-b")
    assert successor["generation"] == lease["generation"] + 1
    await _save_text(cm, "a.txt", "taken over", lease=_token(successor))


async def test_stale_generation_and_unknown_token(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt")

    future = {"lease_id": lease["lease_id"], "generation": lease["generation"] + 1}
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "x", lease=future)
    _assert_conflict(excinfo, "stale_generation")

    bogus = {"lease_id": "not-a-real-lease", "generation": lease["generation"]}
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "x", lease=bogus)
    _assert_conflict(excinfo, "lease_held")

    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "other.txt", "x", lease=bogus)
    _assert_conflict(excinfo, "lease_unknown")


async def test_unleased_writes_allowed_in_optin_mode(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    await _acquire(cm, "a.txt")
    # Legacy clients without a token still write in opt-in mode.
    await _save_text(cm, "a.txt", "legacy")
    assert await _read_text(cm, "a.txt") == "legacy"


# -------------- The two-analyst scenario ----------------------------


async def test_stale_client_cannot_overwrite_after_takeover(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "report.txt", text="v1")

    # Analyst A opens the report, then disconnects.
    lease_a = await _acquire(cm, "report.txt", holder="analyst-a")
    token_a = _token(lease_a)

    # Support takes the lease over for analyst B and must record a reason.
    lease_b = await ensure_async(
        cm.takeover_lease(
            "report.txt", holder="analyst-b", reason="A disconnected mid-edit", by="support-1"
        )
    )
    assert lease_b["generation"] == lease_a["generation"] + 1
    token_b = _token(lease_b)

    # B saves the confirmed new version.
    await _save_text(cm, "report.txt", "v2-confirmed", lease=token_b)

    # A reconnects and tries to save its stale buffer with the old token.
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "report.txt", "v1-stale", lease=token_a)
    conflict = _assert_conflict(excinfo, "lease_invalidated")
    assert conflict["ended_by"] == "takeover"
    assert conflict["ended_reason"] == "A disconnected mid-edit"
    assert conflict["recovery"] == "open"

    # The confirmed version was not silently overwritten.
    assert await _read_text(cm, "report.txt") == "v2-confirmed"

    # The old token stays invalid no matter how often A retries.
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "report.txt", "v1-stale", lease=token_a)
    _assert_conflict(excinfo, "lease_invalidated")

    # Support can see the audit trail instead of comparing mtimes.
    view = await ensure_async(cm.inspect_lease("report.txt"))
    assert view["active"]
    assert view["holder"] == "analyst-b"
    assert view["invalidations"][0]["reason"] == "A disconnected mid-edit"
    assert view["invalidations"][0]["by"] == "support-1"


async def test_takeover_requires_reason(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    await _acquire(cm, "a.txt")
    from tornado.web import HTTPError

    with pytest.raises(HTTPError) as excinfo:
        await ensure_async(cm.takeover_lease("a.txt", holder="analyst-b", reason=""))
    assert excinfo.value.status_code == 400
    with pytest.raises(HTTPError) as excinfo:
        await ensure_async(cm.takeover_lease("a.txt", holder="analyst-b", reason="   "))
    assert excinfo.value.status_code == 400


# -------------- Duplicate submissions ----------------------------


async def test_duplicate_save_is_replayed(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt")
    token = _token(lease, request_id="req-1")

    await _save_text(cm, "a.txt", "first", lease=token)
    # The client retries the same request after a reconnect; the second
    # attempt must not be applied again nor reported as a conflict.
    await _save_text(cm, "a.txt", "second", lease=token)
    assert await _read_text(cm, "a.txt") == "first"

    # A new request id performs a real save again.
    await _save_text(cm, "a.txt", "third", lease=_token(lease, request_id="req-2"))
    assert await _read_text(cm, "a.txt") == "third"


async def test_request_id_reused_for_other_operation(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt")
    await _save_text(cm, "a.txt", "first", lease=_token(lease, request_id="req-1"))

    with pytest.raises(LeaseConflictError) as excinfo:
        await ensure_async(cm.delete("a.txt", lease=_token(lease, request_id="req-1")))
    _assert_conflict(excinfo, "duplicate_request")
    assert await ensure_async(cm.file_exists("a.txt"))


async def test_duplicate_delete_is_replayed(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt")
    token = _token(lease, request_id="req-del")
    await ensure_async(cm.delete("a.txt", lease=token))
    assert not await ensure_async(cm.file_exists("a.txt"))
    # Retrying the same delete reports success instead of a 404.
    await ensure_async(cm.delete("a.txt", lease=token))


# -------------- External modification ----------------------------


async def test_external_modification_blocks_save(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt", "v1")
    lease = await _acquire(cm, "a.txt")
    token = _token(lease)

    # Something outside the contents service rewrites the file.
    os_path = cm._get_os_path("a.txt")
    with open(os_path, "w", encoding="utf-8") as f:
        f.write("external")

    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "v2", lease=token)
    conflict = _assert_conflict(excinfo, "external_modification")
    assert conflict["recovery"] == "reload"
    with open(os_path, encoding="utf-8") as f:
        assert f.read() == "external"

    # Re-acquiring without reloading is rejected as well; the client must
    # release the stale lease first.
    with pytest.raises(LeaseConflictError) as excinfo:
        await ensure_async(cm.open_lease("a.txt", holder="analyst-a"))
    _assert_conflict(excinfo, "external_modification")

    # After reloading, the client releases and re-acquires with a fresh
    # baseline, then saves normally.
    await ensure_async(cm.release_lease("a.txt", token["lease_id"], token["generation"]))
    lease2 = await _acquire(cm, "a.txt")
    await _save_text(cm, "a.txt", "v2", lease=_token(lease2))
    assert await _read_text(cm, "a.txt") == "v2"


async def test_external_delete_blocks_save(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt", "v1")
    lease = await _acquire(cm, "a.txt")
    token = _token(lease)
    os.unlink(cm._get_os_path("a.txt"))
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "v2", lease=token)
    _assert_conflict(excinfo, "external_modification")


# -------------- Restart and clock skew ----------------------------


async def test_lease_state_survives_restart(tmp_path, jp_file_contents_manager_class):
    cm1 = jp_file_contents_manager_class(root_dir=str(tmp_path))
    await _make_file(cm1, "a.txt", "v1")
    lease_a = await _acquire(cm1, "a.txt", holder="analyst-a")
    token_a = _token(lease_a)
    lease_b = await ensure_async(
        cm1.takeover_lease("a.txt", holder="analyst-b", reason="handover", by="ops")
    )
    token_b = _token(lease_b)

    # A fresh manager over the same root directory simulates a restart.
    cm2 = jp_file_contents_manager_class(root_dir=str(tmp_path))

    # The invalidated token stays invalid across the restart.
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm2, "a.txt", "stale", lease=token_a)
    conflict = _assert_conflict(excinfo, "lease_invalidated")
    assert conflict["ended_reason"] == "handover"

    # The live lease still validates, and generations keep increasing.
    await _save_text(cm2, "a.txt", "current", lease=token_b)
    await ensure_async(cm2.release_lease("a.txt", token_b["lease_id"], token_b["generation"]))
    lease_c = await _acquire(cm2, "a.txt", holder="analyst-c")
    assert lease_c["generation"] == lease_b["generation"] + 1


async def test_duplicate_request_replayed_after_restart(tmp_path, jp_file_contents_manager_class):
    cm1 = jp_file_contents_manager_class(root_dir=str(tmp_path))
    await _make_file(cm1, "a.txt")
    lease = await _acquire(cm1, "a.txt")
    token = _token(lease, request_id="req-1")
    await _save_text(cm1, "a.txt", "committed", lease=token)

    # The response was lost and the server restarted; the client retries.
    cm2 = jp_file_contents_manager_class(root_dir=str(tmp_path))
    await _save_text(cm2, "a.txt", "committed-retry", lease=token)
    assert await _read_text(cm2, "a.txt") == "committed"


async def test_restart_with_forward_clock_skew_fails_closed(
    tmp_path, jp_file_contents_manager_class, monkeypatch
):
    cm1 = jp_file_contents_manager_class(root_dir=str(tmp_path))
    await _make_file(cm1, "a.txt")
    lease = await _acquire(cm1, "a.txt", ttl=100)
    token = _token(lease)

    # The wall clock jumps forwards across a restart: the lease must be
    # treated as expired rather than silently accepted.
    real_now = time.time()
    monkeypatch.setattr(time, "time", lambda: real_now + 10_000)
    cm2 = jp_file_contents_manager_class(root_dir=str(tmp_path))
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm2, "a.txt", "stale", lease=token)
    conflict = _assert_conflict(excinfo, "lease_invalidated")
    assert conflict["ended_by"] == "expired"


async def test_restart_with_backward_clock_skew_caps_ttl(
    tmp_path, jp_file_contents_manager_class, monkeypatch
):
    cm1 = jp_file_contents_manager_class(root_dir=str(tmp_path))
    await _make_file(cm1, "a.txt")
    lease = await _acquire(cm1, "a.txt", ttl=100)
    token = _token(lease)

    # The wall clock jumps backwards across a restart: the remaining lease
    # time is capped at the TTL instead of being extended by the skew.
    real_now = time.time()
    monkeypatch.setattr(time, "time", lambda: real_now - 10_000)
    cm2 = jp_file_contents_manager_class(root_dir=str(tmp_path))
    await _save_text(cm2, "a.txt", "still valid", lease=token)
    assert await _read_text(cm2, "a.txt") == "still valid"


# -------------- Checkpoints ----------------------------


async def test_checkpoint_records_generation_but_does_not_extend_lease(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt", ttl=0.4)
    token = _token(lease)

    await asyncio.sleep(0.2)
    checkpoint = await ensure_async(cm.create_checkpoint("a.txt", lease=token))
    assert checkpoint["lease_generation"] == lease["generation"]

    # The lease expires on its original schedule even though a checkpoint
    # was created halfway through its lifetime.
    await asyncio.sleep(0.3)
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "late", lease=token)
    _assert_conflict(excinfo, "lease_expired")


async def test_restore_checkpoint_validates_lease(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt", "v1")
    lease = await _acquire(cm, "a.txt")
    token = _token(lease)
    checkpoint = await ensure_async(cm.create_checkpoint("a.txt", lease=token))
    await _save_text(cm, "a.txt", "v2", lease=token)

    other = await ensure_async(
        cm.takeover_lease("a.txt", holder="analyst-b", reason="handover", by="ops")
    )
    with pytest.raises(LeaseConflictError) as excinfo:
        await ensure_async(cm.restore_checkpoint(checkpoint["id"], "a.txt", lease=token))
    _assert_conflict(excinfo, "lease_invalidated")

    # Restoring is a write: it succeeds with the current token.
    await ensure_async(cm.restore_checkpoint(checkpoint["id"], "a.txt", lease=_token(other)))
    assert await _read_text(cm, "a.txt") == "v1"


# -------------- Rename and delete ----------------------------


async def test_rename_transfers_lease(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt")
    token = _token(lease)

    await ensure_async(cm.update({"path": "b.txt"}, "a.txt", lease=token))
    assert await ensure_async(cm.file_exists("b.txt"))

    # The same token now writes at the new path...
    await _save_text(cm, "b.txt", "renamed", lease=token)
    assert await _read_text(cm, "b.txt") == "renamed"

    # ...and no longer means anything at the old path.
    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "x", lease=token)
    _assert_conflict(excinfo, "lease_unknown")


async def test_rename_with_stale_token_rejected(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    await _make_file(cm, "b.txt")
    lease_a = await _acquire(cm, "a.txt", holder="analyst-a")
    lease_b = await _acquire(cm, "b.txt", holder="analyst-b")

    # A cannot rename over B's live lease.
    with pytest.raises(LeaseConflictError) as excinfo:
        await ensure_async(cm.rename("a.txt", "b.txt", lease=_token(lease_a)))
    _assert_conflict(excinfo, "lease_held")
    assert await ensure_async(cm.file_exists("a.txt"))

    # B's own token cannot rename A's file either.
    with pytest.raises(LeaseConflictError) as excinfo:
        await ensure_async(cm.rename("a.txt", "c.txt", lease=_token(lease_b)))
    _assert_conflict(excinfo, "lease_held")


async def test_delete_invalidates_lease(jp_contents_manager):
    cm = jp_contents_manager
    await _make_file(cm, "a.txt")
    lease = await _acquire(cm, "a.txt")
    token = _token(lease)
    await ensure_async(cm.delete("a.txt", lease=token))
    assert not await ensure_async(cm.file_exists("a.txt"))

    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "zombie", lease=token)
    conflict = _assert_conflict(excinfo, "lease_invalidated")
    assert conflict["ended_by"] == "delete"


# -------------- require_lease mode ----------------------------


async def test_require_lease_rejects_unleased_writes(tmp_path, jp_file_contents_manager_class):
    cm = jp_file_contents_manager_class(root_dir=str(tmp_path), require_lease=True)

    with pytest.raises(LeaseConflictError) as excinfo:
        await _save_text(cm, "a.txt", "x")
    conflict = _assert_conflict(excinfo, "lease_required")
    assert conflict["recovery"] == "open"

    lease = await _acquire(cm, "a.txt")
    token = _token(lease)
    await _save_text(cm, "a.txt", "x", lease=token)

    with pytest.raises(LeaseConflictError) as excinfo:
        await ensure_async(cm.delete("a.txt"))
    _assert_conflict(excinfo, "lease_required")

    await ensure_async(cm.delete("a.txt", lease=token))
    assert not await ensure_async(cm.file_exists("a.txt"))


# -------------- Notebook content sanity ----------------------------


async def test_notebook_save_with_lease(jp_contents_manager):
    cm = jp_contents_manager
    await ensure_async(cm.new(model={"type": "notebook"}, path="nb.ipynb"))
    lease = await _acquire(cm, "nb.ipynb")
    nb = new_notebook()
    nb["metadata"]["origin"] = "leased"
    model = {"type": "notebook", "content": nb, "format": "json"}
    await ensure_async(cm.save(model, "nb.ipynb", lease=_token(lease)))
    saved = await ensure_async(cm.get("nb.ipynb"))
    assert saved["content"]["metadata"]["origin"] == "leased"


# -------------- HTTP API ----------------------------


@pytest.fixture(params=["FileContentsManager", "AsyncFileContentsManager"])
def jp_argv(request):
    return [
        "--ServerApp.contents_manager_class=jupyter_server.services.contents.filemanager."
        + request.param
    ]


async def _api_make_file(jp_fetch, path, text="initial"):
    model = {"type": "file", "content": text, "format": "text"}
    await jp_fetch("api", "contents", path, method="PUT", body=json.dumps(model))


async def _api_acquire(jp_fetch, path, holder, ttl=None):
    body = {"holder": holder}
    if ttl is not None:
        body["ttl"] = ttl
    r = await jp_fetch("api", "contents", path, "lease", method="POST", body=json.dumps(body))
    return r, json.loads(r.body.decode())


async def test_lease_api_flow(jp_fetch):
    await _api_make_file(jp_fetch, "flow.txt")

    # Acquire a lease.
    r, lease = await _api_acquire(jp_fetch, "flow.txt", "analyst-a")
    assert r.code == 201
    assert lease["generation"] == 1
    token = {"lease_id": lease["lease_id"], "generation": lease["generation"]}

    # Re-acquiring as the same holder recovers the lease after a reconnect.
    r, again = await _api_acquire(jp_fetch, "flow.txt", "analyst-a")
    assert r.code == 200
    assert again["lease_id"] == lease["lease_id"]

    # The inspection view supports operations staff without leaking tokens.
    r = await jp_fetch("api", "contents", "flow.txt", "lease", method="GET")
    view = json.loads(r.body.decode())
    assert view["active"] and view["holder"] == "analyst-a"
    assert "lease_id" not in view

    # Save with the lease embedded in the model body.
    model = {"type": "file", "content": "v2", "format": "text", "lease": token}
    r = await jp_fetch("api", "contents", "flow.txt", method="PUT", body=json.dumps(model))
    assert r.code == 200

    # A second analyst is told who holds the lease and how to proceed.
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await _api_acquire(jp_fetch, "flow.txt", "analyst-b")
    assert expected_http_error(e, 409)
    payload = json.loads(e.value.response.body.decode())
    assert payload["lease_conflict"]["code"] == "lease_held"
    assert payload["lease_conflict"]["holder"] == "analyst-a"
    assert payload["lease_conflict"]["recovery"] == "retry"

    # A takeover must carry a reason.
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await jp_fetch(
            "api",
            "contents",
            "flow.txt",
            "lease",
            "takeover",
            method="POST",
            body=json.dumps({"holder": "analyst-b"}),
        )
    assert expected_http_error(e, 400)

    r = await jp_fetch(
        "api",
        "contents",
        "flow.txt",
        "lease",
        "takeover",
        method="POST",
        body=json.dumps({"holder": "analyst-b", "reason": "A went offline"}),
    )
    assert r.code == 201
    lease_b = json.loads(r.body.decode())
    assert lease_b["generation"] == lease["generation"] + 1

    # The old token is permanently invalidated and the client is told why.
    model["lease"] = token
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await jp_fetch("api", "contents", "flow.txt", method="PUT", body=json.dumps(model))
    assert expected_http_error(e, 409)
    payload = json.loads(e.value.response.body.decode())
    assert payload["lease_conflict"]["code"] == "lease_invalidated"
    assert payload["lease_conflict"]["ended_reason"] == "A went offline"

    # Renew and release with the new token.
    r = await jp_fetch(
        "api",
        "contents",
        "flow.txt",
        "lease",
        method="PUT",
        body=json.dumps(
            {"lease_id": lease_b["lease_id"], "generation": lease_b["generation"], "ttl": 120}
        ),
    )
    assert json.loads(r.body.decode())["generation"] == lease_b["generation"]

    r = await jp_fetch(
        "api",
        "contents",
        "flow.txt",
        "lease",
        method="DELETE",
        params={"lease_id": lease_b["lease_id"], "lease_generation": lease_b["generation"]},
    )
    assert r.code == 204

    r = await jp_fetch("api", "contents", "flow.txt", "lease", method="GET")
    view = json.loads(r.body.decode())
    assert not view["active"]
    # The takeover reason remains on record for support staff.
    takeover_entries = [i for i in view["invalidations"] if i["ended_by"] == "takeover"]
    assert takeover_entries[-1]["reason"] == "A went offline"
    assert view["invalidations"][-1]["ended_by"] == "released"


async def test_lease_via_headers_and_save_conflict(jp_fetch):
    await _api_make_file(jp_fetch, "hdr.txt")
    _, lease = await _api_acquire(jp_fetch, "hdr.txt", "analyst-a")

    # Save using headers instead of a body field.
    headers = {
        "X-Jupyter-Lease-Id": lease["lease_id"],
        "X-Jupyter-Lease-Generation": str(lease["generation"]),
        "X-Jupyter-Request-Id": "req-header-1",
    }
    model = {"type": "file", "content": "via headers", "format": "text"}
    r = await jp_fetch(
        "api", "contents", "hdr.txt", method="PUT", body=json.dumps(model), headers=headers
    )
    assert r.code == 200

    # A retry with the same request id replays instead of failing.
    model["content"] = "changed on retry"
    r = await jp_fetch(
        "api", "contents", "hdr.txt", method="PUT", body=json.dumps(model), headers=headers
    )
    assert r.code == 200
    r = await jp_fetch("api", "contents", "hdr.txt", method="GET")
    assert json.loads(r.body.decode())["content"] == "via headers"

    # A wrong generation is a recoverable conflict, not an overwrite.
    headers["X-Jupyter-Lease-Generation"] = str(lease["generation"] + 5)
    del headers["X-Jupyter-Request-Id"]
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await jp_fetch(
            "api", "contents", "hdr.txt", method="PUT", body=json.dumps(model), headers=headers
        )
    assert expected_http_error(e, 409)
    payload = json.loads(e.value.response.body.decode())
    assert payload["lease_conflict"]["code"] == "stale_generation"


async def test_lease_api_rename_delete_and_checkpoints(jp_fetch):
    await _api_make_file(jp_fetch, "doc.txt", "v1")
    _, lease = await _api_acquire(jp_fetch, "doc.txt", "analyst-a")
    token = {"lease_id": lease["lease_id"], "generation": lease["generation"]}

    # Checkpoints record the lease generation they were taken under.
    r = await jp_fetch(
        "api",
        "contents",
        "doc.txt",
        "checkpoints",
        method="POST",
        body=json.dumps({}),
        params={"lease_id": token["lease_id"], "lease_generation": token["generation"]},
    )
    assert r.code == 201
    checkpoint = json.loads(r.body.decode())
    assert checkpoint["lease_generation"] == lease["generation"]

    # Rename (PATCH) with the lease in the body follows the file.
    r = await jp_fetch(
        "api",
        "contents",
        "doc.txt",
        method="PATCH",
        body=json.dumps({"path": "renamed.txt", "lease": token}),
    )
    assert r.code == 200

    # Delete with the lease as query arguments.
    r = await jp_fetch(
        "api",
        "contents",
        "renamed.txt",
        method="DELETE",
        params={"lease_id": token["lease_id"], "lease_generation": token["generation"]},
    )
    assert r.code == 204

    # The deleted file's token is finished for good.
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await jp_fetch(
            "api",
            "contents",
            "renamed.txt",
            method="PUT",
            body=json.dumps(
                {"type": "file", "content": "zombie", "format": "text", "lease": token}
            ),
        )
    assert expected_http_error(e, 409)
    payload = json.loads(e.value.response.body.decode())
    assert payload["lease_conflict"]["code"] == "lease_invalidated"
    assert payload["lease_conflict"]["ended_by"] == "delete"

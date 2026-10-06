"""End-to-end HTTP tests for the fencing-token edit lease API."""

import json
import warnings

import pytest
import tornado
from nbformat import from_dict
from nbformat.v4 import new_markdown_cell

from jupyter_server.utils import url_path_join


@pytest.fixture(autouse=True)
def suppress_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The synchronous ContentsManager",
            category=DeprecationWarning,
        )
        yield


@pytest.fixture(params=["FileContentsManager", "AsyncFileContentsManager"])
def jp_argv(request):
    return [
        "--ServerApp.contents_manager_class=jupyter_server.services.contents.filemanager."
        + request.param
    ]


TOKEN_H = "X-Jupyter-Edit-Lease"
GEN_H = "X-Jupyter-Edit-Generation"
VER_H = "X-Jupyter-Edit-Version"


async def _open_lease(jp_fetch, path, body=None):
    r = await jp_fetch(
        "api",
        "contents",
        path,
        "lease",
        method="POST",
        body=json.dumps(body or {}),
        allow_nonstandard_methods=True,
    )
    assert r.code == 201
    return json.loads(r.body.decode())


async def _get_nb(jp_fetch, path):
    r = await jp_fetch("api", "contents", path, method="GET")
    return json.loads(r.body.decode())


def _save_body(source):
    nb = from_dict({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    nb.cells.append(new_markdown_cell(source))
    return json.dumps({"type": "notebook", "content": nb})


async def test_open_renew_release_cycle(jp_fetch, contents):
    path = "foo/a.ipynb"
    lease = await _open_lease(jp_fetch, path, {"owner_label": "session-1"})
    assert lease["token"] and lease["generation"] == 1
    assert lease["owner_label"] == "session-1"

    # the contents model reports the open generation but never the token
    model = await _get_nb(jp_fetch, path)
    assert model["lease"]["generation"] == 1
    assert "token" not in model["lease"]

    # second open is a conflict naming the holder
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc:
        await _open_lease(jp_fetch, path)
    err = exc.value.response
    assert err.code == 409
    payload = json.loads(err.body.decode())
    assert payload["reason"] == "lease_held_by_other"
    assert payload["held_by"] == lease["owner"]

    # renew keeps the same token + generation
    r = await jp_fetch(
        "api",
        "contents",
        path,
        "lease",
        method="PUT",
        headers={TOKEN_H: lease["token"], GEN_H: "1"},
        body=json.dumps({}),
        allow_nonstandard_methods=True,
    )
    renewed = json.loads(r.body.decode())
    assert renewed["token"] == lease["token"]
    assert renewed["generation"] == 1

    # renew without a token is a client error
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc:
        await jp_fetch(
            "api",
            "contents",
            path,
            "lease",
            method="PUT",
            body=json.dumps({}),
            allow_nonstandard_methods=True,
        )
    assert exc.value.response.code == 400

    # release closes the lease, allowing it to be opened again
    await jp_fetch(
        "api",
        "contents",
        path,
        "lease",
        method="DELETE",
        headers={TOKEN_H: lease["token"]},
        allow_nonstandard_methods=True,
    )
    again = await _open_lease(jp_fetch, path)
    assert again["generation"] == 2


async def test_save_with_and_without_fence(jp_fetch, contents):
    path = "foo/a.ipynb"
    lease = await _open_lease(jp_fetch, path)

    # tokenless PUT while the document is open -> structured 409
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc:
        await jp_fetch(
            "api",
            "contents",
            path,
            method="PUT",
            body=_save_body("no fence"),
        )
    err = exc.value.response
    assert err.code == 409
    payload = json.loads(err.body.decode())
    assert payload["error"] == "lease_conflict"
    assert payload["reason"] == "lease_held_by_other"
    assert payload["path"] == path
    assert payload["recoverable"] is True

    # wrong generation -> recoverable conflict metadata
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc:
        await jp_fetch(
            "api",
            "contents",
            path,
            method="PUT",
            headers={TOKEN_H: lease["token"], GEN_H: "7"},
            body=_save_body("wrong gen"),
        )
    payload = json.loads(exc.value.response.body.decode())
    assert payload["reason"] == "generation_mismatch"
    assert payload["current_generation"] == 1

    # correct fence -> saved, response reports the new write version
    body = _save_body("fenced v1")
    r = await jp_fetch(
        "api",
        "contents",
        path,
        method="PUT",
        headers={TOKEN_H: lease["token"], GEN_H: "1", VER_H: "0"},
        body=body,
    )
    saved = json.loads(r.body.decode())
    assert saved["lease"]["lease_version"] == 1

    # byte-identical duplicate submission is accepted idempotently
    r = await jp_fetch(
        "api",
        "contents",
        path,
        method="PUT",
        headers={TOKEN_H: lease["token"], GEN_H: "1", VER_H: "0"},
        body=body,
    )
    assert json.loads(r.body.decode())["lease"]["lease_version"] == 1


async def test_admin_takeover_kills_old_token(jp_fetch, contents):
    path = "foo/a.ipynb"
    early = await _open_lease(jp_fetch, path, {"owner_label": "early client"})

    # takeover without reason is refused
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc:
        await jp_fetch(
            "api",
            "contents",
            path,
            "lease",
            "takeover",
            method="POST",
            body=json.dumps({}),
            allow_nonstandard_methods=True,
        )
    assert exc.value.response.code == 400

    # takeover with reason returns a new fence and records why
    r = await jp_fetch(
        "api",
        "contents",
        path,
        "lease",
        "takeover",
        method="POST",
        body=json.dumps({"reason": "support session #9123"}),
        allow_nonstandard_methods=True,
    )
    admin = json.loads(r.body.decode())
    assert admin["generation"] == 2
    assert admin["takeover_reason"] == "support session #9123"

    # the early client's reconnect/save is a permanent conflict
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc:
        await jp_fetch(
            "api",
            "contents",
            path,
            method="PUT",
            headers={TOKEN_H: early["token"], GEN_H: "1", VER_H: "0"},
            body=_save_body("stale reconnect"),
        )
    payload = json.loads(exc.value.response.body.decode())
    assert payload["reason"] == "stale_token"
    assert payload["recoverable"] is False
    assert "support session #9123" in payload["message"]

    # the admin fence works
    r = await jp_fetch(
        "api",
        "contents",
        path,
        method="PUT",
        headers={TOKEN_H: admin["token"], GEN_H: "2", VER_H: "0"},
        body=_save_body("support version"),
    )
    assert r.code == 200
    current = from_dict((await _get_nb(jp_fetch, path))["content"])
    assert current.cells[0].source == "support version"


async def test_rename_and_delete_are_fenced(jp_fetch, contents):
    path = "foo/a.ipynb"
    moved = "foo/a-moved.ipynb"
    lease = await _open_lease(jp_fetch, path)

    # PATCH rename without the fence is rejected
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc:
        await jp_fetch(
            "api",
            "contents",
            path,
            method="PATCH",
            body=json.dumps({"path": moved}),
        )
    assert json.loads(exc.value.response.body.decode())["reason"] == "lease_held_by_other"

    # with the fence the rename succeeds and the lease follows
    r = await jp_fetch(
        "api",
        "contents",
        path,
        method="PATCH",
        headers={TOKEN_H: lease["token"], GEN_H: "1"},
        body=json.dumps({"path": moved}),
    )
    assert json.loads(r.body.decode())["path"] == moved

    # DELETE must fence as well, using the same generation at the new name
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc:
        await jp_fetch("api", "contents", moved, method="DELETE")
    assert exc.value.response.code == 409

    await jp_fetch(
        "api",
        "contents",
        moved,
        method="DELETE",
        headers={TOKEN_H: lease["token"], GEN_H: "1"},
    )
    with pytest.raises(tornado.httpclient.HTTPClientError):
        await jp_fetch("api", "contents", moved, method="GET")


async def test_checkpoint_carries_lease_version(jp_fetch, contents):
    path = "foo/a.ipynb"
    lease = await _open_lease(jp_fetch, path)
    await jp_fetch(
        "api",
        "contents",
        path,
        method="PUT",
        headers={TOKEN_H: lease["token"], GEN_H: "1", VER_H: "0"},
        body=_save_body("v1"),
    )
    r = await jp_fetch(
        "api",
        "contents",
        path,
        "checkpoints",
        method="POST",
        allow_nonstandard_methods=True,
    )
    cp = json.loads(r.body.decode())
    assert cp["lease_generation"] == 1
    assert cp["lease_version"] == 1

    # restore without a fence is a conflict
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc:
        await jp_fetch(
            "api",
            "contents",
            path,
            "checkpoints",
            cp["id"],
            method="POST",
            body=b"",
            allow_nonstandard_methods=True,
        )
    assert exc.value.response.code == 409

    # restore with a fence works
    await jp_fetch(
        "api",
        "contents",
        path,
        "checkpoints",
        cp["id"],
        method="POST",
        headers={TOKEN_H: lease["token"], GEN_H: "1", VER_H: "1"},
        body=b"",
        allow_nonstandard_methods=True,
    )

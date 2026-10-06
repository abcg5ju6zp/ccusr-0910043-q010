"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import json
from http import HTTPStatus
from typing import Any

try:
    from jupyter_client.jsonutil import json_default
except ImportError:
    from jupyter_client.jsonutil import date_default as json_default

from jupyter_core.utils import ensure_async
from tornado import web

from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.base.handlers import APIHandler, JupyterHandler, path_regex
from jupyter_server.utils import url_escape, url_path_join

from .leases import LeaseCredentials
from .manager import LeaseHTTPError

AUTH_RESOURCE = "contents"

#: Header carrying the opaque fencing token.
LEASE_TOKEN_HEADER = "X-Jupyter-Edit-Lease"
#: Header carrying the generation the client edited against.
LEASE_GENERATION_HEADER = "X-Jupyter-Edit-Generation"
#: Header carrying the last observed lease write version (idempotency key).
LEASE_VERSION_HEADER = "X-Jupyter-Edit-Version"


def _parse_int_header(handler, name):
    """项目内部接口说明。"""
    raw = handler.request.headers.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as e:
        raise web.HTTPError(400, f"Header {name} must be an integer, got {raw!r}") from e


def credentials_from_request(handler, model=None):
    """Build :class:`LeaseCredentials` from request headers (or JSON body).

    Headers take precedence.  Absence of any lease signal returns ``None``,
    which preserves the legacy lease-free API for clients that don't opt in.
    """
    token = handler.request.headers.get(LEASE_TOKEN_HEADER)
    if token is None and model:
        token = model.get("lease_token")
    if not token:
        return None
    generation = _parse_int_header(handler, LEASE_GENERATION_HEADER)
    if generation is None and model:
        generation = model.get("lease_generation")
    lease_version = _parse_int_header(handler, LEASE_VERSION_HEADER)
    if lease_version is None and model:
        lease_version = model.get("lease_version")
    return LeaseCredentials.from_parts(token, generation=generation, lease_version=lease_version)


def _validate_keys(expect_defined: bool, model: dict[str, Any], keys: list[str]):
    """项目内部接口说明。"""

    if expect_defined:
        errors = [key for key in keys if model[key] is None]
        if errors:
            raise web.HTTPError(
                500,
                f"Keys unexpectedly None: {errors}",
            )
    else:
        errors = {key: model[key] for key in keys if model[key] is not None}  # type: ignore[assignment]
        if errors:
            raise web.HTTPError(
                500,
                f"Keys unexpectedly not None: {errors}",
            )


def validate_model(model, expect_content=False, expect_hash=False):
    """项目内部接口说明。"""
    required_keys = {
        "name",
        "path",
        "type",
        "writable",
        "created",
        "last_modified",
        "mimetype",
        "content",
        "format",
    }
    if expect_hash:
        required_keys.update(["hash", "hash_algorithm"])
    missing = required_keys - set(model.keys())
    if missing:
        raise web.HTTPError(
            500,
            f"Missing Model Keys: {missing}",
        )

    content_keys = ["content", "format"]
    _validate_keys(expect_content, model, content_keys)
    if expect_hash:
        _validate_keys(expect_hash, model, ["hash", "hash_algorithm"])


class ContentsAPIHandler(APIHandler):
    """项目内部接口说明。"""

    auth_resource = AUTH_RESOURCE

    def write_error(self, status_code, **kwargs):
        """Serialize lease conflicts with structured recovery information."""
        exc_info = kwargs.get("exc_info")
        if exc_info and isinstance(exc_info[1], LeaseHTTPError):
            self.set_header("Content-Type", "application/json")
            payload = dict(exc_info[1].lease_conflict)
            payload.setdefault("status", status_code)
            self.finish(json.dumps(payload))
            return
        super().write_error(status_code, **kwargs)


def _lease_response(record):
    """Serialize a lease record for a client (includes the fencing token)."""
    data = {
        "token": record.token,
        "generation": record.generation,
        "lease_version": record.lease_version,
        "owner": record.owner,
    }
    if record.owner_label is not None:
        data["owner_label"] = record.owner_label
    if record.takeover_reason is not None:
        data["takeover_reason"] = record.takeover_reason
    return data


class ContentsHandler(ContentsAPIHandler):
    """项目内部接口说明。"""

    def location_url(self, path):
        """项目内部接口说明。"""
        return url_path_join(self.base_url, "api", "contents", url_escape(path))

    def _finish_model(self, model, location=True):
        """项目内部接口说明。"""
        if location:
            location = self.location_url(model["path"])
            self.set_header("Location", location)
        self.set_header("Last-Modified", model["last_modified"])
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(model, default=json_default))

    async def _finish_error(self, code, message):
        """项目内部接口说明。"""
        self.set_status(code)
        self.write(message)
        await self.finish()

    @web.authenticated
    @authorized
    async def get(self, path=""):
        """项目内部接口说明。"""
        path = path or ""
        cm = self.contents_manager

        type = self.get_query_argument("type", default=None)
        if type not in {None, "directory", "file", "notebook"}:
            # fall back to file if unknown type
            type = "file"

        format = self.get_query_argument("format", default=None)
        if format not in {None, "text", "base64"}:
            raise web.HTTPError(400, "Format %r is invalid" % format)
        content_str = self.get_query_argument("content", default="1")
        if content_str not in {"0", "1"}:
            raise web.HTTPError(400, "Content %r is invalid" % content_str)
        content = int(content_str or "")

        hash_str = self.get_query_argument("hash", default="0")
        if hash_str not in {"0", "1"}:
            raise web.HTTPError(
                400, f"Hash argument {hash_str!r} is invalid. It must be '0' or '1'."
            )
        require_hash = int(hash_str)

        if not cm.allow_hidden and await ensure_async(cm.is_hidden(path)):
            await self._finish_error(
                HTTPStatus.NOT_FOUND, f"file or directory {path!r} does not exist"
            )
            return

        try:
            expect_hash = require_hash
            try:
                model = await ensure_async(
                    self.contents_manager.get(
                        path=path,
                        type=type,
                        format=format,
                        content=content,
                        require_hash=require_hash,
                    )
                )
            except TypeError:
                # Fallback for ContentsManager not handling the require_hash argument
                # introduced in 2.11
                expect_hash = False
                model = await ensure_async(
                    self.contents_manager.get(
                        path=path,
                        type=type,
                        format=format,
                        content=content,
                    )
                )
            validate_model(model, expect_content=content, expect_hash=expect_hash)
            info = cm.lease_model(path)
            if info is not None:
                # The fencing token itself is never exposed on a content read.
                info.pop("token", None)
                model["lease"] = info
            self._finish_model(model, location=False)
        except web.HTTPError as exc:
            # 404 is okay in this context, catch exception and return 404 code to prevent stack trace on client
            if exc.status_code == HTTPStatus.NOT_FOUND:
                await self._finish_error(
                    HTTPStatus.NOT_FOUND, f"file or directory {path!r} does not exist"
                )
            raise

    @web.authenticated
    @authorized
    async def patch(self, path=""):
        """项目内部接口说明。"""
        cm = self.contents_manager
        model = self.get_json_body()
        if model is None:
            raise web.HTTPError(400, "JSON body missing")

        credentials = credentials_from_request(self, model)

        old_path = model.get("path")
        if (
            old_path
            and not cm.allow_hidden
            and (
                await ensure_async(cm.is_hidden(path)) or await ensure_async(cm.is_hidden(old_path))
            )
        ):
            raise web.HTTPError(400, f"Cannot rename file or directory {path!r}")

        model = await ensure_async(cm.update(model, path, credentials=credentials))
        validate_model(model)
        self._finish_model(model)

    async def _copy(self, copy_from, copy_to=None):
        """项目内部接口说明。"""
        self.log.info(
            "Copying %r to %r",
            copy_from,
            copy_to or "",
        )
        model = await ensure_async(self.contents_manager.copy(copy_from, copy_to))
        self.set_status(201)
        validate_model(model)
        self._finish_model(model)

    async def _upload(self, model, path, credentials=None):
        """项目内部接口说明。"""
        self.log.info("Uploading file to %s", path)
        model = await ensure_async(self.contents_manager.new(model, path, credentials=credentials))
        self.set_status(201)
        validate_model(model)
        self._finish_model(model)

    async def _new_untitled(self, path, type="", ext=""):
        """项目内部接口说明。"""
        self.log.info("Creating new %s in %s", type or "file", path)
        model = await ensure_async(
            self.contents_manager.new_untitled(path=path, type=type, ext=ext)
        )
        self.set_status(201)
        validate_model(model)
        self._finish_model(model)

    async def _save(self, model, path, credentials=None):
        """项目内部接口说明。"""
        chunk = model.get("chunk", None)
        if not chunk or chunk == -1:  # Avoid tedious log information
            self.log.info("Saving file at %s", path)
        model = await ensure_async(self.contents_manager.save(model, path, credentials=credentials))
        validate_model(model)
        self._finish_model(model)

    @web.authenticated
    @authorized
    async def post(self, path=""):
        """项目内部接口说明。"""

        cm = self.contents_manager

        file_exists = await ensure_async(cm.file_exists(path))
        if file_exists:
            raise web.HTTPError(400, "Cannot POST to files, use PUT instead.")

        model = self.get_json_body()
        if model:
            copy_from = model.get("copy_from")
            if copy_from:
                if not cm.allow_hidden and (
                    await ensure_async(cm.is_hidden(path))
                    or await ensure_async(cm.is_hidden(copy_from))
                ):
                    raise web.HTTPError(400, f"Cannot copy file or directory {path!r}")
                else:
                    await self._copy(copy_from, path)
            else:
                ext = model.get("ext", "")
                type = model.get("type", "")
                if type not in {None, "", "directory", "file", "notebook"}:
                    # fall back to file if unknown type
                    type = "file"
                await self._new_untitled(path, type=type, ext=ext)
        else:
            await self._new_untitled(path)

    @web.authenticated
    @authorized
    async def put(self, path=""):
        """项目内部接口说明。"""
        model = self.get_json_body()
        cm = self.contents_manager

        credentials = credentials_from_request(self, model)

        if model:
            if model.get("copy_from"):
                raise web.HTTPError(400, "Cannot copy with PUT, only POST")
            if not cm.allow_hidden and (
                (model.get("path") and await ensure_async(cm.is_hidden(model.get("path"))))
                or await ensure_async(cm.is_hidden(path))
            ):
                raise web.HTTPError(400, f"Cannot create file or directory {path!r}")

            exists = await ensure_async(self.contents_manager.file_exists(path))
            if model.get("type", "") not in {None, "", "directory", "file", "notebook"}:
                # fall back to file if unknown type
                model["type"] = "file"
            if exists:
                await self._save(model, path, credentials=credentials)
            else:
                await self._upload(model, path, credentials=credentials)
        else:
            await self._new_untitled(path)

    @web.authenticated
    @authorized
    async def delete(self, path=""):
        """项目内部接口说明。"""
        cm = self.contents_manager

        credentials = credentials_from_request(self)

        if not cm.allow_hidden and await ensure_async(cm.is_hidden(path)):
            raise web.HTTPError(400, f"Cannot delete file or directory {path!r}")

        self.log.warning("delete %s", path)
        await ensure_async(cm.delete(path, credentials=credentials))
        self.set_status(204)
        self.finish()


class EditLeaseHandler(ContentsAPIHandler):
    """Manage the fencing-token edit lease of one document.

    ``POST /api/contents/<path>/lease``
        open a lease (body: ``{"known_generation": N?, "owner_label": ...?}``)
    ``PUT`` on the same URL
        renew (headers carry the token + generation)
    ``DELETE`` on the same URL
        release (close the document)
    ``POST .../lease/takeover``
        administrative takeover (body must include a non-empty ``reason``)
    """

    def _owner(self):
        user = self.current_user
        return getattr(user, "username", None) or "unknown"

    def _owner_label(self, body):
        return (body or {}).get("owner_label")

    @web.authenticated
    @authorized
    async def post(self, path=""):
        """Open a lease."""
        cm = self.contents_manager
        body = self.get_json_body() or {}
        known_generation = body.get("known_generation")
        if known_generation is not None:
            try:
                known_generation = int(known_generation)
            except (TypeError, ValueError) as e:
                raise web.HTTPError(400, "known_generation must be an integer") from e
        record = await ensure_async(
            cm.open_lease(
                path,
                self._owner(),
                known_generation=known_generation,
                owner_label=self._owner_label(body),
            )
        )
        self.set_status(201)
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(_lease_response(record)))

    @web.authenticated
    @authorized
    async def put(self, path=""):
        """Renew a lease. Token/generation travel in headers or JSON body."""
        cm = self.contents_manager
        body = self.get_json_body()
        credentials = credentials_from_request(self, body)
        if credentials is None:
            raise web.HTTPError(400, "Lease token required for renewal")
        record = await ensure_async(cm.renew_lease(path, credentials))
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(_lease_response(record)))

    @web.authenticated
    @authorized
    async def delete(self, path=""):
        """Release a lease."""
        cm = self.contents_manager
        credentials = credentials_from_request(self)
        if credentials is None:
            raise web.HTTPError(400, "Lease token required for release")
        await ensure_async(cm.release_lease(path, credentials))
        self.set_status(204)
        self.finish()


class EditLeaseTakeoverHandler(ContentsAPIHandler):
    """Administrative takeover of a document's edit lease."""

    @web.authenticated
    @authorized(action="admin")
    async def post(self, path=""):
        """Revoke the existing lease and take it over.

        A non-empty ``reason`` is mandatory; it is recorded against the dead
        token so the evicted owner can see why it was revoked.
        """
        cm = self.contents_manager
        body = self.get_json_body() or {}
        reason = body.get("reason")
        if not reason or not str(reason).strip():
            raise web.HTTPError(400, "A non-empty 'reason' is required for takeover")
        user = self.current_user
        admin = getattr(user, "username", None) or "unknown"
        record = await ensure_async(
            cm.takeover_lease(
                path,
                admin=admin,
                reason=str(reason),
                owner_label=body.get("owner_label"),
            )
        )
        self.set_status(201)
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(_lease_response(record)))


class CheckpointsHandler(ContentsAPIHandler):
    """项目内部接口说明。"""

    @web.authenticated
    @authorized
    async def get(self, path=""):
        """项目内部接口说明。"""
        cm = self.contents_manager
        checkpoints = await ensure_async(cm.list_checkpoints(path))
        data = json.dumps(checkpoints, default=json_default)
        self.finish(data)

    @web.authenticated
    @authorized
    async def post(self, path=""):
        """项目内部接口说明。"""
        cm = self.contents_manager
        checkpoint = await ensure_async(cm.create_checkpoint(path))
        data = json.dumps(checkpoint, default=json_default)
        location = url_path_join(
            self.base_url,
            "api/contents",
            url_escape(path),
            "checkpoints",
            url_escape(checkpoint["id"]),
        )
        self.set_header("Location", location)
        self.set_status(201)
        self.finish(data)


class ModifyCheckpointsHandler(ContentsAPIHandler):
    """项目内部接口说明。"""

    @web.authenticated
    @authorized
    async def post(self, path, checkpoint_id):
        """项目内部接口说明。"""
        cm = self.contents_manager
        credentials = credentials_from_request(self)
        await ensure_async(cm.restore_checkpoint(checkpoint_id, path, credentials=credentials))
        self.set_status(204)
        self.finish()

    @web.authenticated
    @authorized
    async def delete(self, path, checkpoint_id):
        """项目内部接口说明。"""
        cm = self.contents_manager
        await ensure_async(cm.delete_checkpoint(checkpoint_id, path))
        self.set_status(204)
        self.finish()


class NotebooksRedirectHandler(JupyterHandler):
    """项目内部接口说明。"""

    SUPPORTED_METHODS = (
        "GET",
        "PUT",
        "PATCH",
        "POST",
        "DELETE",
    )

    @allow_unauthenticated
    def get(self, path):
        """项目内部接口说明。"""
        self.log.warning("/api/notebooks is deprecated, use /api/contents")
        self.redirect(url_path_join(self.base_url, "api/contents", url_escape(path)))

    put = patch = post = delete = get


class TrustNotebooksHandler(JupyterHandler):
    """项目内部接口说明。"""

    @web.authenticated
    @authorized(resource=AUTH_RESOURCE)
    async def post(self, path=""):
        """项目内部接口说明。"""
        cm = self.contents_manager
        await ensure_async(cm.trust_notebook(path))
        self.set_status(201)
        self.finish()


# -----------------------------------------------------------------------------
# URL to handler mappings
# -----------------------------------------------------------------------------


_checkpoint_id_regex = r"(?P<checkpoint_id>[\w-]+)"

#: An API path of at least one segment (same shape as ``path_regex``) used to
#: scope the trailing ``/lease`` resource.
_lease_path_regex = r"(?P<path>(?:/[^/]+)+)"


default_handlers = [
    (
        rf"/api/contents{_lease_path_regex}/lease/takeover",
        EditLeaseTakeoverHandler,
    ),
    (rf"/api/contents{_lease_path_regex}/lease", EditLeaseHandler),
    (r"/api/contents%s/checkpoints" % path_regex, CheckpointsHandler),
    (
        rf"/api/contents{path_regex}/checkpoints/{_checkpoint_id_regex}",
        ModifyCheckpointsHandler,
    ),
    (r"/api/contents%s/trust" % path_regex, TrustNotebooksHandler),
    (r"/api/contents%s" % path_regex, ContentsHandler),
    (r"/api/notebooks/?(.*)", NotebooksRedirectHandler),
]

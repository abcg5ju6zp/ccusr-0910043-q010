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

from .leases import LeaseConflictError, supports_lease_kwarg

AUTH_RESOURCE = "contents"


def _call_with_lease(method, *args, lease, **kwargs):
    """项目内部接口说明。"""
    if lease is None:
        return method(*args, **kwargs)
    if not supports_lease_kwarg(method):
        raise web.HTTPError(400, "The configured contents manager does not support edit leases")
    return method(*args, lease=lease, **kwargs)


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

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        """项目内部接口说明。"""
        exc_info = kwargs.get("exc_info")
        if exc_info and isinstance(exc_info[1], LeaseConflictError):
            error = exc_info[1]
            self.set_header("Content-Type", "application/json")
            self.finish(
                json.dumps(
                    {
                        "message": error.log_message or "Edit lease conflict",
                        "reason": error.reason,
                        "lease_conflict": error.conflict,
                    },
                    default=json_default,
                )
            )
            return
        super().write_error(status_code, **kwargs)

    def _lease_from_request(self, model: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """项目内部接口说明。"""
        lease: dict[str, Any] = {}
        if model:
            body_lease = model.pop("lease", None)
            if body_lease is not None:
                if not isinstance(body_lease, dict):
                    raise web.HTTPError(400, "lease must be an object")
                lease.update(body_lease)
        headers = self.request.headers
        for header, key in (
            ("X-Jupyter-Lease-Id", "lease_id"),
            ("X-Jupyter-Lease-Generation", "generation"),
            ("X-Jupyter-Request-Id", "request_id"),
        ):
            value = headers.get(header)
            if value:
                lease[key] = value
        for argument, key in (
            ("lease_id", "lease_id"),
            ("lease_generation", "generation"),
            ("request_id", "request_id"),
        ):
            value = self.get_query_argument(argument, None)
            if value is not None:
                lease[key] = value
        if not lease:
            return None
        if "generation" in lease:
            try:
                lease["generation"] = int(lease["generation"])
            except (TypeError, ValueError):
                raise web.HTTPError(400, "Lease generation must be an integer") from None
        return lease

    def _lease_actor(self) -> Any:
        """项目内部接口说明。"""
        user = self.current_user
        if user is None:
            return None
        username = getattr(user, "username", None)
        if username:
            return username
        if isinstance(user, dict):
            return user.get("username")
        if isinstance(user, str):
            return user
        return None


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

        old_path = model.get("path")
        if (
            old_path
            and not cm.allow_hidden
            and (
                await ensure_async(cm.is_hidden(path)) or await ensure_async(cm.is_hidden(old_path))
            )
        ):
            raise web.HTTPError(400, f"Cannot rename file or directory {path!r}")

        lease = self._lease_from_request(model)
        model = await ensure_async(_call_with_lease(cm.update, model, path, lease=lease))
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

    async def _upload(self, model, path, lease=None):
        """项目内部接口说明。"""
        self.log.info("Uploading file to %s", path)
        model = await ensure_async(
            _call_with_lease(self.contents_manager.new, model, path, lease=lease)
        )
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

    async def _save(self, model, path, lease=None):
        """项目内部接口说明。"""
        chunk = model.get("chunk", None)
        if not chunk or chunk == -1:  # Avoid tedious log information
            self.log.info("Saving file at %s", path)
        if lease is None:
            lease = self._lease_from_request(model)
        model = await ensure_async(
            _call_with_lease(self.contents_manager.save, model, path, lease=lease)
        )
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
            lease = self._lease_from_request(model)
            if exists:
                await self._save(model, path, lease=lease)
            else:
                await self._upload(model, path, lease=lease)
        else:
            await self._new_untitled(path)

    @web.authenticated
    @authorized
    async def delete(self, path=""):
        """项目内部接口说明。"""
        cm = self.contents_manager

        if not cm.allow_hidden and await ensure_async(cm.is_hidden(path)):
            raise web.HTTPError(400, f"Cannot delete file or directory {path!r}")

        self.log.warning("delete %s", path)
        lease = self._lease_from_request()
        await ensure_async(_call_with_lease(cm.delete, path, lease=lease))
        self.set_status(204)
        self.finish()


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
        lease = self._lease_from_request()
        checkpoint = await ensure_async(_call_with_lease(cm.create_checkpoint, path, lease=lease))
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
        lease = self._lease_from_request()
        await ensure_async(
            _call_with_lease(cm.restore_checkpoint, checkpoint_id, path, lease=lease)
        )
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


class LeaseHandler(ContentsAPIHandler):
    """项目内部接口说明。"""

    async def _check_hidden(self, path):
        """项目内部接口说明。"""
        cm = self.contents_manager
        if not cm.allow_hidden and await ensure_async(cm.is_hidden(path)):
            raise web.HTTPError(404, f"file or directory {path!r} does not exist")

    @web.authenticated
    @authorized
    async def get(self, path=""):
        """项目内部接口说明。"""
        await self._check_hidden(path)
        cm = self.contents_manager
        view = await ensure_async(cm.inspect_lease(path))
        self.finish(json.dumps(view, default=json_default))

    @web.authenticated
    @authorized
    async def post(self, path=""):
        """项目内部接口说明。"""
        await self._check_hidden(path)
        cm = self.contents_manager
        body = self.get_json_body() or {}
        holder = body.get("holder")
        if not isinstance(holder, str) or not holder.strip():
            raise web.HTTPError(400, "A lease holder is required")
        lease, created = await ensure_async(cm.open_lease(path, holder=holder, ttl=body.get("ttl")))
        self.set_status(201 if created else 200)
        self.finish(json.dumps(lease, default=json_default))

    @web.authenticated
    @authorized
    async def put(self, path=""):
        """项目内部接口说明。"""
        await self._check_hidden(path)
        cm = self.contents_manager
        body = self.get_json_body()
        if body is None:
            raise web.HTTPError(400, "JSON body missing")
        lease_id = body.get("lease_id")
        generation = body.get("generation")
        if not lease_id or generation is None:
            raise web.HTTPError(400, "lease_id and generation are required")
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            raise web.HTTPError(400, "generation must be an integer") from None
        lease = await ensure_async(cm.renew_lease(path, lease_id, generation, ttl=body.get("ttl")))
        self.finish(json.dumps(lease, default=json_default))

    @web.authenticated
    @authorized
    async def delete(self, path=""):
        """项目内部接口说明。"""
        await self._check_hidden(path)
        cm = self.contents_manager
        body = self.get_json_body() or {}
        lease_id = body.get("lease_id") or self.get_query_argument("lease_id", None)
        generation = body.get("generation")
        if generation is None:
            generation = self.get_query_argument("lease_generation", None)
        if not lease_id or generation is None:
            raise web.HTTPError(400, "lease_id and generation are required")
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            raise web.HTTPError(400, "generation must be an integer") from None
        await ensure_async(cm.release_lease(path, lease_id, generation))
        self.set_status(204)
        self.finish()


class LeaseTakeoverHandler(ContentsAPIHandler):
    """项目内部接口说明。"""

    @web.authenticated
    @authorized
    async def post(self, path=""):
        """项目内部接口说明。"""
        cm = self.contents_manager
        if not cm.allow_hidden and await ensure_async(cm.is_hidden(path)):
            raise web.HTTPError(404, f"file or directory {path!r} does not exist")
        body = self.get_json_body()
        if body is None:
            raise web.HTTPError(400, "JSON body missing")
        holder = body.get("holder")
        reason = body.get("reason")
        if not isinstance(holder, str) or not holder.strip():
            raise web.HTTPError(400, "A lease holder is required")
        if not isinstance(reason, str) or not reason.strip():
            raise web.HTTPError(400, "A takeover reason is required")
        lease = await ensure_async(
            cm.takeover_lease(
                path,
                holder=holder,
                reason=reason,
                ttl=body.get("ttl"),
                by=self._lease_actor(),
            )
        )
        self.set_status(201)
        self.finish(json.dumps(lease, default=json_default))


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


default_handlers = [
    (r"/api/contents%s/checkpoints" % path_regex, CheckpointsHandler),
    (
        rf"/api/contents{path_regex}/checkpoints/{_checkpoint_id_regex}",
        ModifyCheckpointsHandler,
    ),
    (r"/api/contents%s/lease/takeover" % path_regex, LeaseTakeoverHandler),
    (r"/api/contents%s/lease" % path_regex, LeaseHandler),
    (r"/api/contents%s/trust" % path_regex, TrustNotebooksHandler),
    (r"/api/contents%s" % path_regex, ContentsHandler),
    (r"/api/notebooks/?(.*)", NotebooksRedirectHandler),
]

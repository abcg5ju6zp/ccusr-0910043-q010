"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import itertools
import json
import os
import re
import typing as t
import warnings
from fnmatch import fnmatch

from jupyter_core.utils import ensure_async, run_sync
from jupyter_events import EventLogger
from nbformat import ValidationError, sign
from nbformat import validate as validate_nb
from nbformat.v4 import new_notebook
from tornado.web import HTTPError, RequestHandler
from traitlets import (
    Any,
    Bool,
    Dict,
    Float,
    Instance,
    List,
    TraitError,
    Type,
    Unicode,
    default,
    validate,
)
from traitlets.config.configurable import LoggingConfigurable

from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
from jupyter_server.transutils import _i18n
from jupyter_server.utils import import_item

from ...files.handlers import FilesHandler
from .checkpoints import AsyncCheckpoints, Checkpoints
from .leases import LeaseConflictReason, LeaseCredentials, LeaseError, LeaseStore

copy_pat = re.compile(r"\-Copy\d*\.")


class LeaseHTTPError(HTTPError):
    """An HTTP 409 carrying structured, recoverable lease-conflict info.

    Handlers serialize :attr:`lease_conflict` as JSON so callers can decide
    how to recover (re-open, merge, refresh) instead of treating the response
    as an opaque save failure.
    """

    def __init__(self, conflict, status_code: int = 409):
        self.lease_conflict = conflict.to_dict()
        super().__init__(
            status_code,
            conflict.message,
            reason="lease_conflict",
        )


class ContentsManager(LoggingConfigurable):
    """项目内部接口说明。"""

    event_schema_id = JUPYTER_SERVER_EVENTS_URI + "/contents_service/v1"
    event_logger = Instance(EventLogger).tag(config=True)

    @default("event_logger")
    def _default_event_logger(self):
        if self.parent and hasattr(self.parent, "event_logger"):
            return self.parent.event_logger
        else:
            # If parent does not have an event logger, create one.
            logger = EventLogger()
            schema_path = DEFAULT_EVENTS_SCHEMA_PATH / "contents_service" / "v1.yaml"
            logger.register_event_schema(schema_path)
            return logger

    def emit(self, data):
        """项目内部接口说明。"""
        self.event_logger.emit(schema_id=self.event_schema_id, data=data)

    # -- Edit leases (fencing tokens) --------------------------------------

    lease_ttl_seconds = Float(
        300.0,
        config=True,
        help="""Time after which an idle edit lease expires. Clients are
        expected to renew before this deadline; a clock-skew grace is added
        server-side so a slightly slow client never loses a live lease.""",
    )

    lease_clock_skew_grace_seconds = Float(
        30.0,
        config=True,
        help="Extra lifetime granted past a lease deadline to tolerate client/server clock drift.",
    )

    lease_store = Instance(LeaseStore)

    @default("lease_store")
    def _default_lease_store(self):
        return LeaseStore(
            ttl_seconds=self.lease_ttl_seconds,
            clock_skew_grace_seconds=self.lease_clock_skew_grace_seconds,
        )

    def _owner_name(self, owner: str | None = None) -> str:
        if owner:
            return owner
        # Filled in by handlers via the keyword argument; fall back keeps the
        # manager usable when driven directly (tests, other services).
        return "unknown"

    def open_lease(self, path, owner, *, known_generation=None, owner_label=None):
        """Open an edit lease (fencing token) on ``path``."""
        path = path.strip("/")
        try:
            record = self.lease_store.acquire(
                path,
                self._owner_name(owner),
                known_generation=known_generation,
                owner_label=owner_label,
            )
            key = self._lease_fingerprint(path)
            if key is not None:
                self.lease_store.set_baseline(path, record.token, key)
            return record
        except LeaseError as e:
            raise LeaseHTTPError(e.conflict) from e

    def renew_lease(self, path, credentials):
        """Renew a lease without changing its token or generation."""
        try:
            return self.lease_store.renew(
                path.strip("/"),
                credentials.token,
                expected_generation=credentials.generation,
            )
        except LeaseError as e:
            raise LeaseHTTPError(e.conflict) from e

    def release_lease(self, path, credentials):
        """Close a lease. Idempotent for already-dead tokens."""
        try:
            self.lease_store.release(path.strip("/"), credentials.token)
        except LeaseError as e:
            raise LeaseHTTPError(e.conflict) from e

    def takeover_lease(self, path, *, admin, reason, owner_label=None):
        """Revoke the current lease as an administrator; reason is recorded."""
        path = path.strip("/")
        prior = self.lease_store.get(path)
        prior_owner = prior.current.owner if prior is not None and prior.current else None
        try:
            record = self.lease_store.takeover(
                path,
                admin=self._owner_name(admin),
                reason=reason,
                owner_label=owner_label,
            )
        except LeaseError as e:
            raise LeaseHTTPError(e.conflict) from e
        # Administrative actions are audited: who took over, whose lease was
        # revoked and the mandatory reason. The reason is also stored on the
        # dead token's tombstone, surfaced to the evicted client on next use.
        self.log.warning(
            "Edit lease on %r taken over by admin %r from %r; reason: %s",
            path,
            record.owner,
            prior_owner,
            reason.strip(),
        )
        # The new owner inherits the file as it is on disk right now;
        # any earlier out-of-band state becomes its baseline.
        key = self._lease_fingerprint(path)
        if key is not None:
            self.lease_store.set_baseline(path, record.token, key)
        return record

    def lease_model(self, path):
        """Public lease state for a contents model (never exposes other tokens)."""
        return self.lease_store.model_for(path.strip("/"))

    def _check_lease(self, path, credentials):
        """Validate credentials for a mutating op; raises 409 on failure."""
        try:
            return self.lease_store.check(
                path.strip("/"),
                credentials.token if credentials is not None else None,
                expected_generation=credentials.generation if credentials is not None else None,
                permit_when_absent=credentials is None,
            )
        except LeaseError as e:
            raise LeaseHTTPError(e.conflict) from e

    def _lease_fingerprint(self, path):
        """Opaque fingerprint of the stored document.

        Subclasses override this to detect out-of-band modifications (e.g.
        ``(mtime_ns, size)`` for local files).  Returning ``None`` disables
        external-modification detection for a manager.
        """
        return None

    def _attach_lease_info(self, model, path, record=None):
        """Expose non-secret lease state on a returned contents model."""
        if model is None:
            return model
        info = self.lease_store.model_for(path.strip("/"))
        if info is not None:
            # Never leak the fencing token through a contents model; it only
            # travels in dedicated lease responses/headers.
            info.pop("token", None)
            model["lease"] = info
        return model

    root_dir = Unicode("/", config=True)

    preferred_dir = Unicode(
        "",
        config=True,
        help=_i18n(
            "Preferred starting directory to use for notebooks. This is an API path (`/` separated, relative to root dir)"
        ),
    )

    @validate("preferred_dir")
    def _validate_preferred_dir(self, proposal):
        value = proposal["value"].strip("/")
        try:
            import inspect

            if inspect.iscoroutinefunction(self.dir_exists):
                dir_exists = run_sync(self.dir_exists)(value)
            else:
                dir_exists = self.dir_exists(value)
        except HTTPError as e:
            raise TraitError(e.log_message) from e
        if not dir_exists:
            raise TraitError(_i18n("Preferred directory not found: %r") % value)
        if self.parent:
            try:
                if value != self.parent.preferred_dir:
                    self.parent.preferred_dir = os.path.join(self.root_dir, *value.split("/"))
            except TraitError:
                pass
        return value

    allow_hidden = Bool(False, config=True, help="Allow access to hidden files")

    notary = Instance(sign.NotebookNotary)

    @default("notary")
    def _notary_default(self):
        return sign.NotebookNotary(parent=self)

    hide_globs = List(
        Unicode(),
        [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "*~",
        ],
        config=True,
        help="""
        Glob patterns to hide in file and directory listings.
    """,
    )

    untitled_notebook = Unicode(
        _i18n("Untitled"),
        config=True,
        help="The base name used when creating untitled notebooks.",
    )

    untitled_file = Unicode(
        "untitled", config=True, help="The base name used when creating untitled files."
    )

    untitled_directory = Unicode(
        "Untitled Folder",
        config=True,
        help="The base name used when creating untitled directories.",
    )

    pre_save_hook = Any(
        None,
        config=True,
        allow_none=True,
        help="""Python callable or importstring thereof

        To be called on a contents model prior to save.

        This can be used to process the structure,
        such as removing notebook outputs or other side effects that
        should not be saved.

        It will be called as (all arguments passed by keyword)::

            hook(path=path, model=model, contents_manager=self)

        - model: the model to be saved. Includes file contents.
          Modifying this dict will affect the file that is stored.
        - path: the API path of the save destination
        - contents_manager: this ContentsManager instance
        """,
    )

    @validate("pre_save_hook")
    def _validate_pre_save_hook(self, proposal):
        value = proposal["value"]
        if isinstance(value, str):
            value = import_item(self.pre_save_hook)
        if not callable(value):
            msg = "pre_save_hook must be callable"
            raise TraitError(msg)
        if callable(self.pre_save_hook):
            warnings.warn(
                f"Overriding existing pre_save_hook ({self.pre_save_hook.__name__}) with a new one ({value.__name__}).",
                stacklevel=2,
            )
        return value

    post_save_hook = Any(
        None,
        config=True,
        allow_none=True,
        help="""Python callable or importstring thereof

        to be called on the path of a file just saved.

        This can be used to process the file on disk,
        such as converting the notebook to a script or HTML via nbconvert.

        It will be called as (all arguments passed by keyword)::

            hook(os_path=os_path, model=model, contents_manager=instance)

        - path: the filesystem path to the file just written
        - model: the model representing the file
        - contents_manager: this ContentsManager instance
        """,
    )

    @validate("post_save_hook")
    def _validate_post_save_hook(self, proposal):
        value = proposal["value"]
        if isinstance(value, str):
            value = import_item(value)
        if not callable(value):
            msg = "post_save_hook must be callable"
            raise TraitError(msg)
        if callable(self.post_save_hook):
            warnings.warn(
                f"Overriding existing post_save_hook ({self.post_save_hook.__name__}) with a new one ({value.__name__}).",
                stacklevel=2,
            )
        return value

    def run_pre_save_hook(self, model, path, **kwargs):
        """项目内部接口说明。"""
        warnings.warn(
            "run_pre_save_hook is deprecated, use run_pre_save_hooks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.pre_save_hook:
            try:
                self.log.debug("Running pre-save hook on %s", path)
                self.pre_save_hook(model=model, path=path, contents_manager=self, **kwargs)
            except HTTPError:
                # allow custom HTTPErrors to raise,
                # rejecting the save with a message.
                raise
            except Exception:
                # unhandled errors don't prevent saving,
                # which could cause frustrating data loss
                self.log.error("Pre-save hook failed on %s", path, exc_info=True)

    def run_post_save_hook(self, model, os_path):
        """项目内部接口说明。"""
        warnings.warn(
            "run_post_save_hook is deprecated, use run_post_save_hooks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.post_save_hook:
            try:
                self.log.debug("Running post-save hook on %s", os_path)
                self.post_save_hook(os_path=os_path, model=model, contents_manager=self)
            except Exception:
                self.log.error("Post-save hook failed o-n %s", os_path, exc_info=True)
                msg = "fUnexpected error while running post hook save: {e}"
                raise HTTPError(500, msg) from None

    _pre_save_hooks: List[t.Any] = List()
    _post_save_hooks: List[t.Any] = List()

    def register_pre_save_hook(self, hook):
        """项目内部接口说明。"""
        if isinstance(hook, str):
            hook = import_item(hook)
        if not callable(hook):
            msg = "hook must be callable"
            raise RuntimeError(msg)
        self._pre_save_hooks.append(hook)

    def register_post_save_hook(self, hook):
        """项目内部接口说明。"""
        if isinstance(hook, str):
            hook = import_item(hook)
        if not callable(hook):
            msg = "hook must be callable"
            raise RuntimeError(msg)
        self._post_save_hooks.append(hook)

    def run_pre_save_hooks(self, model, path, **kwargs):
        """项目内部接口说明。"""
        pre_save_hooks = [self.pre_save_hook] if self.pre_save_hook is not None else []
        pre_save_hooks += self._pre_save_hooks
        for pre_save_hook in pre_save_hooks:
            try:
                self.log.debug("Running pre-save hook on %s", path)
                pre_save_hook(model=model, path=path, contents_manager=self, **kwargs)
            except HTTPError:
                # allow custom HTTPErrors to raise,
                # rejecting the save with a message.
                raise
            except Exception:
                # unhandled errors don't prevent saving,
                # which could cause frustrating data loss
                self.log.error(
                    "Pre-save hook %s failed on %s",
                    pre_save_hook.__name__,
                    path,
                    exc_info=True,
                )

    def run_post_save_hooks(self, model, os_path):
        """项目内部接口说明。"""
        post_save_hooks = [self.post_save_hook] if self.post_save_hook is not None else []
        post_save_hooks += self._post_save_hooks
        for post_save_hook in post_save_hooks:
            try:
                self.log.debug("Running post-save hook on %s", os_path)
                post_save_hook(os_path=os_path, model=model, contents_manager=self)
            except Exception as e:
                self.log.error(
                    "Post-save %s hook failed on %s",
                    post_save_hook.__name__,
                    os_path,
                    exc_info=True,
                )
                raise HTTPError(500, "Unexpected error while running post hook save: %s" % e) from e

    checkpoints_class = Type(Checkpoints, config=True)
    checkpoints = Instance(Checkpoints, config=True)
    checkpoints_kwargs = Dict(config=True)

    @default("checkpoints")
    def _default_checkpoints(self):
        return self.checkpoints_class(**self.checkpoints_kwargs)

    @default("checkpoints_kwargs")
    def _default_checkpoints_kwargs(self):
        return {
            "parent": self,
            "log": self.log,
        }

    files_handler_class = Type(
        FilesHandler,
        klass=RequestHandler,
        allow_none=True,
        config=True,
        help="""handler class to use when serving raw file requests.

        Default is a fallback that talks to the ContentsManager API,
        which may be inefficient, especially for large files.

        Local files-based ContentsManagers can use a StaticFileHandler subclass,
        which will be much more efficient.

        Access to these files should be Authenticated.
        """,
    )

    files_handler_params = Dict(
        config=True,
        help="""Extra parameters to pass to files_handler_class.

        For example, StaticFileHandlers generally expect a `path` argument
        specifying the root directory from which to serve files.
        """,
    )

    def get_extra_handlers(self):
        """项目内部接口说明。"""
        handlers = []
        if self.files_handler_class:
            handlers.append((r"/files/(.*)", self.files_handler_class, self.files_handler_params))
        return handlers

    # ContentsManager API part 1: methods that must be
    # implemented in subclasses.

    def dir_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def is_hidden(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def file_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def exists(self, path):
        """项目内部接口说明。"""
        return self.file_exists(path) or self.dir_exists(path)

    def get(self, path, content=True, type=None, format=None, require_hash=False):
        """项目内部接口说明。"""
        raise NotImplementedError

    def save(self, model, path, credentials=None):
        """项目内部接口说明。"""
        raise NotImplementedError

    def delete_file(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def rename_file(self, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    # ContentsManager API part 2: methods that have usable default
    # implementations, but can be overridden in subclasses.

    def delete(self, path, credentials=None):
        """项目内部接口说明。"""
        path = path.strip("/")
        if not path:
            raise HTTPError(400, "Can't delete root")
        record = self._check_lease(path, credentials)
        self.delete_file(path)
        self.checkpoints.delete_all_checkpoints(path)
        if record is not None:
            # The document is gone: close the lease and kill its token so a
            # reconnecting client cannot fence a delete-then-recreate file.
            self.lease_store.release(path, record.token)
        self.emit(data={"action": "delete", "path": path})

    def rename(self, old_path, new_path, credentials=None):
        """项目内部接口说明。"""
        old_path = old_path.strip("/")
        new_path = new_path.strip("/")
        record = self._check_lease(old_path, credentials)
        try:
            # Refuse to move onto a name another client has open.
            self.lease_store.check_rename_destination(new_path)
        except LeaseError as e:
            raise LeaseHTTPError(e.conflict) from e
        self.rename_file(old_path, new_path)
        self.checkpoints.rename_all_checkpoints(old_path, new_path)
        if record is not None:
            # The lease (and all dead tokens) follows the document so the
            # client keeps fencing with the same generation at the new name.
            self.lease_store.move(old_path, new_path)
        self.emit(data={"action": "rename", "path": new_path, "source_path": old_path})

    def update(self, model, path, credentials=None):
        """项目内部接口说明。"""
        path = path.strip("/")
        new_path = model.get("path", path).strip("/")
        if path != new_path:
            self.rename(path, new_path, credentials=credentials)
        model = self.get(new_path, content=False)
        return self._attach_lease_info(model, new_path)

    def info_string(self):
        """项目内部接口说明。"""
        return "Serving contents"

    def get_kernel_path(self, path, model=None):
        """项目内部接口说明。"""
        return ""

    def increment_filename(self, filename, path="", insert=""):
        """项目内部接口说明。"""
        # Extract the full suffix from the filename (e.g. .tar.gz)
        path = path.strip("/")
        basename, dot, ext = filename.rpartition(".")
        if ext != "ipynb":
            basename, dot, ext = filename.partition(".")

        suffix = dot + ext

        for i in itertools.count():
            insert_i = f"{insert}{i}" if i else ""
            name = f"{basename}{insert_i}{suffix}"
            if not self.exists(f"{path}/{name}"):
                break
        return name

    def validate_notebook_model(self, model, validation_error=None):
        """项目内部接口说明。"""
        try:
            # If we're given a validation_error dictionary, extract the exception
            # from it and raise the exception, else call nbformat's validate method
            # to determine if the notebook is valid.  This 'else' condition may
            # pertain to server extension not using the server's notebook read/write
            # functions.
            if validation_error is not None:
                e = validation_error.get("ValidationError")
                if isinstance(e, ValidationError):
                    raise e
            else:
                validate_nb(model["content"])
        except ValidationError as e:
            model["message"] = "Notebook validation failed: {}:\n{}".format(
                str(e),
                json.dumps(e.instance, indent=1, default=lambda obj: "<UNKNOWN>"),
            )
        return model

    def new_untitled(self, path="", type="", ext=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        if not self.dir_exists(path):
            raise HTTPError(404, "No such directory: %s" % path)

        model = {}
        if type:
            model["type"] = type

        if ext == ".ipynb":
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        insert = ""
        if model["type"] == "directory":
            untitled = self.untitled_directory
            insert = " "
        elif model["type"] == "notebook":
            untitled = self.untitled_notebook
            ext = ".ipynb"
        elif model["type"] == "file":
            untitled = self.untitled_file
        else:
            raise HTTPError(400, "Unexpected model type: %r" % model["type"])

        name = self.increment_filename(untitled + ext, path, insert=insert)
        path = f"{path}/{name}"
        return self.new(model, path)

    def new(self, model=None, path="", credentials=None):
        """项目内部接口说明。"""
        path = path.strip("/")
        if model is None:
            model = {}

        if path.endswith(".ipynb"):
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        # no content, not a directory, so fill out new-file model
        if "content" not in model and model["type"] != "directory":
            if model["type"] == "notebook":
                model["content"] = new_notebook()
                model["format"] = "json"
            else:
                model["content"] = ""
                model["type"] = "file"
                model["format"] = "text"

        model = self.save(model, path, credentials=credentials)
        return model

    def copy(self, from_path, to_path=None):
        """项目内部接口说明。"""
        path = from_path.strip("/")

        if to_path is not None:
            to_path = to_path.strip("/")

        if "/" in path:
            from_dir, from_name = path.rsplit("/", 1)
        else:
            from_dir = ""
            from_name = path

        model = self.get(path)
        model.pop("path", None)
        model.pop("name", None)
        if model["type"] == "directory":
            raise HTTPError(400, "Can't copy directories")

        is_destination_specified = to_path is not None
        if not is_destination_specified:
            to_path = from_dir
        if self.dir_exists(to_path):
            name = copy_pat.sub(".", from_name)
            to_name = self.increment_filename(name, to_path, insert="-Copy")
            to_path = f"{to_path}/{to_name}"
        elif is_destination_specified:
            if "/" in to_path:
                to_dir, to_name = to_path.rsplit("/", 1)
                if not self.dir_exists(to_dir):
                    raise HTTPError(404, "No such parent directory: %s to copy file in" % to_dir)
        else:
            raise HTTPError(404, "No such directory: %s" % to_path)

        model = self.save(model, to_path)
        self.emit(data={"action": "copy", "path": to_path, "source_path": from_path})
        return model

    def log_info(self):
        """项目内部接口说明。"""
        self.log.info(self.info_string())

    def trust_notebook(self, path):
        """项目内部接口说明。"""
        model = self.get(path)
        nb = model["content"]
        self.log.warning("Trusting notebook %s", path)
        self.notary.mark_cells(nb, True)
        self.check_and_sign(nb, path)

    def check_and_sign(self, nb, path="", *, _retrying=False):
        """项目内部接口说明。"""
        try:
            if self.notary.check_cells(nb):
                self.notary.sign(nb)
            else:
                self.log.warning("Notebook %s is not trusted", path)
        except Exception:
            if _retrying:
                raise
            self.log.warning(
                "Signature store for notebook %s is corrupted or unavailable; "
                "recreating the store.",
                path,
                exc_info=True,
            )
            # The default implementation uses SQLiteSignatureStore if SQLite3 is available
            # and falls back to MemorySignatureStore if not; SQLiteSignatureStore will
            # attempt to recreate the database if it detects errors during initialization,
            # and fallback to in-memory (`:memory:`) SQLite database if necessary.
            self.notary.store = self.notary.store_factory()
            self.check_and_sign(nb, path, _retrying=True)

    def mark_trusted_cells(self, nb, path=""):
        """项目内部接口说明。"""
        trusted = self.notary.check_signature(nb)
        if not trusted:
            self.log.warning("Notebook %s is not trusted", path)
        self.notary.mark_cells(nb, trusted)

    def should_list(self, name):
        """项目内部接口说明。"""
        return not any(fnmatch(name, glob) for glob in self.hide_globs)

    # Part 3: Checkpoints API
    def _annotate_checkpoint_lease(self, model, path):
        """Associate a checkpoint with the current lease version.

        Purely informational: snapshotting never extends the lease TTL, never
        bumps the write version and never requires a token.
        """
        entry = self.lease_store.get(path.strip("/"))
        if entry is not None and entry.current is not None:
            cur = entry.current
            model["lease_generation"] = cur.generation
            model["lease_version"] = cur.lease_version
        return model

    def create_checkpoint(self, path):
        """项目内部接口说明。"""
        model = self.checkpoints.create_checkpoint(self, path)
        return self._annotate_checkpoint_lease(model, path)

    def restore_checkpoint(self, checkpoint_id, path, credentials=None):
        """项目内部接口说明。"""
        path = path.strip("/")
        backend = self.checkpoints
        uses_save = getattr(backend, "restore_uses_save", False)
        record = None
        if not uses_save:
            # Byte-copy backends do not fence themselves.
            record = self._check_lease(path, credentials)
        backend.restore_checkpoint(self, checkpoint_id, path, credentials=credentials)
        if record is not None:
            # The restored content is the new known state of the file; TTL is
            # unchanged (a restore does not extend the lease).
            self.lease_store.commit_write(path, record.token, new_key=self._lease_fingerprint(path))

    def list_checkpoints(self, path):
        return self.checkpoints.list_checkpoints(path)

    def delete_checkpoint(self, checkpoint_id, path):
        return self.checkpoints.delete_checkpoint(checkpoint_id, path)


class AsyncContentsManager(ContentsManager):
    """项目内部接口说明。"""

    checkpoints_class = Type(AsyncCheckpoints, config=True)
    checkpoints = Instance(AsyncCheckpoints, config=True)
    checkpoints_kwargs = Dict(config=True)

    @default("checkpoints")
    def _default_checkpoints(self):
        return self.checkpoints_class(**self.checkpoints_kwargs)

    @default("checkpoints_kwargs")
    def _default_checkpoints_kwargs(self):
        return {
            "parent": self,
            "log": self.log,
        }

    # ContentsManager API part 1: methods that must be
    # implemented in subclasses.

    async def dir_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def is_hidden(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def file_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def exists(self, path):
        """项目内部接口说明。"""
        return await ensure_async(self.file_exists(path)) or await ensure_async(
            self.dir_exists(path)
        )

    async def get(self, path, content=True, type=None, format=None, require_hash=False):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def save(self, model, path, credentials=None):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def delete_file(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def rename_file(self, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    # ContentsManager API part 2: methods that have usable default
    # implementations, but can be overridden in subclasses.

    async def resolve_path(self, path: str) -> str | None:
        """项目内部接口说明。"""
        return None

    async def delete(self, path, credentials=None):
        """项目内部接口说明。"""
        path = path.strip("/")
        if not path:
            raise HTTPError(400, "Can't delete root")

        record = self._check_lease(path, credentials)
        await self.delete_file(path)
        await self.checkpoints.delete_all_checkpoints(path)
        if record is not None:
            self.lease_store.release(path, record.token)
        self.emit(data={"action": "delete", "path": path})

    async def rename(self, old_path, new_path, credentials=None):
        """项目内部接口说明。"""
        old_path = old_path.strip("/")
        new_path = new_path.strip("/")
        record = self._check_lease(old_path, credentials)
        try:
            self.lease_store.check_rename_destination(new_path)
        except LeaseError as e:
            raise LeaseHTTPError(e.conflict) from e
        await self.rename_file(old_path, new_path)
        await self.checkpoints.rename_all_checkpoints(old_path, new_path)
        if record is not None:
            self.lease_store.move(old_path, new_path)
        self.emit(data={"action": "rename", "path": new_path, "source_path": old_path})

    async def update(self, model, path, credentials=None):
        """项目内部接口说明。"""
        path = path.strip("/")
        new_path = model.get("path", path).strip("/")
        if path != new_path:
            await self.rename(path, new_path, credentials=credentials)
        model = await self.get(new_path, content=False)
        return self._attach_lease_info(model, new_path)

    async def increment_filename(self, filename, path="", insert=""):
        """项目内部接口说明。"""
        # Extract the full suffix from the filename (e.g. .tar.gz)
        path = path.strip("/")
        basename, dot, ext = filename.rpartition(".")
        if ext != "ipynb":
            basename, dot, ext = filename.partition(".")

        suffix = dot + ext

        for i in itertools.count():
            insert_i = f"{insert}{i}" if i else ""
            name = f"{basename}{insert_i}{suffix}"
            file_exists = await ensure_async(self.exists(f"{path}/{name}"))
            if not file_exists:
                break
        return name

    async def new_untitled(self, path="", type="", ext=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        dir_exists = await ensure_async(self.dir_exists(path))
        if not dir_exists:
            raise HTTPError(404, "No such directory: %s" % path)

        model = {}
        if type:
            model["type"] = type

        if ext == ".ipynb":
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        insert = ""
        if model["type"] == "directory":
            untitled = self.untitled_directory
            insert = " "
        elif model["type"] == "notebook":
            untitled = self.untitled_notebook
            ext = ".ipynb"
        elif model["type"] == "file":
            untitled = self.untitled_file
        else:
            raise HTTPError(400, "Unexpected model type: %r" % model["type"])

        name = await self.increment_filename(untitled + ext, path, insert=insert)
        path = f"{path}/{name}"
        return await self.new(model, path)

    async def new(self, model=None, path="", credentials=None):
        """项目内部接口说明。"""
        path = path.strip("/")
        if model is None:
            model = {}

        if path.endswith(".ipynb"):
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        # no content, not a directory, so fill out new-file model
        if "content" not in model and model["type"] != "directory":
            if model["type"] == "notebook":
                model["content"] = new_notebook()
                model["format"] = "json"
            else:
                model["content"] = ""
                model["type"] = "file"
                model["format"] = "text"

        model = await self.save(model, path, credentials=credentials)
        return model

    async def copy(self, from_path, to_path=None):
        """项目内部接口说明。"""
        path = from_path.strip("/")

        if to_path is not None:
            to_path = to_path.strip("/")

        if "/" in path:
            from_dir, from_name = path.rsplit("/", 1)
        else:
            from_dir = ""
            from_name = path

        model = await self.get(path)
        model.pop("path", None)
        model.pop("name", None)
        if model["type"] == "directory":
            raise HTTPError(400, "Can't copy directories")

        is_destination_specified = to_path is not None
        if not is_destination_specified:
            to_path = from_dir
        if await ensure_async(self.dir_exists(to_path)):
            name = copy_pat.sub(".", from_name)
            to_name = await self.increment_filename(name, to_path, insert="-Copy")
            to_path = f"{to_path}/{to_name}"
        elif is_destination_specified:
            if "/" in to_path:
                to_dir, to_name = to_path.rsplit("/", 1)
                if not await ensure_async(self.dir_exists(to_dir)):
                    raise HTTPError(404, "No such parent directory: %s to copy file in" % to_dir)
        else:
            raise HTTPError(404, "No such directory: %s" % to_path)

        model = await self.save(model, to_path)
        self.emit(data={"action": "copy", "path": to_path, "source_path": from_path})
        return model

    async def trust_notebook(self, path):
        """项目内部接口说明。"""
        model = await self.get(path)
        nb = model["content"]
        self.log.warning("Trusting notebook %s", path)
        self.notary.mark_cells(nb, True)
        self.check_and_sign(nb, path)

    # Part 3: Checkpoints API
    async def create_checkpoint(self, path):
        """项目内部接口说明。"""
        model = await self.checkpoints.create_checkpoint(self, path)
        return self._annotate_checkpoint_lease(model, path)

    async def restore_checkpoint(self, checkpoint_id, path, credentials=None):
        """项目内部接口说明。"""
        path = path.strip("/")
        backend = self.checkpoints
        uses_save = getattr(backend, "restore_uses_save", False)
        record = None
        if not uses_save:
            record = self._check_lease(path, credentials)
        await backend.restore_checkpoint(self, checkpoint_id, path, credentials=credentials)
        if record is not None:
            self.lease_store.commit_write(path, record.token, new_key=self._lease_fingerprint(path))

    async def list_checkpoints(self, path):
        """项目内部接口说明。"""
        return await self.checkpoints.list_checkpoints(path)

    async def delete_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        return await self.checkpoints.delete_checkpoint(checkpoint_id, path)

import base64
import os

from anyio.to_thread import run_sync
from tornado import web

from jupyter_server.services.contents.filemanager import (
    AsyncFileContentsManager,
    FileContentsManager,
)


class LargeFileManager(FileContentsManager):
    """项目内部接口说明。"""

    def save(self, model, path="", credentials=None):
        """项目内部接口说明。"""
        chunk = model.get("chunk", None)
        if chunk is not None:
            path = path.strip("/")

            if chunk == 1:
                self.run_pre_save_hooks(model=model, path=path)

            # Every piece of a chunked upload must carry a live fence if the
            # client opted into leases. prepare_write() is not used because
            # the file legitimately changes between pieces; the write version
            # is bumped once, when the final piece lands.
            lease_record = self._check_lease(path, credentials)

            if "type" not in model:
                raise web.HTTPError(400, "No file type provided")
            if model["type"] != "file":
                raise web.HTTPError(
                    400,
                    'File type "{}" is not supported for large file transfer'.format(model["type"]),
                )
            if "content" not in model and model["type"] != "directory":
                raise web.HTTPError(400, "No file content provided")

            os_path = self._get_os_path(path)
            if chunk == -1:
                self.log.debug(f"Saving last chunk of file {os_path}")
            else:
                self.log.debug(f"Saving chunk {chunk} of file {os_path}")

            try:
                if chunk == 1:
                    super()._save_file(os_path, model["content"], model.get("format"))
                else:
                    self._save_large_file(os_path, model["content"], model.get("format"))
            except web.HTTPError:
                raise
            except Exception as e:
                self.log.error("Error while saving file: %s %s", path, e, exc_info=True)
                raise web.HTTPError(500, f"Unexpected error while saving file: {path} {e}") from e

            model = self.get(path, content=False)

            # Last chunk
            if chunk == -1:
                self.run_post_save_hooks(model=model, os_path=os_path)
                if lease_record is not None:
                    self.lease_store.commit_write(
                        path,
                        lease_record.token,
                        new_key=self._lease_fingerprint(path),
                    )
            self.emit(data={"action": "save", "path": path})
            return self._attach_lease_info(model, path, lease_record)
        else:
            return super().save(model, path, credentials=credentials)

    def _save_large_file(self, os_path, content, format):
        """项目内部接口说明。"""
        if format not in {"text", "base64"}:
            raise web.HTTPError(
                400,
                "Must specify format of file contents as 'text' or 'base64'",
            )
        try:
            if format == "text":
                bcontent = content.encode("utf8")
            else:
                b64_bytes = content.encode("ascii")
                bcontent = base64.b64decode(b64_bytes)
        except Exception as e:
            raise web.HTTPError(400, f"Encoding error saving {os_path}: {e}") from e

        with self.perm_to_403(os_path):
            if os.path.islink(os_path):
                os_path = os.path.join(os.path.dirname(os_path), os.readlink(os_path))
            with open(os_path, "ab") as f:
                f.write(bcontent)


class AsyncLargeFileManager(AsyncFileContentsManager):
    """项目内部接口说明。"""

    async def save(self, model, path="", credentials=None):
        """项目内部接口说明。"""
        chunk = model.get("chunk", None)
        if chunk is not None:
            path = path.strip("/")

            if chunk == 1:
                self.run_pre_save_hooks(model=model, path=path)

            lease_record = self._check_lease(path, credentials)

            if "type" not in model:
                raise web.HTTPError(400, "No file type provided")
            if model["type"] != "file":
                raise web.HTTPError(
                    400,
                    'File type "{}" is not supported for large file transfer'.format(model["type"]),
                )
            if "content" not in model and model["type"] != "directory":
                raise web.HTTPError(400, "No file content provided")

            os_path = self._get_os_path(path)
            if chunk == -1:
                self.log.debug(f"Saving last chunk of file {os_path}")
            else:
                self.log.debug(f"Saving chunk {chunk} of file {os_path}")

            try:
                if chunk == 1:
                    await super()._save_file(os_path, model["content"], model.get("format"))
                else:
                    await self._save_large_file(os_path, model["content"], model.get("format"))
            except web.HTTPError:
                raise
            except Exception as e:
                self.log.error("Error while saving file: %s %s", path, e, exc_info=True)
                raise web.HTTPError(500, f"Unexpected error while saving file: {path} {e}") from e

            model = await self.get(path, content=False)

            # Last chunk
            if chunk == -1:
                self.run_post_save_hooks(model=model, os_path=os_path)
                if lease_record is not None:
                    self.lease_store.commit_write(
                        path,
                        lease_record.token,
                        new_key=self._lease_fingerprint(path),
                    )

            self.emit(data={"action": "save", "path": path})
            return self._attach_lease_info(model, path, lease_record)
        else:
            return await super().save(model, path, credentials=credentials)

    async def _save_large_file(self, os_path, content, format):
        """项目内部接口说明。"""
        if format not in {"text", "base64"}:
            raise web.HTTPError(
                400,
                "Must specify format of file contents as 'text' or 'base64'",
            )
        try:
            if format == "text":
                bcontent = content.encode("utf8")
            else:
                b64_bytes = content.encode("ascii")
                bcontent = base64.b64decode(b64_bytes)
        except Exception as e:
            raise web.HTTPError(400, f"Encoding error saving {os_path}: {e}") from e

        with self.perm_to_403(os_path):
            if os.path.islink(os_path):
                os_path = os.path.join(os.path.dirname(os_path), os.readlink(os_path))
            with open(os_path, "ab") as f:  # noqa: ASYNC230
                await run_sync(f.write, bcontent)

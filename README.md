# Jupyter Server 内容服务

本项目提供服务端内容、目录、检查点、会话和鉴权接口。生产源码位于 `jupyter_server/`，核心回归测试位于 `tests/`。

## 安装

`python3 -m pip install --break-system-packages --no-build-isolation -e '.[test]'`

## 测试

`python3 -m pytest -q`

## 构建

`python3 -m compileall -q jupyter_server`

`python3 -m build --wheel --no-isolation`

## 使用

内容管理器可在本地目录上执行保存、复制、改名、删除和检查点操作，HTTP 处理器提供对应服务端接口。

## 编辑租约（栅栏令牌）

内容服务提供带栅栏令牌（fencing token）的编辑租约，用于防止断线重连的旧客户端覆盖已确认的新版本。

- `POST /api/contents/{path}/lease`：获取租约，请求体 `{"holder": "...", "ttl": 秒}`；同一持有人重复获取返回原租约（用于重连恢复），他人获取返回 409。
- `PUT /api/contents/{path}/lease`：续租，请求体 `{"lease_id": "...", "generation": N, "ttl": 秒}`；续租不改变代际。
- `DELETE /api/contents/{path}/lease`：释放租约。
- `GET /api/contents/{path}/lease`：查看当前租约与失效审计记录（不泄露 lease_id），供支持人员排查。
- `POST /api/contents/{path}/lease/takeover`：管理员接管，请求体必须包含 `reason`；旧令牌永久失效并记录原因与操作人。

保存（PUT）、改名（PATCH）、删除（DELETE）、检查点创建与恢复均可携带租约令牌（请求体 `lease` 对象、`X-Jupyter-Lease-Id`/`X-Jupyter-Lease-Generation`/`X-Jupyter-Request-Id` 请求头，或 `lease_id`/`lease_generation`/`request_id` 查询参数），服务端校验同一代际后才允许写入。检查点会记录创建时的租约代际（`lease_generation`），但不会延长租约。

冲突一律返回 409 与可恢复的 JSON 信息（`lease_conflict.code`：`lease_held`、`lease_expired`、`lease_invalidated`、`stale_generation`、`external_modification`、`lease_required`、`duplicate_request`，以及 `recovery` 提示）。携带相同 `request_id` 的重复提交按幂等重放处理；租约状态持久化于运行目录，服务重启、时钟漂移与文件被外部修改时均拒绝静默覆盖。默认租约为可选模式；设置 `ContentsManager.require_lease=True` 可强制所有写操作持有有效租约。

"""本地 HTTP 服务：界面 ⇄ 后端的唯一通道（浏览器 / pywebview 窗口 / 测试都走这里）。

路由（合同 5.8）：
  GET  /                          → 302 到 /ui/index.html?t=<token>
  GET  /ui/<文件>                 → 静态界面文件（防路径穿越，只放行白名单扩展名）
  POST /api/<method>              → Api.<method>(**body) → {"ok": true, "data": ...} / {"ok": false, "error": "中文"}
  GET  /events?t=<token>          → SSE：Engine 事件 100ms 合并成一批推送，定时心跳
  GET  /img/<job>/<index>/<kind>  → 页面图片（src|clean|out|thumb），带 v 参数时长缓存；
                                    带 w=<最长边> 时现场缩成 JPEG（封面 / 胶片条用，clean 补丁除外）
  GET  /font-preview?id=<font_id> → 字体预览 PNG（黑字透明底，界面用 CSS mask 着色；扩展路由）

安全：
  - 只监听 127.0.0.1，随机端口；每次启动生成随机 token（常量时间比较）。
  - Host 头白名单（防 DNS rebinding）；POST 带了跨站 Origin 直接拒绝；不发任何 CORS 头。
  - /api 只允许调用 Api 类上公开（不以 _ 开头）的方法。
"""
from __future__ import annotations

import collections
import dataclasses
import hmac
import inspect
import json
import logging
import math
import os
import re
import secrets
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

log = logging.getLogger("hanhua.server")

BATCH_WINDOW_S = 0.1          # SSE 合并窗口：100ms 内的事件合成一批
HEARTBEAT_S = 15.0            # 没有事件时的心跳间隔
MAX_BODY = 32 * 1024 * 1024   # /api 请求体上限（术语表导入、整页块数据都远小于此）
MAX_LOGS_PER_BATCH = 50       # 一批里最多带多少条 log 事件（防止刷屏拖垮界面）

_STATIC_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".txt": "text/plain; charset=utf-8",
}

# 浏览器能直接显示的图片；其余（tif 等）现场转成 JPEG
_IMAGE_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".avif": "image/avif",
}

_IMG_ROUTE = re.compile(r"^/img/([A-Za-z0-9_\-]{1,64})/(\d{1,7})/(src|clean|out|thumb)$")
_FONT_ROUTE = re.compile(r"^/font-preview/([^/]{1,600})$")   # 旧写法：id 放路径里（百分号编码）
THUMB_MIN, THUMB_MAX = 32, 2048          # /img 的 w 参数范围
TRANSCODE_CACHE_BYTES = 48 * 1024 * 1024  # 转码 / 缩略图缓存上限
_API_ROUTE = re.compile(r"^/api/([a-z][a-z0-9_]{0,63})$")


# --------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------
def _sanitize(obj: Any, depth: int = 0) -> Any:
    """转成可 JSON 序列化的纯数据：NaN/Inf → None，numpy / tuple / set / dataclass 等统一处理。"""
    if depth > 64:
        return None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_sanitize(v, depth + 1) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        to_dict = getattr(obj, "to_dict", None)
        return _sanitize(to_dict() if callable(to_dict) else dataclasses.asdict(obj), depth + 1)
    if isinstance(obj, (bytes, bytearray)):
        return None
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    # numpy 标量 / 数组（不直接 import numpy，保持本模块轻量）
    item = getattr(obj, "item", None)
    tolist = getattr(obj, "tolist", None)
    if callable(tolist) and hasattr(obj, "shape"):
        return _sanitize(tolist(), depth + 1)
    if callable(item):
        try:
            return _sanitize(item(), depth + 1)
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def to_json_bytes(obj: Any) -> bytes:
    return json.dumps(_sanitize(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


# --------------------------------------------------------------------------------------
# 事件总线 + SSE 订阅
# --------------------------------------------------------------------------------------
def coalesce_events(events: list[dict], max_logs: int = MAX_LOGS_PER_BATCH) -> list[dict]:
    """合并一个窗口内的事件。

    - job：同一任务只保留最新 summary（summary 是全量快照）
    - engine：只保留最新
    - page：同一 (job_id, index) 只保留最新阶段 / 版本
    - log / 其它：按顺序保留（log 最多 max_logs 条）
    """
    out: list[Optional[dict]] = []
    pos: dict[tuple, int] = {}
    logs = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        key: Optional[tuple] = None
        if t == "job":
            job = ev.get("job") if isinstance(ev.get("job"), dict) else {}
            key = ("job", job.get("id"))
        elif t == "engine":
            key = ("engine",)
        elif t == "page":
            key = ("page", ev.get("job_id"), ev.get("index"))
        elif t == "log":
            logs += 1
            if logs > max_logs:
                continue
        if key is not None:
            old = pos.get(key)
            if old is not None:
                out[old] = None
            pos[key] = len(out)
        out.append(ev)
    return [e for e in out if e is not None]


class Subscription:
    """一个 SSE 连接的事件缓冲（有界：客户端卡住时丢最旧的，并让它重新同步）。"""

    def __init__(self, maxlen: int = 4000) -> None:
        self._q: collections.deque[dict] = collections.deque()
        self._maxlen = max(16, int(maxlen))
        self._cond = threading.Condition()
        self._dropped = False
        self.closed = False

    def put(self, event: dict) -> None:
        with self._cond:
            if self.closed:
                return
            if len(self._q) >= self._maxlen:
                self._q.popleft()
                self._dropped = True
            self._q.append(event)
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def get_batch(self, timeout: float, window: float = BATCH_WINDOW_S) -> Optional[list[dict]]:
        """等第一条事件（最多 timeout 秒），再收集 window 秒内的后续事件并合并。

        返回：事件列表；超时无事件返回 []；订阅已关闭返回 None。
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while not self._q and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cond.wait(remaining)
            if self.closed:
                return None
        if window > 0:
            # 合并窗口：让同一时刻的一串事件（多页完成 + 任务进度）合成一次推送
            end = time.monotonic() + window
            with self._cond:
                while not self.closed:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(remaining)
        with self._cond:
            items = list(self._q)
            self._q.clear()
            dropped, self._dropped = self._dropped, False
            closed = self.closed
        if closed and not items:
            return None
        batch = coalesce_events(items)
        if dropped:
            batch.insert(0, {"type": "resync"})
        return batch


class EventBus:
    """Engine 事件 → 所有 SSE 订阅者 + 进程内监听者。publish 永不阻塞、永不抛异常。"""

    def __init__(self, maxlen: int = 4000) -> None:
        self._subs: set[Subscription] = set()
        self._listeners: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()
        self._maxlen = maxlen

    def publish(self, event: dict) -> None:
        if not isinstance(event, dict):
            return
        with self._lock:
            subs = list(self._subs)
            listeners = list(self._listeners)
        for s in subs:
            s.put(event)
        for cb in listeners:
            try:
                cb(event)
            except Exception:  # noqa: BLE001
                log.exception("事件监听器出错")

    def subscribe(self) -> Subscription:
        sub = Subscription(self._maxlen)
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        sub.close()
        with self._lock:
            self._subs.discard(sub)

    def add_listener(self, cb: Callable[[dict], None]) -> None:
        with self._lock:
            self._listeners.append(cb)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def close_subscribers(self) -> None:
        """关闭所有 SSE 连接（服务器停止时调用；总线本身仍可继续 publish）。"""
        with self._lock:
            subs = list(self._subs)
            self._subs.clear()
        for s in subs:
            s.close()


# --------------------------------------------------------------------------------------
# HTTP 服务器
# --------------------------------------------------------------------------------------
def _public_methods(api: Any) -> dict[str, str]:
    """Api 类上可以从网页调用的方法名（不以 _ 开头的普通方法）。"""
    names: dict[str, str] = {}
    for klass in type(api).__mro__:
        if klass is object:
            continue
        for n, v in vars(klass).items():
            if n.startswith("_") or n in names:
                continue
            if inspect.isfunction(v) or isinstance(v, (staticmethod, classmethod)):
                names[n] = n
    return names


def _safe_join(root: str, rel: str) -> Optional[str]:
    """把 URL 相对路径安全地拼到 root 下；任何穿越 / 绝对路径 / 盘符 / 隐藏文件都返回 None。"""
    try:
        rel = urllib.parse.unquote(rel, errors="strict")
    except UnicodeDecodeError:
        return None
    if not rel or "\x00" in rel or "\\" in rel or ":" in rel:
        return None
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or p.startswith(".") for p in parts):
        return None
    full = os.path.realpath(os.path.join(root, *parts))
    try:
        if os.path.commonpath([full, root]) != root:
            return None
    except ValueError:  # 不同盘符
        return None
    return full


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False   # Windows 上 SO_REUSEADDR 允许别的进程抢同一端口
    request_queue_size = 64
    app: "AppServer"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)  # type: ignore[attr-defined]
            except OSError:
                pass
        # 不调用 HTTPServer.server_bind：它会做 getfqdn 反向 DNS，某些电脑上要卡好几秒
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            log.debug("客户端断开：%s %s", client_address, exc)
            return
        log.exception("HTTP 请求处理异常：%s", client_address)


class _Handler(BaseHTTPRequestHandler):
    server_version = "Hanhua/4.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    server: _HTTPServer  # type: ignore[assignment]

    # ---- 基础 -------------------------------------------------------------------------
    @property
    def app(self) -> "AppServer":
        return self.server.app

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - 覆盖父类
        log.debug("%s - " + fmt, self.client_address[0], *args)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_HEAD(self) -> None:  # noqa: N802
        self.close_connection = True
        self._send_plain(405, "Method Not Allowed")

    def do_PUT(self) -> None:  # noqa: N802
        self.close_connection = True
        self._send_plain(405, "Method Not Allowed")

    do_DELETE = do_PUT
    do_OPTIONS = do_PUT   # 不回应 CORS 预检 → 跨站页面无法调用 API

    def _dispatch(self, method: str) -> None:
        try:
            if not self._host_ok():
                self.close_connection = True
                self._send_plain(403, "Forbidden")
                return
            parts = urllib.parse.urlsplit(self.path)
            path = parts.path
            query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
            if method == "POST":
                m = _API_ROUTE.match(path)
                if m:
                    self._handle_api(m.group(1))
                else:
                    self.close_connection = True
                    self._send_plain(404, "Not Found")
                return
            if path in ("/", "/index.html"):
                self._redirect(f"/ui/index.html?t={urllib.parse.quote(self.app.token)}")
            elif path.startswith("/ui/"):
                self._handle_static(path[len("/ui/"):])
            elif path == "/events":
                self._handle_events(query)
            elif path.startswith("/img/"):
                self._handle_image(path, query)
            elif path == "/font-preview" or path.startswith("/font-preview/"):
                self._handle_font_preview(path, query)
            elif path == "/favicon.ico":
                self._send_bytes(204, b"", "image/x-icon")
            else:
                self._send_plain(404, "Not Found")
        except (ConnectionError, TimeoutError) as e:
            log.debug("连接中断：%s", e)
            self.close_connection = True
        except Exception:  # noqa: BLE001
            log.exception("处理请求出错：%s %s", method, self.path)
            self.close_connection = True
            try:
                self._send_json(500, {"ok": False, "error": "程序内部出错，详情已写入日志"})
            except Exception:  # noqa: BLE001
                pass

    # ---- 校验 -------------------------------------------------------------------------
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        return host in self.app.allowed_hosts

    def _token_ok(self, given: Optional[str]) -> bool:
        if not given:
            return False
        return hmac.compare_digest(str(given).encode("utf-8"), self.app.token.encode("utf-8"))

    def _query_token_ok(self, query: dict[str, list[str]]) -> bool:
        return self._token_ok((query.get("t") or [""])[0])

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin or origin == "null":
            return True   # 同源 fetch / pywebview 可能不带 Origin
        try:
            netloc = urllib.parse.urlsplit(origin).netloc.lower()
        except ValueError:
            return False
        return netloc in self.app.allowed_hosts

    # ---- 发送 -------------------------------------------------------------------------
    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")

    def _send_bytes(self, status: int, body: bytes, ctype: str, extra: Optional[dict[str, str]] = None) -> None:
        self.send_response(status)
        self._common_headers()
        if ctype:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_plain(self, status: int, text: str) -> None:
        self._send_bytes(status, text.encode("utf-8"), "text/plain; charset=utf-8", {"Cache-Control": "no-store"})

    def _send_json(self, status: int, obj: Any) -> None:
        self._send_bytes(status, to_json_bytes(obj), "application/json; charset=utf-8", {"Cache-Control": "no-store"})

    def _redirect(self, location: str) -> None:
        self._send_bytes(302, b"", "", {"Location": location, "Cache-Control": "no-store"})

    def _send_file(self, path: str, ctype: str, cache: str) -> None:
        """流式发送文件（支持 ETag / 304）。"""
        st = os.stat(path)
        etag = f'W/"{st.st_size:x}-{st.st_mtime_ns:x}"'
        if self.headers.get("If-None-Match") == etag:
            self._send_bytes(304, b"", "", {"ETag": etag, "Cache-Control": cache})
            return
        self.send_response(200)
        self._common_headers()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(st.st_size))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", cache)
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(256 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # ---- 路由实现 ---------------------------------------------------------------------
    def _handle_static(self, rel: str) -> None:
        full = _safe_join(self.app.ui_dir, rel)
        ext = os.path.splitext(full or "")[1].lower()
        if not full or ext not in _STATIC_TYPES or not os.path.isfile(full):
            self._send_plain(404, "Not Found")
            return
        self._send_file(full, _STATIC_TYPES[ext], "no-cache")

    def _handle_api(self, name: str) -> None:
        # 先读完请求体（keep-alive 下不读完会把下一个请求读乱）
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self.close_connection = True
            self._send_json(413, {"ok": False, "error": "请求数据太大或格式不对"})
            return
        raw = self.rfile.read(length) if length else b""
        if not self._token_ok(self.headers.get("X-Token")):
            self._send_json(403, {"ok": False, "error": "访问令牌无效，请关闭后重新打开软件"})
            return
        if not self._origin_ok():
            self._send_json(403, {"ok": False, "error": "拒绝来自其它网站的请求"})
            return
        try:
            params = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "error": "请求数据格式不对（不是有效的 JSON）"})
            return
        if not isinstance(params, dict):
            self._send_json(400, {"ok": False, "error": "请求参数必须是对象"})
            return
        self._send_json(200, self.app.call_api(name, params))

    def _handle_events(self, query: dict[str, list[str]]) -> None:
        if not self._query_token_ok(query):
            self._send_plain(403, "Forbidden")
            return
        self.close_connection = True
        self.send_response(200)
        self._common_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        app = self.app
        sub = app.bus.subscribe()
        try:
            # retry：断线 1.5 秒后浏览器自动重连；hello 让界面知道连接已建立（并据此重新同步）
            self._sse_write(b"retry: 1500\n\n")
            self._sse_write(b"data: " + to_json_bytes([{"type": "hello", "ts": time.time()}]) + b"\n\n")
            while not app.stopping:
                batch = sub.get_batch(app.heartbeat_s, app.batch_window_s)
                if batch is None:
                    break
                if not batch:
                    self._sse_write(b": ping\n\n")
                    continue
                self._sse_write(b"data: " + to_json_bytes(batch) + b"\n\n")
        except (ConnectionError, TimeoutError, OSError) as e:
            log.debug("SSE 连接结束：%s", e)
        finally:
            app.bus.unsubscribe(sub)

    def _sse_write(self, data: bytes) -> None:
        self.wfile.write(data)
        self.wfile.flush()

    def _handle_image(self, path: str, query: dict[str, list[str]]) -> None:
        m = _IMG_ROUTE.match(path)
        if not m:
            self._send_plain(404, "Not Found")
            return
        if not self._query_token_ok(query):
            self._send_plain(403, "Forbidden")
            return
        job_id, index, kind = m.group(1), int(m.group(2)), m.group(3)
        file_path = self.app.image_path(job_id, index, kind)
        if not file_path or not os.path.isfile(file_path):
            self._send_plain(404, "Not Found")
            return
        cache = "private, max-age=31536000, immutable" if query.get("v") else "no-cache"
        ext = os.path.splitext(file_path)[1].lower()
        ctype = _IMAGE_TYPES.get(ext)
        max_side = self._thumb_side(query) if kind != "clean" else 0   # clean 是 RGBA 补丁，不能转 JPEG
        if ctype and not max_side:
            self._send_file(file_path, ctype, cache)
            return
        data = self.app.transcode_image(file_path, max_side)
        if data is None:
            if ctype:   # 缩略图失败（图坏了 / 内存不够）→ 退回原图，至少能显示
                self._send_file(file_path, ctype, cache)
            else:
                self._send_plain(415, "Unsupported Media Type")
            return
        self._send_bytes(200, data, "image/jpeg", {"Cache-Control": cache})

    @staticmethod
    def _thumb_side(query: dict[str, list[str]]) -> int:
        """w 参数 → 缩略图最长边（非法 / 缺省 → 0 = 原图）。"""
        raw = (query.get("w") or [""])[0]
        if not raw.isdigit():
            return 0
        return max(THUMB_MIN, min(THUMB_MAX, int(raw)))

    def _handle_font_preview(self, path: str, query: dict[str, list[str]]) -> None:
        if not self._query_token_ok(query):
            self._send_plain(403, "Forbidden")
            return
        if path == "/font-preview":
            font_id = (query.get("id") or [""])[0]
        else:
            m = _FONT_ROUTE.match(path)
            font_id = urllib.parse.unquote(m.group(1)) if m else ""
        # 字体 id 可能含中文 / 冒号（用户字体 user:文件名#序号），只拦控制字符和超长
        if not font_id or len(font_id) > 200 or any(ord(c) < 32 for c in font_id):
            self._send_plain(404, "Not Found")
            return
        text = (query.get("text") or [""])[0][:40]
        data = self.app.font_preview(font_id, text)
        if not data:
            self._send_plain(404, "Not Found")
            return
        self._send_bytes(200, data, "image/png", {"Cache-Control": "private, max-age=3600"})


class AppServer:
    """`AppServer(api, ui_dir)`：绑 127.0.0.1 随机端口的 ThreadingHTTPServer。"""

    def __init__(
        self,
        api: Any,
        ui_dir: str,
        bus: Optional[EventBus] = None,
        host: str = "127.0.0.1",
        port: int = 0,
        heartbeat_s: float = HEARTBEAT_S,
        batch_window_s: float = BATCH_WINDOW_S,
    ) -> None:
        self.api = api
        self.ui_dir = os.path.realpath(ui_dir)
        bus_obj = bus if bus is not None else getattr(api, "bus", None)
        self.bus: EventBus = bus_obj if isinstance(bus_obj, EventBus) else EventBus()
        self.token = secrets.token_urlsafe(24)
        self.heartbeat_s = float(heartbeat_s)
        self.batch_window_s = float(batch_window_s)
        self._host = host
        self._port = int(port)
        self._httpd: Optional[_HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._methods = _public_methods(api)
        self._sig_cache: dict[str, inspect.Signature] = {}
        self._transcode_cache: collections.OrderedDict[tuple, bytes] = collections.OrderedDict()
        self._transcode_lock = threading.Lock()
        self._transcode_bytes = 0
        self.allowed_hosts: set[str] = set()

    # ---- 生命周期 ---------------------------------------------------------------------
    @property
    def port(self) -> int:
        return self._httpd.server_port if self._httpd else self._port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def url(self) -> str:
        """窗口 / 浏览器要打开的完整地址（带 token）。"""
        return f"{self.base_url}/ui/index.html?t={urllib.parse.quote(self.token)}"

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def start(self) -> "AppServer":
        if self._httpd is not None:
            return self
        httpd = _HTTPServer((self._host, self._port), _Handler)
        httpd.app = self
        self._httpd = httpd
        port = httpd.server_port
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.25}, name="hanhua-http", daemon=True
        )
        self._thread.start()
        log.info("界面服务已启动：%s", self.base_url)
        return self

    def stop(self) -> None:
        self._stopping.set()
        self.bus.close_subscribers()
        httpd, self._httpd = self._httpd, None
        if httpd is None:
            return
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:  # noqa: BLE001
            log.exception("关闭界面服务出错")
        if self._thread is not None:
            self._thread.join(timeout=3)
        log.info("界面服务已停止")

    # ---- API 调用 ---------------------------------------------------------------------
    def call_api(self, name: str, params: dict) -> dict:
        """调用 Api 的公开方法并包装成统一结果；任何异常都转成中文错误，绝不抛出。"""
        if name not in self._methods:
            return {"ok": False, "error": f"未知的操作：{name}"}
        fn = getattr(self.api, name, None)
        if not callable(fn):
            return {"ok": False, "error": f"未知的操作：{name}"}
        try:
            sig = self._sig_cache.get(name)
            if sig is None:
                sig = inspect.signature(fn)
                self._sig_cache[name] = sig
            sig.bind(**params)
        except TypeError as e:
            return {"ok": False, "error": f"参数不正确：{e}"}
        except ValueError:
            pass
        try:
            return {"ok": True, "data": fn(**params)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": self._error_text(name, e)}

    @staticmethod
    def _error_text(name: str, e: BaseException) -> str:
        # HanhuaError（含 Api 的 ApiError）的 message 本身就是给客户看的中文
        try:
            from .types import HanhuaError
        except Exception:  # noqa: BLE001
            HanhuaError = ()  # type: ignore[assignment]
        msg = str(e).strip()
        if HanhuaError and isinstance(e, HanhuaError):
            return msg or "操作失败"
        log.error("API %s 出错", name, exc_info=e)
        return f"操作失败：{msg[:200]}" if msg else "操作失败，详情已写入日志"

    # ---- 图片 / 字体（委托给 Api 的内部方法，Api 没有时退回 engine）---------------------
    def image_path(self, job_id: str, index: int, kind: str) -> Optional[str]:
        try:
            fn = getattr(self.api, "_image_path", None)
            if callable(fn):
                return fn(job_id, index, kind)
            engine = getattr(self.api, "engine", None)
            if engine is not None:
                return engine.page_image_path(job_id, index, kind)
        except Exception:  # noqa: BLE001
            log.debug("取页面图片路径失败 %s/%s/%s", job_id, index, kind, exc_info=True)
        return None

    def font_preview(self, font_id: str, text: str) -> Optional[bytes]:
        fn = getattr(self.api, "_font_preview", None)
        if not callable(fn):
            return None
        try:
            return fn(font_id, text)
        except Exception:  # noqa: BLE001
            log.debug("字体预览失败 %s", font_id, exc_info=True)
            return None

    def transcode_image(self, path: str, max_side: int = 0) -> Optional[bytes]:
        """转成 JPEG：浏览器不认的格式（tif 等），或 max_side>0 时缩成缩略图。

        按 (路径, 修改时间, 大小, 边长) 做 LRU 缓存，总量不超过 TRANSCODE_CACHE_BYTES。
        """
        try:
            st = os.stat(path)
            key = (path, st.st_mtime_ns, st.st_size, int(max_side))
            with self._transcode_lock:
                hit = self._transcode_cache.get(key)
                if hit is not None:
                    self._transcode_cache.move_to_end(key)
                    return hit
            from . import imageio

            img = imageio.read(path)
            if max_side > 0:
                img = imageio.thumbnail(img, int(max_side))
            data = imageio.encode(img, "jpg", 86 if max_side else 90)
            with self._transcode_lock:
                self._transcode_cache[key] = data
                self._transcode_bytes += len(data)
                while self._transcode_cache and (
                    self._transcode_bytes > TRANSCODE_CACHE_BYTES or len(self._transcode_cache) > 512
                ):
                    _, old = self._transcode_cache.popitem(last=False)
                    self._transcode_bytes -= len(old)
            return data
        except Exception:  # noqa: BLE001
            log.warning("图片转码失败：%s", path, exc_info=True)
            return None

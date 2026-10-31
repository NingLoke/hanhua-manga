"""界面通道测试：AppServer（静态文件 / 路径穿越 / token / Host / Origin / API / SSE / 图片 / 字体预览）
+ Api（演示引擎端到端：任务、校对往返、后台导出、设置、术语表、打开文件安全）+ 命令行冒烟。"""
from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from hanhua import imageio
from hanhua.app import (
    Api, ApiError, UnavailableEngine, clean_glossary, dump_glossary_csv, parse_glossary_csv, short_gpu_name, ui_dir,
)
from hanhua.config import DEFAULTS, SettingsStore
from hanhua.server import AppServer, EventBus, Subscription, coalesce_events
from hanhua.ui.demo import DemoEngine


# --------------------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------------------
class Http:
    """直接用 http.client：能精确控制 Host / Origin / X-Token 等请求头。"""

    def __init__(self, server: AppServer) -> None:
        self.server = server

    def request(self, method: str, path: str, body: Any = None, headers: Optional[dict] = None,
                token: Optional[str] = "auto") -> tuple[int, dict, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=10)
        h = {"Host": f"127.0.0.1:{self.server.port}"}
        if token == "auto":
            token = self.server.token
        if token is not None and path.startswith("/api/"):
            h["X-Token"] = token
        data: Optional[bytes] = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        try:
            conn.request(method, path, body=data, headers=h)
            r = conn.getresponse()
            return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
        finally:
            conn.close()

    def api(self, name: str, **params: Any) -> dict:
        status, _, body = self.request("POST", f"/api/{name}", params)
        res = json.loads(body.decode("utf-8"))
        assert isinstance(res, dict) and "ok" in res, res
        return res

    def ok(self, name: str, **params: Any) -> Any:
        res = self.api(name, **params)
        assert res["ok"], res
        return res["data"]


class _FakeApi:
    """最小 Api：只用于服务器层测试。"""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.images: dict[tuple[str, int, str], str] = {}

    def ping(self) -> dict:
        return {"pong": True}

    def echo(self, value: Any = None, n: int = 0) -> dict:
        return {"value": value, "n": n}

    def boom(self) -> None:
        raise RuntimeError("内部炸了")

    def friendly(self) -> None:
        from hanhua.types import HanhuaError

        raise HanhuaError("给客户看的中文原因")

    def weird(self) -> dict:
        return {"nan": float("nan"), "arr": np.arange(3), "tup": (1, 2), "path": Path("a/b")}

    def _secret(self) -> str:
        return "不能被调用"

    def _image_path(self, job_id: str, index: int, kind: str) -> Optional[str]:
        return self.images.get((job_id, index, kind))

    def _font_preview(self, font_id: str, text: str) -> Optional[bytes]:
        return imageio.encode(np.zeros((8, 8, 3), np.uint8), "png") if font_id == "ok" else None


@pytest.fixture
def static_dir(tmp_path: Path) -> Path:
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    (ui / "app.css").write_text("body{}", encoding="utf-8")
    (ui / "app.js").write_text("console.log(1)", encoding="utf-8")
    (ui / "中文 名.svg").write_text("<svg/>", encoding="utf-8")
    (ui / ".hidden.txt").write_text("hidden", encoding="utf-8")
    (ui / "tool.exe").write_bytes(b"MZ")
    (tmp_path / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")
    return ui


@pytest.fixture
def fake(static_dir: Path):
    api = _FakeApi()
    srv = AppServer(api, str(static_dir), heartbeat_s=0.3, batch_window_s=0.05).start()
    try:
        yield api, srv, Http(srv)
    finally:
        srv.stop()


@pytest.fixture
def demo(tmp_path: Path, hanhua_data: Path):
    """演示引擎 + 真 Api + 真 AppServer（真实界面目录）。"""
    bus = EventBus()
    store = SettingsStore(str(hanhua_data / "settings.json"))
    engine = DemoEngine(store, bus.publish, work_root=str(tmp_path / "demo"), speed=4.0)
    opened: list[tuple[str, bool]] = []
    api = Api(store, engine, bus, demo=True, opener=lambda p, r: opened.append((p, r)))
    srv = AppServer(api, ui_dir(), bus=bus, heartbeat_s=0.5).start()
    try:
        yield api, srv, Http(srv), opened
    finally:
        srv.stop()
        engine.shutdown()


def _job_by_status(api: Api, status: str) -> dict:
    return next(j for j in api.list_jobs() if j["status"] == status)


# --------------------------------------------------------------------------------------
# 静态文件 / 路径穿越 / 安全头
# --------------------------------------------------------------------------------------
def test_root_redirects_with_token(fake) -> None:
    _, srv, h = fake
    status, headers, _ = h.request("GET", "/")
    assert status == 302
    assert headers["location"] == f"/ui/index.html?t={srv.token}"
    assert srv.url.startswith(srv.base_url) and srv.token in srv.url


def test_static_files_and_types(fake) -> None:
    _, _, h = fake
    status, headers, body = h.request("GET", "/ui/index.html")
    assert status == 200 and headers["content-type"].startswith("text/html") and b"<title>" in body
    assert headers["x-content-type-options"] == "nosniff"
    assert h.request("GET", "/ui/app.css")[1]["content-type"].startswith("text/css")
    assert h.request("GET", "/ui/app.js")[1]["content-type"].startswith("text/javascript")
    assert h.request("GET", "/ui/%E4%B8%AD%E6%96%87%20%E5%90%8D.svg")[0] == 200   # 中文文件名
    assert h.request("GET", "/ui/missing.js")[0] == 404
    assert h.request("GET", "/ui/tool.exe")[0] == 404          # 扩展名不在白名单
    assert h.request("GET", "/nothing")[0] == 404


def test_static_etag_304(fake) -> None:
    _, _, h = fake
    _, headers, _ = h.request("GET", "/ui/app.css")
    status, _, body = h.request("GET", "/ui/app.css", headers={"If-None-Match": headers["etag"]})
    assert status == 304 and body == b""


@pytest.mark.parametrize("path", [
    "/ui/../secret.txt", "/ui/%2e%2e/secret.txt", "/ui/..%2fsecret.txt", "/ui/%2e%2e%2fsecret.txt",
    "/ui/..%5csecret.txt", "/ui/%2e%2e%5csecret.txt", "/ui/.hidden.txt", "/ui/C:/Windows/win.ini",
    "/ui/C%3A%5CWindows%5Cwin.ini", "/ui//etc/passwd", "/ui/%00index.html", "/ui/index.html%00.txt",
    "/ui/./../secret.txt", "/ui/a/../../secret.txt", "/ui/%252e%252e/secret.txt", "/ui/%ff%fe.txt",
])
def test_path_traversal_rejected(fake, path: str) -> None:
    _, _, h = fake
    status, _, body = h.request("GET", path)
    assert status == 404, path
    assert b"TOP-SECRET" not in body and b"hidden" not in body


def test_host_header_whitelist(fake) -> None:
    _, srv, h = fake
    assert h.request("GET", "/ui/index.html", headers={"Host": "evil.example:80"})[0] == 403
    assert h.request("GET", "/ui/index.html", headers={"Host": f"localhost:{srv.port}"})[0] == 200
    assert h.request("POST", "/api/ping", {}, headers={"Host": "evil.example"})[0] == 403


def test_other_methods_rejected(fake) -> None:
    _, _, h = fake
    for m in ("OPTIONS", "PUT", "DELETE", "HEAD"):
        status, headers, _ = h.request(m, "/api/ping")
        assert status == 405, m
        assert "access-control-allow-origin" not in headers


# --------------------------------------------------------------------------------------
# /api
# --------------------------------------------------------------------------------------
def test_api_token_required(fake) -> None:
    _, _, h = fake
    for tok in (None, "", "wrong-token"):
        status, _, body = h.request("POST", "/api/ping", {}, token=tok)
        res = json.loads(body)
        assert status == 403 and res["ok"] is False and "令牌" in res["error"]
    assert h.ok("ping") == {"pong": True}


def test_api_origin_check(fake) -> None:
    _, srv, h = fake
    status, _, body = h.request("POST", "/api/ping", {}, headers={"Origin": "http://evil.example"})
    assert status == 403 and json.loads(body)["ok"] is False
    status, _, _ = h.request("POST", "/api/ping", {}, headers={"Origin": f"http://127.0.0.1:{srv.port}"})
    assert status == 200
    assert h.request("POST", "/api/ping", {}, headers={"Origin": "null"})[0] == 200


def test_api_dispatch_and_errors(fake) -> None:
    _, _, h = fake
    assert h.ok("echo", value="中文", n=3) == {"value": "中文", "n": 3}
    res = h.api("echo", nope=1)
    assert not res["ok"] and "参数不正确" in res["error"]
    res = h.api("missing_method")
    assert not res["ok"] and "未知的操作" in res["error"]
    res = h.api("boom")
    assert not res["ok"] and res["error"].startswith("操作失败")
    assert h.api("friendly") == {"ok": False, "error": "给客户看的中文原因"}
    assert h.request("POST", "/api/_secret", {})[0] == 404          # 下划线方法连路由都进不去
    assert h.request("POST", "/api/__init__", {})[0] == 404
    assert h.request("GET", "/api/ping")[0] == 404
    assert h.request("POST", "/api/ping", b"{bad json")[0] == 400
    assert h.request("POST", "/api/ping", b"[1,2]")[0] == 400
    assert h.request("POST", "/api/ping", b"\xff\xfe")[0] == 400
    assert h.request("POST", "/api/ping", b"", headers={"Content-Length": "999999999"})[0] == 413
    data = h.ok("weird")                                              # NaN / numpy / tuple / Path 都能序列化
    assert data == {"nan": None, "arr": [0, 1, 2], "tup": [1, 2], "path": os.path.join("a", "b")}


def test_api_keepalive_multiple_requests(fake) -> None:
    _, srv, _ = fake
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
    try:
        for i in range(3):
            conn.request("POST", "/api/echo", body=json.dumps({"n": i}),
                         headers={"Host": f"127.0.0.1:{srv.port}", "X-Token": srv.token})
            assert json.loads(conn.getresponse().read())["data"]["n"] == i
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# SSE
# --------------------------------------------------------------------------------------
def _sse_open(srv: AppServer, token: str) -> socket.socket:
    s = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
    s.sendall(f"GET /events?t={token} HTTP/1.1\r\nHost: 127.0.0.1:{srv.port}\r\n\r\n".encode())
    return s


def _sse_read_until(s: socket.socket, needle: bytes, timeout: float = 5.0) -> bytes:
    buf = b""
    end = time.monotonic() + timeout
    while needle not in buf and time.monotonic() < end:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf


def test_sse_hello_events_and_heartbeat(fake) -> None:
    api, srv, _ = fake
    s = _sse_open(srv, srv.token)
    try:
        head = _sse_read_until(s, b'"hello"')
        assert b"200 OK" in head and b"text/event-stream" in head and b"retry: 1500" in head
        for i in range(5):    # 合并窗口内的 5 条同任务进度只推最新一条
            api.bus.publish({"type": "job", "job": {"id": "j1", "done": i}})
        api.bus.publish({"type": "log", "level": "info", "message": "测试消息"})
        buf = _sse_read_until(s, "测试消息".encode("utf-8"))
        line = next(ln for ln in buf.split(b"\n") if "测试消息".encode("utf-8") in ln)
        batch = json.loads(line[len(b"data: "):].decode("utf-8"))
        jobs = [e for e in batch if e["type"] == "job"]
        assert len(jobs) == 1 and jobs[0]["job"]["done"] == 4
        assert b": ping" in _sse_read_until(s, b": ping", timeout=3)   # 空闲时心跳
    finally:
        s.close()


def test_sse_bad_token(fake) -> None:
    _, srv, h = fake
    assert h.request("GET", "/events?t=wrong")[0] == 403
    assert h.request("GET", "/events")[0] == 403


def test_server_stop_closes_sse(static_dir: Path) -> None:
    api = _FakeApi()
    srv = AppServer(api, str(static_dir), heartbeat_s=5).start()
    s = _sse_open(srv, srv.token)
    try:
        _sse_read_until(s, b'"hello"')
        assert api.bus.subscriber_count == 1
        srv.stop()
        s.settimeout(5)
        rest = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            rest += chunk
        assert api.bus.subscriber_count == 0
    finally:
        s.close()


def test_coalesce_and_resync() -> None:
    evs = [
        {"type": "job", "job": {"id": "a", "done": 1}}, {"type": "page", "job_id": "a", "index": 0, "stage": "detected"},
        {"type": "job", "job": {"id": "b", "done": 1}}, {"type": "page", "job_id": "a", "index": 0, "stage": "rendered"},
        {"type": "job", "job": {"id": "a", "done": 2}}, {"type": "engine", "running": True},
        {"type": "engine", "running": False}, "垃圾", {"type": "op", "op": "export"},
    ] + [{"type": "log", "message": str(i)} for i in range(80)]
    out = coalesce_events(evs, max_logs=10)
    jobs = [e["job"] for e in out if e["type"] == "job"]
    assert sorted((j["id"], j["done"]) for j in jobs) == [("a", 2), ("b", 1)]
    assert [e["stage"] for e in out if e["type"] == "page"] == ["rendered"]
    assert [e["running"] for e in out if e["type"] == "engine"] == [False]
    assert sum(e["type"] == "log" for e in out) == 10 and any(e["type"] == "op" for e in out)
    sub = Subscription(maxlen=16)
    for i in range(40):
        sub.put({"type": "log", "message": str(i)})
    batch = sub.get_batch(0.5, 0)
    assert batch and batch[0] == {"type": "resync"}
    assert sub.get_batch(0.05, 0) == []
    sub.close()
    assert sub.get_batch(0.05, 0) is None


# --------------------------------------------------------------------------------------
# /img 与字体预览
# --------------------------------------------------------------------------------------
def test_img_route(fake, tmp_path: Path) -> None:
    api, srv, h = fake
    img = np.full((300, 200, 3), 200, np.uint8)
    jpg, tif = tmp_path / "页 1.jpg", tmp_path / "p.tiff"
    imageio.write(str(jpg), img)
    tif.write_bytes(imageio.encode(img, "png"))   # 浏览器不认的扩展名 → 服务器转码成 JPEG
    api.images[("job1", 0, "src")] = str(jpg)
    api.images[("job1", 1, "out")] = str(tif)
    t = srv.token
    status, headers, body = h.request("GET", f"/img/job1/0/src?t={t}")
    assert status == 200 and headers["content-type"] == "image/jpeg" and body[:2] == b"\xff\xd8"
    assert headers["cache-control"] == "no-cache"
    status, headers, body = h.request("GET", f"/img/job1/0/src?t={t}&v=3&w=64")
    assert status == 200 and "immutable" in headers["cache-control"]
    assert max(imageio.decode(body).shape[:2]) <= 64
    status, headers, body = h.request("GET", f"/img/job1/1/out?t={t}")
    assert status == 200 and headers["content-type"] == "image/jpeg" and imageio.decode(body).shape[:2] == (300, 200)
    assert h.request("GET", f"/img/job1/0/src?t=bad")[0] == 403
    assert h.request("GET", f"/img/job1/0/src")[0] == 403
    assert h.request("GET", f"/img/job1/9/src?t={t}")[0] == 404
    assert h.request("GET", f"/img/job1/0/evil?t={t}")[0] == 404
    assert h.request("GET", f"/img/..%2f..%2fx/0/src?t={t}")[0] == 404
    assert h.request("GET", f"/img/job1/-1/src?t={t}")[0] == 404


def test_font_preview_route(fake) -> None:
    _, srv, h = fake
    status, headers, body = h.request("GET", f"/font-preview?id=ok&t={srv.token}")
    assert status == 200 and headers["content-type"] == "image/png" and body[:4] == b"\x89PNG"
    assert h.request("GET", f"/font-preview?id=nope&t={srv.token}")[0] == 404
    assert h.request("GET", "/font-preview?id=ok&t=bad")[0] == 403


# --------------------------------------------------------------------------------------
# Api + 演示引擎（端到端）
# --------------------------------------------------------------------------------------
def test_real_ui_files_offline() -> None:
    d = Path(ui_dir())
    html = (d / "index.html").read_text(encoding="utf-8")
    assert "app.css" in html and "app.js" in html and "一键汉化漫画" in html
    for name in ("index.html", "app.css", "app.js"):
        text = (d / name).read_text(encoding="utf-8")
        for bad in ('src="http', "src='http", 'href="http', "@import url(http", "cdn.", "googleapis"):
            assert bad not in text, f"{name} 引用了外部资源：{bad}"


def test_bootstrap_and_static(demo) -> None:
    api, srv, h, _ = demo
    data = h.ok("bootstrap")
    assert data["app"]["title"] == f"一键汉化漫画 {data['app']['version']}" and data["app"]["demo"] is True
    assert data["app"]["window"] is False
    assert set(data["settings"]) >= {"general", "hardware", "translate", "typeset", "output"}
    assert data["enums"]["hardware.profile"] == ["fast", "balanced", "quality"]
    assert len(data["jobs"]) == 5 and data["fonts"] and data["engines"]
    assert "·" in data["hardware"]["chip"]
    for path in ("/ui/index.html", "/ui/app.css", "/ui/app.js"):
        assert h.request("GET", path)[0] == 200


def test_page_image_and_editor_roundtrip(demo) -> None:
    api, srv, h, _ = demo
    done = _job_by_status(api, "done")
    t = srv.token
    for kind, magic in (("src", b"\xff\xd8"), ("out", b"\xff\xd8"), ("thumb", b"\xff\xd8"), ("clean", b"\x89PNG")):
        status, _, body = h.request("GET", f"/img/{done['id']}/0/{kind}?t={t}")
        assert status == 200 and body[:len(magic)] == magic, kind
    assert h.request("GET", f"/img/{done['id']}/0/src?t={t}&w=160")[0] == 200
    pages = h.ok("list_pages", job_id=done["id"])
    assert len(pages) == done["total"] and all(p["stage"] == "rendered" for p in pages)
    page = next(p for i in range(done["total"])
                if (p := h.ok("get_page", job_id=done["id"], index=i))["blocks"])
    index = page["index"]
    assert page["blocks"] and page["width"] > 0
    v0 = page["version"]
    blocks = page["blocks"]
    blocks[0]["translation"] = "校对后的译文"
    blocks[0]["fill"] = "#ff0000"
    blocks[0]["font_size"] = "abc"          # 脏数据 → 自动
    page2 = h.ok("update_page", job_id=done["id"], index=index, blocks=blocks)
    assert page2["version"] > v0 and page2["blocks"][0]["translation"] == "校对后的译文"
    assert page2["blocks"][0]["fill"] == [255, 0, 0] and page2["blocks"][0]["font_size"] is None
    page3 = h.ok("retranslate_page", job_id=done["id"], index=index)
    assert page3["version"] > page2["version"]
    assert "找不到这个任务" in h.api("get_page", job_id="nope", index=0)["error"]
    assert "页码" in h.api("get_page", job_id=done["id"], index=99999)["error"]
    assert not h.api("get_page", job_id="../x", index=0)["ok"]
    assert not h.api("update_page", job_id=done["id"], index=0, blocks="x")["ok"]
    assert "位置" in h.api("update_page", job_id=done["id"], index=0, blocks=[{"id": 0}])["error"]


def test_background_export_and_ops(demo) -> None:
    api, _, h, opened = demo
    done = _job_by_status(api, "done")
    events: list[dict] = []
    api.bus.add_listener(lambda e: events.append(e) if e.get("type") == "op" else None)
    res = h.ok("export", job_id=done["id"])
    assert res["started"] is True
    assert api.wait_ops(15)
    final = [e for e in events if e.get("state") in ("done", "error")]
    assert final and final[-1]["state"] == "done" and os.path.isdir(final[-1]["path"])
    assert h.ok("rerender_all", job_id=done["id"])["started"]
    assert api.wait_ops(10)
    out = final[-1]["path"]
    assert h.ok("reveal", path=out)["path"] == os.path.abspath(out)
    assert opened[-1] == (os.path.abspath(out), True)
    queued = _job_by_status(api, "queued")
    assert h.ok("export", job_id=queued["id"])["started"]          # 后台失败 → 推 error 事件（中文原因）
    assert api.wait_ops(10)
    last = [e for e in events if e.get("job_id") == queued["id"] and e.get("state") != "start"][-1]
    assert last["state"] == "error" and "还没有完成的页面" in last["message"]


def test_job_controls(demo) -> None:
    api, _, h, _ = demo
    removed: list[dict] = []
    api.bus.add_listener(lambda e: removed.append(e) if e.get("type") == "removed" else None)
    res = h.ok("add_paths", paths=["D:\\漫画\\新书 第1卷.cbz", "D:\\漫画\\新书 第1卷.cbz", "", 5])
    assert len(res["jobs"]) == 1 and res["jobs"][0]["title"] == "新书 第1卷"
    jid = res["jobs"][0]["id"]
    assert h.ok("pause", job_id=jid)["paused"]
    assert h.ok("resume", job_id=jid)["resumed"]
    assert h.ok("cancel", job_id=jid)["canceled"]
    assert h.ok("pause")["paused"] and h.ok("start")["running"]
    err = _job_by_status(api, "error")
    assert "读不出页面" in h.api("retry", job_id=err["id"])["error"]
    assert h.ok("remove", job_id=jid, delete_files=True)["removed"] == jid
    assert removed and removed[-1]["job_id"] == jid
    assert all(j["id"] != jid for j in h.ok("list_jobs"))
    assert not h.api("add_paths", paths={"x": 1})["ok"]


def test_settings_and_reset_keep_glossary(demo) -> None:
    api, _, h, _ = demo
    h.ok("save_glossary", items=[{"src": "ルフィ", "dst": "路飞"}, {"src": "  ", "dst": "空"}, "垃圾"])
    s = h.ok("save_settings", patch={"hardware": {"profile": "quality"}, "unknown": {"x": 1}})
    assert s["hardware"]["profile"] == "quality" and "unknown" not in s
    s = h.ok("save_settings", patch={"hardware": {"profile": "超快"}})        # 非法枚举 → 默认
    assert s["hardware"]["profile"] == "balanced"
    h.ok("save_settings", patch={"translate": {"engines": ["edge", "google"], "concurrency": 99}})
    assert h.ok("get_settings")["translate"]["engines"] == ["edge", "google"]
    assert h.ok("get_settings")["translate"]["concurrency"] == 8
    s = h.ok("reset_settings", section="translate")
    assert s["translate"]["engines"] == DEFAULTS["translate"]["engines"]
    assert s["translate"]["local"] == DEFAULTS["translate"]["local"]
    assert s["translate"]["glossary"] == [{"src": "ルフィ", "dst": "路飞", "note": ""}]
    s = h.ok("reset_settings")
    assert s["translate"]["glossary"][0]["dst"] == "路飞"
    assert "设置分组" in h.api("reset_settings", section="nope")["error"]
    assert not h.api("save_settings", patch=[1])["ok"]


def test_glossary_csv_roundtrip(demo) -> None:
    api, _, h, _ = demo
    text = "\ufeff原文,译文,备注\nルフィ,路飞,主角\n\"ゾロ, 剑士\",索隆,\n,空原文,\n"
    res = h.ok("import_glossary", text=text)
    assert res["added"] == 2 and res["updated"] == 0
    res = h.ok("import_glossary", text="ルフィ\t路飞（新）\nナミ\t娜美")      # TSV、无表头
    assert res["added"] == 1 and res["updated"] == 1
    items = h.ok("get_glossary")
    assert [i["src"] for i in items] == ["ルフィ", "ゾロ, 剑士", "ナミ"] and items[0]["dst"] == "路飞（新）"
    exp = h.ok("export_glossary")
    assert exp["csv"].startswith("\ufeff原文,译文,备注") and exp["path"] == "" and exp["count"] == 3
    assert parse_glossary_csv(exp["csv"]) == items
    assert "有效的词条" in h.api("import_glossary", text="只有一列\n")["error"]
    assert h.ok("import_glossary", text="a,b", replace=True)["items"] == [{"src": "a", "dst": "b", "note": ""}]
    assert clean_glossary([{"src": "x", "dst": "1"}, {"src": "x", "dst": "2"}]) == [{"src": "x", "dst": "2", "note": ""}]
    assert dump_glossary_csv([]).count("\r\n") == 1


def test_engines_fonts_hardware(demo) -> None:
    api, srv, h, _ = demo
    res = h.ok("test_engine", engine_id="google")
    assert res["ok"] is True and res["message"]
    res = h.ok("test_engine", engine_id="llm")
    assert res["ok"] is False and "11434" in res["message"]
    assert "引擎名称" in h.api("test_engine", engine_id="../../x")["error"]
    fonts = h.ok("list_fonts")
    ids = {f["id"] for f in fonts}
    assert {"noto-sans-sc-bold", "zcool-kuaile", "lxgw-wenkai"} <= ids
    status, headers, body = h.request("GET", f"/font-preview?id=zcool-kuaile&t={srv.token}&text=%E4%BD%A0%E5%A5%BD")
    assert status == 200 and body[:4] == b"\x89PNG"
    hw = h.ok("hardware")
    assert hw["chip"] and hw["recommend"]["variant"] in ("nvidia", "gpu", "cpu") and hw["version"]
    assert "对话框" in h.api("add_font")["error"]
    assert "ttf" in h.api("add_font", path=__file__)["error"]


# --------------------------------------------------------------------------------------
# 本地大模型 / API 服务（演示引擎的假实现 + 真实 LocalModels 的离线部分）
# --------------------------------------------------------------------------------------
def _wait_llm_state(api: Api, state: str, timeout: float = 15.0) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        dl = api.llm_status()["download"] or {}
        if dl.get("state") == state:
            return dl
        time.sleep(0.05)
    raise AssertionError(f"下载状态没有变成 {state}：{api.llm_status()['download']}")


def test_llm_status_and_fake_download(demo) -> None:
    api, _, h, _ = demo
    api.engine.local_models.DOWNLOAD_S = 1.0
    st = h.ok("llm_status")
    assert set(st) >= {"hardware", "recommend", "backends", "models", "selected", "active", "installed", "download"}
    assert st["selected"] == "auto" and st["active"] == "" and st["installed"] == [] and st["download"] is None
    assert {"gpu", "vram_gb", "ram_gb", "cpu_threads"} <= set(st["hardware"])
    ids = [m["id"] for m in st["models"]]
    assert "qwen3.5-4b-q4" in ids
    assert all({"name", "size_gb", "need_gb", "installed", "recommended", "partial_bytes"} <= set(m) for m in st["models"])
    rec = st["recommend"]
    assert rec and rec["model"]["id"] in ids and rec["reason"] and rec["backend"] in ("cuda", "vulkan", "cpu")
    assert sum(1 for m in st["models"] if m["recommended"]) == 1
    mid = rec["model"]["id"]
    sub = api.bus.subscribe()
    try:
        assert h.ok("llm_download", model_id=mid) == {"started": True, "model_id": mid}
        assert "正在下载" in h.api("llm_download", model_id="qwen3.5-2b-q4")["error"]
        dl = _wait_llm_state(api, "done")
        assert dl["model_id"] == mid and dl["done"] == dl["total"] > 0 and "下载完成" in dl["message"]
        events: list[dict] = []
        end = time.monotonic() + 3
        while time.monotonic() < end and not any(e.get("state") == "done" for e in events):
            events += [e for e in (sub.get_batch(0.2) or []) if e.get("type") == "llm"]
        states = [e["state"] for e in events]
        assert states[0] == "downloading" and states[-1] == "done" and len(states) >= 3
        assert all({"model_id", "done", "total", "speed", "message"} <= set(e) for e in events)
    finally:
        api.bus.unsubscribe(sub)
    st = h.ok("llm_status")
    assert st["installed"] == [mid] and st["active"] == mid and st["download"]["state"] == "done"
    assert "已经下载好" in h.api("llm_download", model_id=mid)["error"]
    assert "模型名称" in h.api("llm_download", model_id="../../evil")["error"]
    assert "没有这个模型" in h.api("llm_download", model_id="nope-1b")["error"]
    h.ok("save_settings", patch={"translate": {"local": {"model": mid}}})
    assert h.ok("llm_status")["selected"] == mid
    st = h.ok("llm_delete", model_id=mid)                 # 删掉当前指定的模型 → 自动回到「自动推荐」
    assert st["installed"] == [] and st["selected"] == "auto" and st["active"] == ""
    assert h.ok("get_settings")["translate"]["local"]["model"] == "auto"


def test_llm_cancel_resume_import_and_api(demo, tmp_path: Path) -> None:
    api, _, h, _ = demo
    models = api.engine.local_models
    models.DOWNLOAD_S = 30.0
    assert h.ok("llm_cancel") == {"canceling": False}
    h.ok("llm_download", model_id="qwen3.5-2b-q4")
    time.sleep(0.6)
    assert h.ok("llm_cancel") == {"canceling": True}
    dl = _wait_llm_state(api, "canceled")
    assert 0 < dl["done"] < dl["total"] and "取消" in dl["message"]
    row = next(m for m in h.ok("llm_status")["models"] if m["id"] == "qwen3.5-2b-q4")
    assert row["partial_bytes"] == dl["done"] and not row["installed"]
    res = h.ok("test_engine", engine_id="local")
    assert res["ok"] is False and "下载" in res["message"]
    models.DOWNLOAD_S = 0.5
    h.ok("llm_download", model_id="qwen3.5-2b-q4")         # 断点续传：从已下载的部分接着走
    assert _wait_llm_state(api, "done")["done"] == row["size"]
    res = h.ok("test_engine", engine_id="local")
    assert res["ok"] is True and res["message"] and res["ms"] >= 0
    # 导入自己的 GGUF：浏览器模式没有对话框；直接给路径可以导入，并设为当前模型
    assert "对话框" in h.api("llm_import")["error"]
    gg = tmp_path / "我的模型.gguf"
    gg.write_bytes(b"GGUF" + b"\0" * 64)
    res = h.ok("llm_import", path=str(gg))
    cid = "custom:我的模型.gguf"
    assert res["imported"]["id"] == cid and res["status"]["selected"] == cid and res["status"]["active"] == cid
    assert "gguf" in h.api("llm_import", path=str(tmp_path / "x.bin"))["error"]
    assert cid not in h.ok("llm_delete", model_id=cid)["installed"]
    # API 服务
    presets = h.ok("api_presets")
    assert [p["id"] for p in presets][:2] == ["deepseek", "siliconflow"]
    assert all({"id", "name", "base_url", "model"} <= set(p) for p in presets)
    assert h.ok("test_engine", engine_id="api")["ok"] is False
    assert "Key" in h.api("api_models", provider="deepseek")["error"]
    res = h.ok("api_models", provider="deepseek", api_key="sk-test")
    assert res["base_url"].startswith("https://api.deepseek.com") and "deepseek-chat" in res["models"]
    assert "地址" in h.api("api_models", base_url="ftp://x", api_key="k")["error"]
    h.ok("save_settings", patch={"translate": {"api": {"enabled": True, "api_key": "sk-test", "model": "deepseek-chat"}}})
    assert h.ok("api_models")["models"]                     # 地址 / Key 留空 → 用已保存的服务商和 Key
    assert h.ok("test_engine", engine_id="api")["ok"] is True
    assert h.ok("test_engine", engine_id="llm")["ok"] is False


def test_real_local_models_offline(tmp_path: Path, hanhua_data: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实 LocalModels（不联网）：状态、导入 GGUF、id 校验、删除。"""
    from hanhua.app import LocalModels, check_model_id
    from hanhua.types import HanhuaError

    monkeypatch.setenv("HANHUA_LLM_DIR", str(tmp_path / "llm"))
    store = SettingsStore(str(hanhua_data / "settings.json"))
    api = Api(store, UnavailableEngine("测试"), EventBus(), local_models=LocalModels())
    st = api.llm_status()
    assert st["models"] and st["selected"] == "auto" and st["installed"] == [] and st["download"] is None
    assert {m["id"] for m in api.api_presets()} >= {"deepseek", "openai", "custom"}
    bad = tmp_path / "fake.gguf"
    bad.write_bytes(b"NOPE")
    with pytest.raises(HanhuaError, match="GGUF"):
        api.llm_import(path=str(bad))
    good = tmp_path / "custom-q4.gguf"
    good.write_bytes(b"GGUF" + b"\0" * 32)
    res = api.llm_import(path=str(good))
    cid = "custom:custom-q4.gguf"
    assert res["imported"]["id"] == cid and cid in res["status"]["installed"] and res["status"]["active"] == cid
    assert store.get()["translate"]["local"]["model"] == cid
    for bad_id in ("custom:../x.gguf", "custom:a\\b.gguf", "custom:.gguf", "custom:x.exe", "../../etc", ""):
        with pytest.raises(ApiError):
            check_model_id(bad_id)
    st = api.llm_delete(cid)
    assert cid not in st["installed"] and store.get()["translate"]["local"]["model"] == "auto"
    assert not (tmp_path / "llm" / "custom-q4.gguf").exists()


def test_open_path_safety(demo, tmp_path: Path) -> None:
    api, _, h, opened = demo
    assert "不存在" in h.api("open_path", path=str(tmp_path / "没有.cbz"))["error"]
    exe = tmp_path / "病毒.exe"
    exe.write_bytes(b"MZ")
    assert "安全" in h.api("open_path", path=str(exe))["error"]
    ok_file = tmp_path / "成品.cbz"
    ok_file.write_bytes(b"PK")
    assert h.ok("open_path", path=str(ok_file))["path"] == str(ok_file)
    assert h.ok("reveal", path=str(tmp_path / "已删除.cbz"))["path"] == str(tmp_path)   # 文件没了 → 打开所在文件夹
    assert opened == [(str(ok_file), False), (str(tmp_path), True)]
    assert not h.api("open_path", path=123)["ok"]
    assert h.ok("ui_error", message="TypeError: x", source="app.js", line=3)["logged"]


def test_dialog_hooks(demo, tmp_path: Path) -> None:
    api, _, h, _ = demo
    assert "浏览器调试模式" in h.api("pick_files")["error"]

    class FakeDialogs:
        def open_files(self, file_types, multiple=True):
            return [str(tmp_path / "a.cbz")]

        def open_folder(self):
            return str(tmp_path)

        def save_file(self, filename, file_types):
            return str(tmp_path / "导出的术语表")

    api.dialogs = FakeDialogs()
    res = h.ok("pick_files")
    assert res["paths"] == [str(tmp_path / "a.cbz")] and len(res["jobs"]) == 1
    assert h.ok("pick_folder")["jobs"][0]["kind"] == "folder"
    s = h.ok("pick_output_dir", field="dir")
    assert s["settings"]["output"]["dir"] == str(tmp_path)
    assert not h.api("pick_output_dir", field="../etc")["ok"]
    h.ok("save_glossary", items=[{"src": "勇者", "dst": "勇者大人"}])
    exp = h.ok("export_glossary")
    assert exp["path"].endswith("导出的术语表.csv")
    assert Path(exp["path"]).read_text(encoding="utf-8-sig").startswith("原文,译文,备注")


def test_unavailable_engine(hanhua_data: Path, tmp_path: Path) -> None:
    store = SettingsStore(str(hanhua_data / "settings.json"))
    api = Api(store, UnavailableEngine("处理引擎加载失败：缺少模型"), engine_error="处理引擎加载失败：缺少模型")
    srv = AppServer(api, ui_dir()).start()
    h = Http(srv)
    try:
        boot = h.ok("bootstrap")
        assert boot["jobs"] == [] and "缺少模型" in boot["engine"]["error"]
        f = tmp_path / "书.cbz"
        f.write_bytes(b"PK")
        assert "缺少模型" in h.api("add_paths", paths=[str(f)])["error"]
        res = h.ok("add_paths", paths=[str(tmp_path / "不存在.cbz")])          # 正式模式：不存在的文件被跳过
        assert res["jobs"] == [] and res["skipped"][0]["reason"]
        assert h.request("GET", f"/img/abc/0/src?t={srv.token}")[0] == 404
    finally:
        srv.stop()


def test_clean_blocks_and_names() -> None:
    blocks = Api._clean_blocks([
        {"id": 3, "box": [1, 2, 30.7, 40], "fill": [300, -5, 10], "stroke": "#00ff00", "font_size": 9999,
         "stroke_width": 0.2, "out_direction": "x", "align": "start", "font": "  ", "kind": "?", "bubble_box": [0, 0, 50, 50],
         "translation": "字" * 9000, "skip": 1},
        "垃圾",
    ])
    b = blocks[0]
    assert b["box"] == [1, 2, 30, 40] and b["fill"] == [255, 0, 10] and b["stroke"] == [0, 255, 0]
    assert b["font_size"] is None and b["stroke_width"] == 0.2 and b["out_direction"] is None and b["align"] == "start"
    assert b["font"] is None and b["kind"] == "bubble" and len(b["translation"]) == 4000 and b["skip"] is True
    with pytest.raises(ApiError):
        Api._clean_blocks({"a": 1})
    assert short_gpu_name("NVIDIA GeForce RTX 5060 Ti") == "RTX 5060 Ti"
    assert short_gpu_name("Intel(R) Arc(TM) A770 Graphics") == "Arc A770 Graphics"
    assert short_gpu_name("AMD Radeon RX 7600") == "Radeon RX 7600"


def test_concurrent_api_calls(demo) -> None:
    api, _, h, _ = demo
    errors: list[str] = []

    def worker() -> None:
        try:
            for _ in range(10):
                assert isinstance(h.ok("list_jobs"), list)
                h.ok("bootstrap")
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=worker) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert not errors


# --------------------------------------------------------------------------------------
# 命令行：--demo --no-window 打印 URL，浏览器可直接打开
# --------------------------------------------------------------------------------------
def test_cli_demo_no_window(project_root: Path, hanhua_data: Path) -> None:
    env = dict(os.environ, HANHUA_DATA=str(hanhua_data), PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "hanhua.app", "--demo", "--no-window"], cwd=str(project_root),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
    )
    try:
        url = ""
        end = time.monotonic() + 60
        while time.monotonic() < end and p.poll() is None:
            line = p.stdout.readline().decode("utf-8", "replace").strip()   # type: ignore[union-attr]
            if line.startswith("http://127.0.0.1:"):
                url = line
                break
        assert url, "没有打印出地址"
        import urllib.request

        with urllib.request.urlopen(url, timeout=10) as r:
            assert r.status == 200 and "一键汉化漫画".encode("utf-8") in r.read()
        assert (hanhua_data / "logs" / "app.log").is_file()
    finally:
        p.kill()
        p.wait(10)

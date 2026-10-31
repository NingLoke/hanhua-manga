"""translate 模块测试。

离线：textproc / cache / sfx_dict / 引擎解析 / 本地假 HTTP 服务（Google 表单接口、OpenAI 兼容接口）/
     TranslationService（假引擎：打包、二分、熔断、退避、全挂、取消、术语、缓存、并发）。
联网（@pytest.mark.network，HANHUA_OFFLINE=1 时跳过）：每个免费引擎翻 p012 三句 + 10 句常见对白，
     结果写到 tests/out/translate/network_<引擎>.json。
"""
from __future__ import annotations

import copy
import json
import os
import re
import sqlite3
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import pytest

from hanhua.config import DEFAULTS
from hanhua.translate import TranslationService, sfx_dict
from hanhua.translate.cache import TranslationCache, cache_key_text
from hanhua.translate.engines import (BaseEngine, EngineError, GoogleFree, LocalLLM, SplitNeeded, _jwt_ttl,
                                      norm_base_url, norm_source, norm_target, parse_numbered)
from hanhua.translate.service import _Breaker, _Limiter, _pack
from hanhua.translate.textproc import (Glossary, apply_stutter, looks_untranslated, normalize_source,
                                       postprocess_zh, punct_passthrough, split_stutter)
from hanhua.types import Cancelled, TextBlock, TranslationUnavailable

P012 = ["そんなことはありません", "マ…マネージャー！？なんでこんな所に…！？", "今、何時だって思ってるの！"]
COMMON = ["ちょっと待って！", "何してるの？", "ありがとう、助かったよ。", "大丈夫か！？", "もう遅いよ…",
          "お前には関係ない！", "本当にごめんなさい。", "一緒に帰ろう。", "嘘でしょ…？", "絶対に負けない！"]
_KANA = re.compile(r"[ぁ-ゖァ-ヺー]")


# ---- 公共工具 -----------------------------------------------------------------------------------
def fake_tr(t: str) -> str:
    """假翻译：去掉假名、前面加「译」（占位符原样保留），保证通过「像没翻」校验。"""
    return "译" + _KANA.sub("", t)


class FakeEngine(BaseEngine):
    def __init__(self, eid: str = "fake", fn: Callable[[list[str]], list[str]] | None = None, *,
                 max_items: int = 50, max_chars: int = 4000, placeholder: bool = True, delay: float = 0.0) -> None:
        super().__init__({})
        self.id = eid
        self.name = eid
        self.max_items = max_items
        self.max_chars = max_chars
        self.min_interval = 0.0
        self.placeholder_glossary = placeholder
        self.fn = fn or (lambda texts: [fake_tr(t) for t in texts])
        self.delay = delay
        self.calls: list[list[str]] = []
        self.active = 0
        self.max_active = 0
        self._lk = threading.Lock()

    def translate_batch(self, texts, src, tgt, *, context=None, glossary=None):  # type: ignore[override]
        with self._lk:
            self.calls.append(list(texts))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self.fn(list(texts))
        finally:
            with self._lk:
                self.active -= 1


def make_settings(**tr: Any) -> dict:
    s = copy.deepcopy(DEFAULTS)
    s["translate"].update({"max_retries": 1, "backoff_base": 0.0, "backoff_max": 0.05,
                           "breaker_cooldown": 60.0, "concurrency": 1})
    s["translate"].update(tr)
    return s


def svc_with(*engines: BaseEngine, cache: str = "", **tr: Any) -> TranslationService:
    return TranslationService(make_settings(**tr), cache, engines=list(engines))


def blk(i: int, text: str, **kw: Any) -> TextBlock:
    return TextBlock(id=i, box=(0, 0, 10, 10), src_text=text, **kw)


@pytest.fixture
def http_server(monkeypatch: pytest.MonkeyPatch):
    """本地假 HTTP 服务：handler(method, path, query, body, headers) -> (status, headers, bytes)。"""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    servers: list[ThreadingHTTPServer] = []

    def start(handler: Callable[..., tuple[int, dict, bytes]]) -> str:
        class H(BaseHTTPRequestHandler):
            def _go(self, method: str) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                u = urllib.parse.urlsplit(self.path)
                status, hdrs, payload = handler(method, u.path, urllib.parse.parse_qs(u.query), body, self.headers)
                self.send_response(status)
                for k, v in hdrs.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                self._go("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._go("POST")

            def log_message(self, *args: Any) -> None:
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def _json_resp(obj: Any, status: int = 200, headers: dict | None = None) -> tuple[int, dict, bytes]:
    return status, {"Content-Type": "application/json; charset=utf-8", **(headers or {})}, \
        json.dumps(obj, ensure_ascii=False).encode("utf-8")


# ---- textproc -----------------------------------------------------------------------------------
def test_textproc_basics() -> None:
    assert normalize_source("ＯＫ．．．！?", "ja") == "OK…！？"
    assert normalize_source("", "ja") == "" and normalize_source(None, "ja") == ""  # type: ignore[arg-type]
    assert split_stutter("マ…マネージャー！？") == (1, "マネージャー！？")
    assert split_stutter("な、な、なんで") == (2, "なんで")
    assert split_stutter("あ、本当だ")[0] == 0
    assert apply_stutter(1, "经理！？") == "经、经理！？"
    assert apply_stutter(1, "Wait") == "W-Wait"
    assert punct_passthrough("…") == "……" and punct_passthrough("!?") == "！？"
    assert postprocess_zh("你好 , 世界...", "こんにちは、世界…") == "你好，世界……"
    assert postprocess_zh("他说“走”。", "行け") == "他说「走」"           # 去掉机翻多补的句号 + 引号风格
    assert postprocess_zh("他说“走”", "", punct_style="western") == "他说“走”"
    assert looks_untranslated("こんにちは", "こんにちは", "ja")
    assert not looks_untranslated("こんにちは", "你好", "ja")
    g = Glossary([{"src": "ナルト", "dst": "鸣人"}, {"src": "", "dst": "x"}, "坏数据"])
    prot, dsts = g.protect("ナルトが来た！")
    assert prot == "{0}が来た！" and dsts == ["鸣人"]
    assert Glossary.restore("{0}来了！", dsts) == ("鸣人来了！", True)
    assert Glossary.restore("来了！", dsts)[1] is False


# ---- cache --------------------------------------------------------------------------------------
def test_cache_roundtrip_and_persist(tmp_path) -> None:
    p = str(tmp_path / "sub" / "cache.sqlite")
    c = TranslationCache(p)
    assert c.available
    c.put_many("google", "ja", "zh-Hans", [("こんにちは", "你好"), (" 空白  あり ", "有空白"), ("", "x"), ("a", "")])
    assert c.get("google", "ja", "zh-Hans", "こんにちは") == "你好"
    assert c.get("google", "ja", "zh-Hans", "空白 あり") == "有空白"            # 规范化 key
    assert c.get("google", "ja", "zh-Hant", "こんにちは") is None               # tgt 不同
    assert c.get("edge", "ja", "zh-Hans", "こんにちは") is None                 # 引擎不同
    assert c.get_many("google", "ja", "zh-Hans", []) == {}
    assert c.stats() == {"google": 2}
    mode = sqlite3.connect(p).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    c.close()
    c.close()                                                                     # 重复关闭无害
    assert c.get("google", "ja", "zh-Hans", "こんにちは") is None                 # 关闭后安全退化
    c2 = TranslationCache(p)
    assert c2.get("google", "ja", "zh-Hans", "こんにちは") == "你好"             # 持久化
    assert c2.delete("google") == 2 and c2.stats() == {}
    c2.close()
    assert cache_key_text("  a　 b ") == "a b"


def test_cache_corrupt_file_recovers(tmp_path) -> None:
    p = tmp_path / "cache.sqlite"
    p.write_bytes(b"this is not a sqlite database" * 100)
    c = TranslationCache(str(p))
    c.put("google", "ja", "zh-Hans", "テスト", "测试")
    assert c.get("google", "ja", "zh-Hans", "テスト") == "测试"
    assert any(".corrupt-" in f.name for f in tmp_path.iterdir())
    c.close()


def test_cache_thread_safe(tmp_path) -> None:
    c = TranslationCache(str(tmp_path / "c.sqlite"))
    errors: list[BaseException] = []

    def work(k: int) -> None:
        try:
            for i in range(60):
                c.put("e", "ja", "zh-Hans", f"文{k}-{i}", f"译{k}-{i}")
                assert c.get("e", "ja", "zh-Hans", f"文{k}-{i}") == f"译{k}-{i}"
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert c.stats() == {"e": 360}
    c.close()


# ---- sfx_dict -----------------------------------------------------------------------------------
def test_sfx_dict_size_and_variants() -> None:
    from hanhua.translate.sfx_dict import JA_SFX, _forms
    assert len({_forms(k)[0] for k, _ in JA_SFX}) >= 150
    cases = {
        "ドキドキ": "怦怦", "どきどき": "怦怦", "ﾄﾞｷﾄﾞｷ": "怦怦", "ドキッ": "怦", "どきっ！": "怦！",
        "ドーン": "轰——", "ドーーン！！": "轰——！！", "ドオオン": "轰——",
        "ドドドド": "咚咚咚咚", "ゴゴゴゴゴ": "轰轰轰轰", "ざわざわざわ": "嘈杂", "トントントン": "咚咚咚",
        "ガシャーン!?": "哐啷——！？", "シーン…": "一片寂静……", "쾅쾅": "轰轰", "BOOOOM!": "轰！", "hahaha": "哈哈哈",
    }
    for src, want in cases.items():
        assert sfx_dict.lookup(src) == want, src
    assert sfx_dict.lookup("ガタン", "zh-Hant") == "哐噹"
    assert sfx_dict.lookup("ドーン", "zh-Hant") == "轟——"
    for bad in ("", None, 123, "そんなことはありません", "!!", "ズモモモ", "\x00﻿"):
        assert sfx_dict.lookup(bad) is None  # type: ignore[arg-type]


# ---- 引擎解析 / HTTP 层 --------------------------------------------------------------------------
def test_engine_helpers() -> None:
    assert norm_target("zh-Hant") == "zh-Hant" and norm_target("zh-TW") == "zh-Hant"
    assert norm_target(None) == "zh-Hans" and norm_target("zh-CN") == "zh-Hans"
    assert norm_source("JA") == "ja" and norm_source("zh-TW") == "zh" and norm_source("xx") == "auto"
    assert norm_base_url("") == "http://127.0.0.1:11434/v1"
    assert norm_base_url("127.0.0.1:1234") == "http://127.0.0.1:1234/v1"
    assert norm_base_url("http://h:1/v1/chat/completions") == "http://h:1/v1"
    assert norm_base_url("https://x.com/api/v1/") == "https://x.com/api/v1"
    assert _jwt_ttl("not-a-jwt", 480.0) == 480.0
    assert GoogleFree.parse(["甲", "乙"], 2) == ["甲", "乙"]
    assert GoogleFree.parse([["甲", "ja"], ["乙", "ko"]], 2) == ["甲", "乙"]
    assert GoogleFree.parse("甲", 1) == ["甲"]
    assert GoogleFree.parse(["甲", "ja"], 1) == ["甲"]
    with pytest.raises(SplitNeeded):
        GoogleFree.parse(["甲"], 2)
    with pytest.raises(EngineError):
        GoogleFree.parse({"x": 1}, 1)


def test_parse_numbered_variants() -> None:
    assert parse_numbered('<think>嗯……</think>\n```json\n{"1": "甲", "2": "乙"}\n```', 2) == ["甲", "乙"]
    assert parse_numbered('好的：{"translations": ["甲", "乙"]}', 2) == ["甲", "乙"]
    assert parse_numbered('["甲", "乙"]', 2) == ["甲", "乙"]
    assert parse_numbered('"1": "甲\\"引号\\"", "2": "乙"  （不完整的 JSON）', 2) == ['甲"引号"', "乙"]
    assert parse_numbered("你好", 1) == ["你好"]
    for bad, n in (('{"1": "甲"}', 2), ("抱歉，我无法翻译。\n第二行", 1), ("", 1), ('["甲"]', 2)):
        with pytest.raises(SplitNeeded):
            parse_numbered(bad, n)


def test_google_engine_against_fake_server(http_server, monkeypatch) -> None:
    seen: dict[str, Any] = {}
    state = {"mode": "ok"}

    def handler(method, path, query, body, headers):
        if state["mode"] == "429":
            return _json_resp({"error": "busy"}, 429, {"Retry-After": "7"})
        if state["mode"] == "404":
            return _json_resp({}, 404)
        if state["mode"] == "html":
            return 200, {"Content-Type": "text/html"}, b"<html>sorry</html>"
        form = urllib.parse.parse_qs(body.decode("utf-8"), keep_blank_values=True)
        seen.update(method=method, path=path, query=query, q=form.get("q", []))
        return _json_resp([("译" + urllib.parse.unquote(q)) for q in form.get("q", [])])

    base = http_server(handler)
    monkeypatch.setattr(GoogleFree, "URL", base + "/translate_a/t")
    eng = GoogleFree({"translate": {"timeout": 5}})
    out = eng.translate_batch(["こんにちは", "A&B <i>", "{0}が来た"], "ja", "zh-Hant")
    assert seen["method"] == "POST" and seen["query"]["client"] == ["gtx"] and seen["query"]["tl"] == ["zh-TW"]
    assert seen["q"] == ["こんにちは", "A&amp;B &lt;i&gt;", "{0}が来た"]        # HTML 转义后发送
    assert out == ["译こんにちは", "译A&B <i>", "译{0}が来た"]                    # 收到后反转义
    state["mode"] = "429"
    with pytest.raises(EngineError) as ei:
        eng.translate_batch(["x"], "ja", "zh-Hans")
    assert ei.value.status == 429 and ei.value.retryable and ei.value.retry_after == 7.0
    state["mode"] = "404"
    with pytest.raises(EngineError) as ei:
        eng.translate_batch(["x"], "ja", "zh-Hans")
    assert ei.value.fatal and not ei.value.retryable
    state["mode"] = "html"
    with pytest.raises(EngineError):
        eng.translate_batch(["x"], "ja", "zh-Hans")
    eng.close()


def test_engine_offline_error_is_chinese(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    eng = LocalLLM({"translate": {"timeout": 3, "llm": {"base_url": "http://127.0.0.1:9", "model": "m"}}})
    with pytest.raises(EngineError) as ei:
        eng.translate_batch(["テスト"], "ja", "zh-Hans")
    assert "本地大模型" in ei.value.message and ei.value.retryable
    eng.close()


def _llm_handler(state: dict) -> Callable[..., tuple[int, dict, bytes]]:
    def handler(method, path, query, body, headers):
        if method == "GET" and path == "/v1/models":
            return _json_resp({"data": [{"id": "fake-llm"}]})
        if method == "POST" and path == "/v1/chat/completions":
            req = json.loads(body.decode("utf-8"))
            state.setdefault("requests", []).append(req)
            state["calls"] = state.get("calls", 0) + 1
            if state.get("bad_first") and state["calls"] == 1:
                content = "抱歉，我不太明白。"
            else:
                user = req["messages"][1]["content"]
                todo = json.loads(re.search(r"待翻译：\n(\{.*\})", user, re.S).group(1))
                ans = {k: fake_tr(v) for k, v in todo.items()}
                content = "<think>先想一想</think>\n```json\n" + json.dumps(ans, ensure_ascii=False) + "\n```"
            return _json_resp({"choices": [{"message": {"role": "assistant", "content": content}}]})
        return _json_resp({"error": "not found"}, 404)
    return handler


def test_local_llm_engine_fake_server(http_server) -> None:
    state: dict = {"bad_first": True}
    base = http_server(_llm_handler(state))
    settings = make_settings(llm={"enabled": True, "base_url": base, "model": "fake-llm", "api_key": "k"})
    eng = LocalLLM(settings)
    assert eng.base_url == base + "/v1" and eng.cache_id == "llm:fake-llm"
    eng.probe()
    out = eng.translate_batch(["こんにちは", "ナルトが来た"], "ja", "zh-Hans",
                              context=["前のセリフ"], glossary=[("ナルト", "鸣人")])
    assert out == ["译", "译来"]
    assert state["calls"] == 2                                                  # 第一次格式不对 → 提醒后重试
    user = state["requests"][0]["messages"][1]["content"]
    assert "ナルト → 鸣人" in user and "前のセリフ" in user and "简体中文" in state["requests"][0]["messages"][0]["content"]
    bad = LocalLLM(make_settings(llm={"base_url": base, "model": "missing-model"}))
    with pytest.raises(EngineError) as ei:
        bad.probe()
    assert "missing-model" in ei.value.message and ei.value.fatal
    eng.close()
    bad.close()


def test_service_with_local_llm_glossary_inline(http_server) -> None:
    state: dict = {}
    base = http_server(_llm_handler(state))
    settings = make_settings(llm={"enabled": True, "base_url": base, "model": "fake-llm"},
                             glossary=[{"src": "ナルト", "dst": "鸣人"}])
    llm = LocalLLM(settings)
    llm.max_items = 2
    svc = TranslationService(settings, "", engines=[llm])
    out = svc.translate_texts(["ナルトが来た！", "こんにちは", "また明日", "さようなら"], "ja")
    assert all(out) and out[0].startswith("译")
    first_user = state["requests"][0]["messages"][1]["content"]
    assert "ナルト → 鸣人" in first_user and "{0}" not in first_user           # 本地大模型：术语写进提示词，不用占位符
    assert "前文" in state["requests"][1]["messages"][1]["content"]           # 第二批带前文
    svc.close()


# ---- service（假引擎）----------------------------------------------------------------------------
def test_service_static_engines_and_order() -> None:
    infos = TranslationService.engines()
    # 本地优先：内置本地大模型 → 填 Key 的 API → 免费在线（测试环境没有下载模型，local 会被自动跳过）
    assert [i["id"] for i in infos] == ["local", "api", "google", "edge", "tencent", "llm"]
    for i in infos:
        assert {"id", "name", "kind", "batch"} <= set(i) and i["kind"] in ("free", "local", "api")
    s = make_settings(engines=["edge", "bogus", 3, "google", "google", "llm"])
    svc = TranslationService(s, "")
    assert svc.engine_order == ["edge", "google"]                              # llm 未开启时剔除
    svc.close()
    s["translate"]["llm"]["enabled"] = True
    svc = TranslationService(s, "")
    assert svc.engine_order == ["edge", "google", "llm"]
    svc.close()
    s["translate"]["engines"] = []
    s["translate"]["llm"]["enabled"] = False
    svc = TranslationService(s, "")
    assert svc.engine_order == ["google"]
    svc.close()


def test_translate_blocks_rules() -> None:
    eng = FakeEngine()
    svc = svc_with(eng)
    pages = [[
        blk(0, "マ…マネージャー！？"),
        blk(1, "…"),
        blk(2, "ドキドキ♡", is_sfx=True),
        blk(3, "こんにちは", translation="人工译文"),
        blk(4, "さようなら", skip=True),
        blk(5, "   "),
        blk(6, "何してるの？"),
    ], [blk(0, "「ありがとう」")]]
    svc.translate_blocks(pages, "ja")
    p = pages[0]
    assert p[0].translation == "译、译！？"                                    # 口吃补回
    assert p[1].translation == "……"                                          # 纯标点直接搬运
    assert p[2].translation == "怦怦♡"                                       # 拟声词词典 + 装饰符号
    assert p[3].translation == "人工译文"                                    # 不覆盖人工修改
    assert p[4].translation == "" and p[5].translation == ""                # skip / 空原文
    assert p[6].translation == "译何？"
    assert pages[1][0].translation == "「译」"
    sent = [t for c in eng.calls for t in c]
    assert "ドキドキ" not in sent and "…" not in sent and "こんにちは" not in sent
    assert eng.calls == [["マネージャー！？", "何してるの？", "「ありがとう」"]]   # 两页合成一批
    svc.close()


def test_sfx_mode_free_text_and_punct_style() -> None:
    eng = FakeEngine()
    svc = svc_with(eng, sfx_mode="keep", translate_free_text=False)
    page = [blk(0, "ドーン", is_sfx=True), blk(1, "看板です", kind="free"), blk(2, "セリフ")]
    svc.translate_blocks([page], "ja")
    assert [b.translation for b in page] == ["", "", "译"]
    svc.close()
    s = make_settings(target="zh-Hant")
    s["typeset"]["punct_style"] = "western"
    svc = TranslationService(s, "", engines=[FakeEngine(fn=lambda ts: ["他说「走」" for _ in ts])])
    page = [blk(0, "ドーン", is_sfx=True), blk(1, "行け")]
    svc.translate_blocks([page], "ja")
    assert page[0].translation == "轟——" and page[1].translation == "他说“走”"
    svc.close()


def test_batching_limits_and_pack() -> None:
    assert _pack(["a", "b", "c", "d"], [0, 0, 1, 1], 3, 100) == [["a", "b"], ["c", "d"]]   # 整页优先
    assert _pack(["a", "b", "c", "d"], [0, 0, 0, 0], 3, 100) == [["a", "b", "c"], ["d"]]   # 单页超限逐条
    assert _pack(["aaaa", "bbbb", "c"], [0, 1, 2], 10, 8) == [["aaaa", "bbbb"], ["c"]]       # 字符数限制
    assert _pack(["x" * 50], [0], 10, 8) == [["x" * 50]]                                    # 超长单句自成一批
    eng = FakeEngine(max_items=4, max_chars=1000)
    svc = svc_with(eng)
    texts = [f"セリフ{i}です" for i in range(10)]
    out = svc.translate_texts(texts + [texts[0]], "ja")
    assert all(out) and out[0] == out[-1]
    assert [len(c) for c in eng.calls] == [4, 4, 2]                           # 去重后 10 条，按 4 条一批
    svc.close()


def test_count_mismatch_bisects() -> None:
    def fn(ts: list[str]) -> list[str]:
        return [fake_tr(t) for t in ts] + (["多余"] if len(ts) > 2 else [])
    eng = FakeEngine(fn=fn)
    svc = svc_with(eng)
    texts = [f"文{i}です" for i in range(7)]
    out = svc.translate_texts(texts, "ja")
    assert out == [f"译文{i}" for i in range(7)]
    # 第一次整批，条数对不上后二分拆小（7 → 3/4 → 1/2/2/2），最终都对齐
    assert eng.calls[0] == texts and max(len(c) for c in eng.calls[1:]) < len(texts)
    assert min(len(c) for c in eng.calls) <= 2
    svc.close()


def test_fallback_untranslated_and_unexpected_errors() -> None:
    echo = FakeEngine("echo", fn=lambda ts: list(ts))                         # 原样退回 → 判为没翻
    boom = FakeEngine("boom", fn=lambda ts: (_ for _ in ()).throw(ValueError("bug")))
    good = FakeEngine("good")
    svc = svc_with(echo, boom, good)
    assert svc.translate_texts(["こんにちは"], "ja") == ["译"]
    assert len(echo.calls) == 1 and len(boom.calls) == 1 and len(good.calls) == 1
    assert svc._breakers["boom"].is_open()                                     # 意外异常 → 熔断，不崩
    svc.close()
    # 所有引擎都「有回应但没翻」：不抛异常、保留为空、绝不用原文
    svc = svc_with(FakeEngine("echo", fn=lambda ts: list(ts)))
    page = [blk(0, "こんにちは")]
    svc.translate_blocks([page], "ja")
    assert page[0].translation == ""
    svc.close()


def test_breaker_cooldown_and_half_open() -> None:
    bad = FakeEngine("bad", fn=lambda ts: (_ for _ in ()).throw(EngineError("连不上", retryable=True)))
    good = FakeEngine("good")
    svc = svc_with(bad, good, max_retries=0, breaker_threshold=2, breaker_cooldown=0.3)
    for i in range(3):
        assert svc.translate_texts([f"文{i}"], "ja") == [f"译文{i}"]
    assert len(bad.calls) == 2                                                 # 第 3 次熔断中，直接跳过
    time.sleep(0.35)
    assert svc.translate_texts(["文3"], "ja") == ["译文3"]
    assert len(bad.calls) == 3 and svc._breakers["bad"].is_open()              # 半开：再失败一次立刻熔断
    svc.close()
    br = _Breaker(3, 60)
    assert br.failure("404", fatal=True) and br.is_open()                      # 致命错误立即熔断
    br.success()
    assert not br.is_open()


def test_retry_backoff_on_429() -> None:
    state = {"n": 0}

    def fn(ts: list[str]) -> list[str]:
        state["n"] += 1
        if state["n"] <= 2:
            raise EngineError("请求过于频繁，被限流（429）", retryable=True, status=429, retry_after=0.01)
        return [fake_tr(t) for t in ts]

    eng = FakeEngine(fn=fn)
    svc = svc_with(eng, max_retries=3, backoff_base=0.01)
    t0 = time.perf_counter()
    assert svc.translate_texts(["テスト文"], "ja") == ["译文"]
    assert len(eng.calls) == 3 and time.perf_counter() - t0 >= 0.02
    svc.close()
    lim = _Limiter(0.05)
    ev = threading.Event()
    t0 = time.perf_counter()
    for _ in range(4):
        lim.wait(ev)
    assert time.perf_counter() - t0 >= 0.14


def test_all_engines_down_raises_but_keeps_partial() -> None:
    def fn(ts: list[str]) -> list[str]:
        if any("×" in t for t in ts):
            raise EngineError("连不上翻译服务器（断网或被屏蔽）", retryable=True)
        return [fake_tr(t) for t in ts]

    a = FakeEngine("google", fn=fn, max_items=1)
    b = FakeEngine("edge", fn=lambda ts: (_ for _ in ()).throw(EngineError("翻译接口已失效（404）", fatal=True)))
    svc = svc_with(a, b)
    pages = [[blk(0, "第一ページです")], [blk(0, "×だめ"), blk(1, "×これも")]]
    with pytest.raises(TranslationUnavailable) as ei:
        svc.translate_blocks(pages, "ja")
    msg = str(ei.value)
    assert "暂停" in msg and "2 句" in msg and "Google" in msg and "404" in msg
    assert pages[0][0].translation == "译第一"                                 # 已翻好的照样写回
    assert pages[1][0].translation == "" and pages[1][1].translation == ""     # 绝不把原文当译文
    svc.close()
    with pytest.raises(TranslationUnavailable):
        svc_with(FakeEngine(fn=fn)).translate_texts(["×だめ"], "ja")    # 单独的「×」是纯符号，直接搬运不请求


def test_cancel_before_and_during() -> None:
    ev = threading.Event()
    ev.set()
    svc = svc_with(FakeEngine())
    with pytest.raises(Cancelled):
        svc.translate_blocks([[blk(0, "テスト")]], "ja", cancel=ev)
    svc.close()
    ev = threading.Event()

    def fn(ts: list[str]) -> list[str]:
        ev.set()                                                               # 第一批完成时用户点了暂停
        return [fake_tr(t) for t in ts]

    eng = FakeEngine(fn=fn, max_items=1)
    svc = svc_with(eng)
    pages = [[blk(0, "一つ目"), blk(1, "二つ目"), blk(2, "三つ目")]]
    with pytest.raises(Cancelled):
        svc.translate_blocks(pages, "ja", cancel=ev)
    assert pages[0][0].translation == "译一目" and pages[0][1].translation == ""
    assert len(eng.calls) == 1
    svc.close()


def test_glossary_placeholder_and_lost_placeholder() -> None:
    gl = [{"src": "ナルト", "dst": "鸣人"}, {"src": "Konoha", "dst": "木叶"}]
    keep = FakeEngine()
    svc = svc_with(keep, glossary=gl)
    out = svc.translate_texts(["ナルトが来た！", "ナルト", "welcome to konoha"], "ja")
    assert out == ["译鸣人来！", "鸣人", "译welcome to 木叶"]
    assert keep.calls == [["{0}が来た！", "welcome to {0}"]]                  # 整句是术语时不请求
    svc.close()
    drop = FakeEngine(fn=lambda ts: [fake_tr(re.sub(r"\{\d\}", "", t)) for t in ts])   # 引擎吞掉占位符
    svc = svc_with(drop, glossary=gl)
    assert svc.translate_texts(["ナルトが来た！"], "ja") == ["译鸣人来！"]
    assert drop.calls == [["{0}が来た！"], ["鸣人が来た！"]]                    # 退路：译名直接写进原文重翻
    svc.close()


def test_cache_used_across_instances(tmp_path) -> None:
    path = str(tmp_path / "tr.sqlite")
    e1 = FakeEngine("google")
    s1 = svc_with(e1, cache=path)
    texts = ["おはよう", "こんばんは"]
    r1 = s1.translate_texts(texts, "ja")
    s1.close()
    e2 = FakeEngine("google", fn=lambda ts: ["不该被调用" for _ in ts])
    s2 = svc_with(e2, cache=path)
    assert s2.translate_texts(texts, "ja") == r1 and e2.calls == []
    assert s2.cache_stats() == {"google": 2}
    assert s2.translate_texts(texts, "ja", target="zh-Hant") == ["不该被调用"] * 2   # 繁体另算
    s2.close()


def test_concurrency_limit_and_threads() -> None:
    eng = FakeEngine(max_items=1, delay=0.05)
    svc = svc_with(eng, concurrency=2)
    out = svc.translate_texts([f"文{i}" for i in range(8)], "ja")
    assert out == [f"译文{i}" for i in range(8)]
    assert eng.max_active == 2
    # 多个线程同时调用同一个服务：全局并发仍 <= 2
    eng.max_active = 0
    results: dict[int, list[str]] = {}

    def run(k: int) -> None:
        results[k] = svc.translate_texts([f"線{k}-{i}" for i in range(3)], "ja")

    ts = [threading.Thread(target=run, args=(k,)) for k in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert eng.max_active <= 2 and all(len(v) == 3 and all(v) for v in results.values())
    svc.close()


def test_translate_texts_alignment_and_garbage() -> None:
    svc = svc_with(FakeEngine())
    out = svc.translate_texts(["", "こんにちは", None, "  ", "…", "！？"], "ja")  # type: ignore[list-item]
    assert out == ["", "译", "", "", "……", "！？"]
    assert svc.translate_texts([], "ja") == []
    svc.translate_blocks([], "ja")
    svc.translate_blocks([None, [], [blk(0, "")]], "ja")  # type: ignore[list-item]
    ok, msg = svc.test_engine("nope")
    assert not ok and "未知" in msg
    ok, msg = svc.test_engine("fake")
    assert ok and "可用" in msg
    svc.close()
    svc.close()


def test_bad_settings_do_not_crash() -> None:
    svc = TranslationService({"translate": {"engines": "google", "concurrency": "x", "glossary": "坏",
                                            "timeout": None, "llm": None}, "typeset": None}, "")
    assert svc.engine_order == ["google"]
    svc.close()
    svc = TranslationService(None, "")  # type: ignore[arg-type]
    assert svc.engine_order == ["google"]
    svc.close()


# ---- 联网实测 -----------------------------------------------------------------------------------
@pytest.mark.network
@pytest.mark.parametrize("eid", ["google", "edge", "tencent"])
def test_network_free_engine(eid: str, out_dir) -> None:
    s = copy.deepcopy(DEFAULTS)
    s["translate"]["engines"] = [eid]
    svc = TranslationService(s, "")                                           # 不用缓存，测真实耗时
    ok, msg = svc.test_engine(eid)
    report: dict[str, Any] = {"engine": eid, "date": time.strftime("%Y-%m-%d %H:%M"), "ok": ok, "test": msg}
    try:
        if not ok:
            if eid == "google":
                pytest.fail(f"google 应当可用：{msg}")
            pytest.skip(f"{eid} 不可用：{msg}")
        t0 = time.perf_counter()
        p012 = svc.translate_texts(P012, "ja")
        report["p012_ms"] = round((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        common = svc.translate_texts(COMMON, "ja")
        report["common_ms"] = round((time.perf_counter() - t0) * 1000)
        report["p012"] = dict(zip(P012, p012))
        report["common"] = dict(zip(COMMON, common))
        assert all(p012) and all(common)
        assert re.match(r"^(.)、\1", p012[1]), p012[1]                         # マ…マネージャー → 经、经理
        for src, zh in zip(P012 + COMMON, p012 + common):
            assert not looks_untranslated(src, zh, "ja") and src != zh
        page = [blk(0, P012[0]), blk(1, "ドキドキ", is_sfx=True), blk(2, "…"), blk(3, COMMON[0], translation="人工")]
        t0 = time.perf_counter()
        svc.translate_blocks([page, [blk(0, P012[2])]], "ja")
        report["blocks_ms"] = round((time.perf_counter() - t0) * 1000)
        assert page[0].translation and page[1].translation == "怦怦" and page[2].translation == "……"
        assert page[3].translation == "人工"
    finally:
        (out_dir / f"network_{eid}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        svc.close()


def test_local_llm_live_if_running(out_dir) -> None:
    """本机开着 Ollama / LM Studio 时顺便实测；没开就跳过（不算失败）。"""
    s = copy.deepcopy(DEFAULTS)
    s["translate"]["llm"]["enabled"] = True
    s["translate"]["engines"] = ["llm"]
    svc = TranslationService(s, "")
    ok, msg = svc.test_engine("llm")
    if not ok:
        svc.close()
        pytest.skip(f"本地大模型未运行：{msg}")
    t0 = time.perf_counter()
    out = svc.translate_texts(P012 + COMMON, "ja")
    report = {"engine": "llm", "model": s["translate"]["llm"]["model"], "ms": round((time.perf_counter() - t0) * 1000),
              "result": dict(zip(P012 + COMMON, out))}
    (out_dir / "network_llm.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    svc.close()
    assert sum(bool(x) for x in out) >= len(out) - 1

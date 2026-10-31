"""翻译引擎：免费在线引擎 + 本地大模型（OpenAI 兼容）。统一接口 translate_batch(texts, src, tgt)。

2026-10-30 在本机（马来西亚网络）实测：
- google：POST translate.googleapis.com/translate_a/t（client=gtx，表单里多个 q=）→ 可用，约 100 ms / 批，
  120 句 / 3250 字一次请求 1.9 s 全部对齐。sl=auto 时返回 [[译文, 语种], ...]，否则 [译文, ...]。
  该接口把输入当 HTML 解析（&amp; 会被解码），所以发送前 html.escape、收到后 html.unescape。
- edge：GET edge.microsoft.com/translate/auth 返回 404 → 授权接口已下线，类保留但不进默认列表。
- tencent：transmart.qq.com 服务器证书过期（TLS 校验失败）→ 类保留但不进默认列表；绝不关闭证书校验。
- llm：Ollama（默认 http://127.0.0.1:11434/v1）/ LM Studio 等 OpenAI 兼容接口，整页编号 JSON + 前文 + 术语表。

约定：
- 引擎只负责「一批文本 → 一批译文」，限速 / 重试 / 熔断 / 缓存 / 后处理都在 service 里做；
- 所有失败都转换成 EngineError（message 是给客户看的简短中文），条数对不上抛 SplitNeeded 让 service 二分；
- 不伪装浏览器、不抓反滥用令牌、不关闭 TLS 校验；429 时由 service 退避，绝不换接口规避限流。
"""
from __future__ import annotations

import base64
import binascii
import html
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import uuid
from typing import Any, Sequence

import requests

try:
    from hanhua.runtime import VERSION
except Exception:  # noqa: BLE001 —— 底座缺失时也能单独 import
    VERSION = "4.0"

log = logging.getLogger("hanhua.translate")

USER_AGENT = f"HanhuaManga/{VERSION}"
DEFAULT_LLM_URL = "http://127.0.0.1:11434/v1"
TARGETS = ("zh-Hans", "zh-Hant")
SOURCES = ("ja", "ko", "en", "zh", "auto")


# ---- 异常 ---------------------------------------------------------------------------------------
class EngineError(Exception):
    """单个引擎的一次请求失败（service 内部用；message 会拼进给客户看的提示，必须是简短中文）。

    retryable：退避后重试可能成功（429 / 5xx / 超时 / 断网）；
    fatal：接口失效 / 证书错误 / 拒绝访问，立即熔断该引擎；
    retry_after：服务器给的 Retry-After 秒数。"""

    def __init__(self, message: str, *, retryable: bool = True, fatal: bool = False,
                 status: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable and not fatal
        self.fatal = fatal
        self.status = status
        self.retry_after = retry_after


class SplitNeeded(EngineError):
    """这一批的返回条数对不上 / 请求体太大 / 大模型 JSON 缺条目：service 会二分后重试。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False, fatal=False)


def count_mismatch(expected: int, got: int) -> SplitNeeded:
    return SplitNeeded(f"返回条数不对（发 {expected} 条，回 {got} 条）")


# ---- 小工具 -------------------------------------------------------------------------------------
def section(settings: Any, name: str) -> dict:
    """安全地取 settings[name]（缺失 / 类型不对时返回 {}）。"""
    if isinstance(settings, dict):
        sec = settings.get(name)
        if isinstance(sec, dict):
            return sec
    return {}


def num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(lo, min(hi, v))


def norm_target(target: Any) -> str:
    t = str(target or "").strip().lower()
    if any(k in t for k in ("hant", "tw", "hk", "繁")):
        return "zh-Hant"
    return "zh-Hans"


def norm_source(src: Any) -> str:
    s = str(src or "").strip().lower()
    if s.startswith("zh"):
        return "zh"
    return s if s in SOURCES else "auto"


def _retry_after(r: requests.Response) -> float | None:
    v = r.headers.get("Retry-After")
    if not v:
        return None
    try:
        return max(0.0, min(120.0, float(v)))
    except ValueError:
        return None


def _json(r: requests.Response) -> Any:
    try:
        return r.json()
    except ValueError as e:
        raise EngineError("返回内容不是 JSON（可能被网络拦截或接口有变）", retryable=True) from e


class _Sessions:
    """每个线程一个 requests.Session（Session 不保证线程安全）。"""

    def __init__(self, headers: dict[str, str]) -> None:
        self._headers = dict(headers)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all: list[requests.Session] = []
        self._gen = 0

    def get(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None or getattr(self._local, "gen", -1) != self._gen:
            s = requests.Session()
            s.headers.update(self._headers)
            self._local.s, self._local.gen = s, self._gen
            with self._lock:
                self._all.append(s)
        return s

    def close(self) -> None:
        with self._lock:
            sessions, self._all = self._all, []
            self._gen += 1
        for s in sessions:
            try:
                s.close()
            except Exception as e:  # noqa: BLE001
                log.debug("关闭 HTTP 会话出错：%s", e)


# ---- 基类 ---------------------------------------------------------------------------------------
class BaseEngine:
    id = "base"
    name = "基类"
    kind = "free"                 # free 免费在线 | local 本地
    batch = True                  # 支持一次请求多条
    max_items = 50                # 每批最多条数
    max_chars = 4000              # 每批最多字符数
    min_interval = 0.3            # 同一引擎两次请求的最小间隔（秒），所有线程共享
    placeholder_glossary = True   # True：术语换成 {0} 占位符；False：术语表写进提示词（本地大模型）

    def __init__(self, settings: dict | None = None) -> None:
        tr = section(settings, "translate")
        self.timeout = num(tr.get("timeout"), 15.0, 3.0, 120.0)
        self._sessions = _Sessions({"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"})

    @property
    def cache_id(self) -> str:
        """缓存 key 里的引擎名（本地大模型带上模型名，换模型不串缓存）。"""
        return self.id

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str, *,
                        context: Sequence[str] | None = None,
                        glossary: Sequence[tuple[str, str]] | None = None) -> list[str]:
        raise NotImplementedError

    def probe(self) -> None:
        """测试前的额外检查（本地大模型：服务是否启动、模型是否存在）。失败抛 EngineError。"""

    def close(self) -> None:
        self._sessions.close()

    # ---- HTTP ----------------------------------------------------------------------------------
    def _request(self, method: str, url: str, *, timeout: float | None = None,
                 offline_msg: str = "连不上翻译服务器（断网或被屏蔽）", **kw: Any) -> requests.Response:
        t = self.timeout if timeout is None else timeout
        try:
            r = self._sessions.get().request(method, url, timeout=(min(8.0, t), t), **kw)
        except requests.exceptions.SSLError as e:
            raise EngineError("安全连接失败（服务器证书无效或已过期）", fatal=True) from e
        except requests.exceptions.Timeout as e:
            raise EngineError("请求超时", retryable=True) from e
        except requests.exceptions.ConnectionError as e:
            raise EngineError(offline_msg, retryable=True) from e
        except requests.exceptions.RequestException as e:
            raise EngineError(f"请求失败（{type(e).__name__}）", retryable=True) from e
        code = r.status_code
        if code == 429:
            raise EngineError("请求过于频繁，被限流（429）", retryable=True, status=code, retry_after=_retry_after(r))
        if code >= 500:
            raise EngineError(f"翻译服务器出错（{code}）", retryable=True, status=code, retry_after=_retry_after(r))
        if code in (413, 414):
            raise SplitNeeded(f"一次发送的内容太多（{code}）")
        if code in (401, 403):
            raise EngineError(f"翻译服务拒绝访问（{code}）", fatal=True, status=code)
        if code in (404, 410):
            raise EngineError(f"翻译接口已失效（{code}）", fatal=True, status=code)
        if code >= 400:
            raise EngineError(f"请求被拒绝（{code}）", retryable=False, status=code)
        return r


# ---- Google（免费网页接口）-------------------------------------------------------------------------
class GoogleFree(BaseEngine):
    id = "google"
    name = "Google 翻译（免费）"
    max_items = 60
    max_chars = 4000
    min_interval = 0.35
    URL = "https://translate.googleapis.com/translate_a/t"
    _SRC = {"ja": "ja", "ko": "ko", "en": "en", "zh": "auto", "auto": "auto"}

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str, *,
                        context: Sequence[str] | None = None,
                        glossary: Sequence[tuple[str, str]] | None = None) -> list[str]:
        if not texts:
            return []
        sl = self._SRC.get(norm_source(src), "auto")
        tl = "zh-TW" if norm_target(tgt) == "zh-Hant" else "zh-CN"
        data = [("q", html.escape(str(t or ""), quote=False)) for t in texts]
        r = self._request("POST", self.URL, params={"client": "gtx", "sl": sl, "tl": tl}, data=data)
        out = self.parse(_json(r), len(texts))
        return [html.unescape(x) for x in out]

    @staticmethod
    def parse(payload: Any, n: int) -> list[str]:
        """解析 translate_a/t 的几种返回形状：[译文…] / [[译文, 语种]…] / "译文"。"""
        if isinstance(payload, str):
            items: list[Any] = [payload]
        elif isinstance(payload, list):
            items = payload
            # 单条 + auto 时个别情况会是扁平的 ["译文", "ja"]
            if (n == 1 and len(items) == 2 and all(isinstance(x, str) for x in items)
                    and re.fullmatch(r"[a-z]{2,3}(-[A-Za-z]{2,4})?", items[1])):
                items = [items[0]]
        else:
            raise EngineError("返回格式无法识别", retryable=True)
        out: list[str] = []
        for it in items:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, list) and it and isinstance(it[0], str):
                out.append(it[0])
            else:
                raise EngineError("返回格式无法识别", retryable=True)
        if len(out) != n:
            raise count_mismatch(n, len(out))
        return out


# ---- 微软 Edge（免费，浏览器内置翻译的授权接口）-----------------------------------------------------
def _jwt_ttl(token: str, default: float) -> float:
    """从 JWT 的 exp 推算剩余有效期（提前 60 s 刷新）；解析不了就用 default。"""
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        exp = float(json.loads(base64.urlsafe_b64decode(part)).get("exp", 0))
        if exp > 0:
            return max(30.0, min(default, exp - time.time() - 60))
    except (ValueError, IndexError, TypeError, AttributeError, binascii.Error):
        pass
    return default


class EdgeFree(BaseEngine):
    id = "edge"
    name = "微软 Edge 翻译（免费）"
    max_items = 100
    max_chars = 8000
    min_interval = 0.25
    AUTH_URL = "https://edge.microsoft.com/translate/auth"
    API_URL = "https://api-edge.cognitive.microsofttranslator.com/translate"
    TOKEN_TTL = 8 * 60.0
    _SRC = {"ja": "ja", "ko": "ko", "en": "en"}

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(settings)
        self._tok = ""
        self._tok_exp = 0.0
        self._tok_lock = threading.Lock()

    def _token(self, force: bool = False) -> str:
        with self._tok_lock:
            if not force and self._tok and time.monotonic() < self._tok_exp:
                return self._tok
            r = self._request("GET", self.AUTH_URL)
            tok = (r.text or "").strip()
            if tok.count(".") != 2 or len(tok) < 40:
                raise EngineError("拿不到微软翻译的授权令牌", retryable=True)
            self._tok = tok
            self._tok_exp = time.monotonic() + _jwt_ttl(tok, self.TOKEN_TTL)
            return tok

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str, *,
                        context: Sequence[str] | None = None,
                        glossary: Sequence[tuple[str, str]] | None = None) -> list[str]:
        if not texts:
            return []
        params = {"to": norm_target(tgt), "api-version": "3.0"}
        frm = self._SRC.get(norm_source(src))
        if frm:
            params["from"] = frm
        body = [{"Text": str(t or "")} for t in texts]
        r: requests.Response | None = None
        for attempt in (0, 1):
            tok = self._token(force=attempt == 1)
            try:
                r = self._request("POST", self.API_URL, params=params, json=body,
                                  headers={"Authorization": "Bearer " + tok})
                break
            except EngineError as e:
                if e.status in (401, 403) and attempt == 0:
                    continue          # 令牌过期：强制刷新再试一次
                raise
        assert r is not None
        data = _json(r)
        if not isinstance(data, list):
            raise EngineError("返回格式无法识别", retryable=True)
        out: list[str] = []
        for d in data:
            try:
                out.append(str(d["translations"][0]["text"]))
            except (KeyError, IndexError, TypeError) as e:
                raise EngineError("返回格式无法识别", retryable=True) from e
        if len(out) != len(texts):
            raise count_mismatch(len(texts), len(out))
        return out


# ---- 腾讯翻译君（TranSmart 网页接口）-----------------------------------------------------------------
class TencentFree(BaseEngine):
    id = "tencent"
    name = "腾讯翻译君（免费）"
    max_items = 50
    max_chars = 3000
    min_interval = 0.5
    URL = "https://transmart.qq.com/api/imt"
    _SRC = {"ja": "ja", "ko": "ko", "en": "en", "zh": "zh", "auto": "auto"}

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(settings)
        self._client_key = f"hanhua-{VERSION}-{uuid.uuid4()}-{int(time.time() * 1000)}"

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str, *,
                        context: Sequence[str] | None = None,
                        glossary: Sequence[tuple[str, str]] | None = None) -> list[str]:
        if not texts:
            return []
        body = {
            "header": {"fn": "auto_translation", "client_key": self._client_key},
            "type": "plain", "model_category": "normal", "text_domain": "general",
            "source": {"lang": self._SRC.get(norm_source(src), "auto"), "text_list": [str(t or "") for t in texts]},
            "target": {"lang": "zh-TW" if norm_target(tgt) == "zh-Hant" else "zh"},
        }
        r = self._request("POST", self.URL, json=body)
        data = _json(r)
        if not isinstance(data, dict):
            raise EngineError("返回格式无法识别", retryable=True)
        header = data.get("header") if isinstance(data.get("header"), dict) else {}
        code = header.get("ret_code")
        if code not in (None, "succ"):
            raise EngineError(f"腾讯翻译返回错误（{code}）", retryable=code in ("busy", "error", "timeout"))
        out = data.get("auto_translation")
        if not isinstance(out, list) or not all(isinstance(x, str) for x in out):
            raise EngineError("返回格式无法识别", retryable=True)
        if len(out) != len(texts):
            raise count_mismatch(len(texts), len(out))
        return list(out)


# ---- 本地大模型（OpenAI 兼容：Ollama / LM Studio / vLLM …）-------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_PAIR_RE = re.compile(r'"(\d{1,3})"\s*:\s*"((?:[^"\\]|\\.)*)"')
_LANG_NAME = {"ja": "日语", "ko": "韩语", "en": "英语", "zh": "中文", "auto": "外语"}
_TGT_NAME = {"zh-Hans": "简体中文", "zh-Hant": "繁体中文（台湾用字）"}


def norm_base_url(url: Any) -> str:
    """http://127.0.0.1:11434 → http://127.0.0.1:11434/v1；去掉误填的 /chat/completions。"""
    u = str(url or "").strip().rstrip("/") or DEFAULT_LLM_URL
    if not re.match(r"^https?://", u, re.I):
        u = "http://" + u
    if u.lower().endswith("/chat/completions"):
        u = u[: -len("/chat/completions")].rstrip("/")
    if urllib.parse.urlsplit(u).path in ("", "/"):
        u += "/v1"
    return u


def _as_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("text", "translation", "zh", "dst"):
            if isinstance(v.get(k), str):
                return v[k]
    if isinstance(v, list):
        return "".join(_as_text(x) for x in v)
    return "" if v is None else str(v)


def parse_numbered(content: str, n: int) -> list[str]:
    """从大模型回复里取出编号 JSON：{"1": "…", "2": "…"} 或长度为 n 的数组。缺条目抛 SplitNeeded。"""
    s = _THINK_RE.sub("", content or "")
    low = s.lower()
    if "</think>" in low:                       # 只剩半截 think 标签
        s = s[low.rfind("</think>") + len("</think>"):]
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1)
    obj: Any = None
    for left, right in (("{", "}"), ("[", "]")):
        i, j = s.find(left), s.rfind(right)
        if 0 <= i < j:
            try:
                obj = json.loads(s[i:j + 1])
                break
            except ValueError:
                obj = None
    if obj is None:
        pairs = _PAIR_RE.findall(s)
        if pairs:
            obj = {}
            for k, v in pairs:
                try:
                    obj[k] = json.loads(f'"{v}"')
                except ValueError:
                    obj[k] = v
    if isinstance(obj, dict):
        for key in ("translations", "translation", "result", "results", "data", "output"):
            if isinstance(obj.get(key), (list, dict)):
                obj = obj[key]
                break
    if isinstance(obj, list):
        if len(obj) != n:
            raise count_mismatch(n, len(obj))
        return [_as_text(x).strip() for x in obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for i in range(1, n + 1):
            v = obj.get(str(i), obj.get(i))
            if v is None:
                raise SplitNeeded(f"大模型漏了第 {i} 条")
            out.append(_as_text(v).strip())
        return out
    plain = s.strip().strip('"“”')
    if n == 1 and plain and "\n" not in plain and len(plain) <= 300:
        return [plain]                           # 单条时允许直接回纯文本
    raise SplitNeeded("大模型没有按 JSON 格式回复")


class LocalLLM(BaseEngine):
    id = "llm"
    name = "本地大模型（Ollama / LM Studio）"
    kind = "local"
    max_items = 30
    max_chars = 1800
    min_interval = 0.0
    placeholder_glossary = False

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(settings)
        llm = section(section(settings, "translate"), "llm")
        self.base_url = norm_base_url(llm.get("base_url"))
        self.model = str(llm.get("model") or "qwen3:8b").strip() or "qwen3:8b"
        self.api_key = str(llm.get("api_key") or "").strip()
        self.temperature = num(llm.get("temperature"), 0.3, 0.0, 2.0)
        self.llm_timeout = num(llm.get("timeout"), max(90.0, self.timeout * 6), 10.0, 900.0)

    @property
    def cache_id(self) -> str:
        return f"llm:{self.model}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _offline(self) -> str:
        return f"连不上本地大模型（{self.base_url}），请确认 Ollama / LM Studio 已启动"

    def probe(self) -> None:
        r = self._request("GET", self.base_url + "/models", headers=self._headers(), timeout=min(10.0, self.timeout),
                          offline_msg=self._offline())
        data = _json(r)
        ids = [str(m.get("id")) for m in (data.get("data") or []) if isinstance(m, dict)] if isinstance(data, dict) else []
        if ids and self.model not in ids:
            shown = "、".join(ids[:6])
            raise EngineError(f"本地大模型里没有「{self.model}」（已安装：{shown}）", fatal=True)

    def messages(self, texts: Sequence[str], src: str, tgt: str, context: Sequence[str] | None,
                 glossary: Sequence[tuple[str, str]] | None) -> list[dict[str, str]]:
        src_name = _LANG_NAME.get(norm_source(src), "外语")
        tgt_name = _TGT_NAME[norm_target(tgt)]
        system = (
            f"你是资深漫画汉化译者，把{src_name}漫画台词翻译成{tgt_name}。要求：\n"
            "1. 口语化、贴合人物语气，简洁（要嵌进对话气泡）；\n"
            "2. 每条单独翻译，不合并、不拆分、不省略，编号一一对应；\n"
            "3. {0}、{1} 这类占位符原样保留；拟声词译成中文拟声词；\n"
            "4. 不要解释，不要注音，不要加引号以外的多余符号。\n"
            '只输出一个 JSON 对象，键是编号字符串，值是译文，例如 {"1": "译文", "2": "译文"}。')
        parts: list[str] = []
        if glossary:
            parts.append("术语表（必须照此翻译）：\n" + "\n".join(f"{s} → {d}" for s, d in glossary))
        ctx = [c for c in (context or []) if c][-8:]
        if ctx:
            parts.append("前文（只作参考，不要翻译）：\n" + "\n".join(ctx))
        numbered = {str(i + 1): str(t or "") for i, t in enumerate(texts)}
        parts.append("待翻译：\n" + json.dumps(numbered, ensure_ascii=False))
        user = "\n\n".join(parts)
        if "qwen3" in self.model.lower():
            user += "\n/no_think"            # Qwen3 的软开关：关掉思考过程，快很多
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _chat(self, messages: list[dict[str, str]]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "stream": False}
        r = self._request("POST", self.base_url + "/chat/completions", json=body, headers=self._headers(),
                          timeout=self.llm_timeout, offline_msg=self._offline())
        data = _json(r)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise EngineError("本地大模型返回格式无法识别", retryable=True) from e
        return content if isinstance(content, str) else _as_text(content)

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str, *,
                        context: Sequence[str] | None = None,
                        glossary: Sequence[tuple[str, str]] | None = None) -> list[str]:
        if not texts:
            return []
        msgs = self.messages(texts, src, tgt, context, glossary)
        content = self._chat(msgs)
        try:
            return parse_numbered(content, len(texts))
        except SplitNeeded as first:
            log.debug("本地大模型格式不对，提醒后重试一次：%s", first.message)
        retry = msgs + [{"role": "assistant", "content": (content or "")[:2000]},
                        {"role": "user", "content": f"格式不对。请只输出 JSON 对象，键为 1 到 {len(texts)}，值为译文。"}]
        return parse_numbered(self._chat(retry), len(texts))


# ---- 内置本地大模型（llama.cpp，默认首选）/ API 服务（填 Key，第二选择）---------------------------------
def _llm_error(e: Exception) -> EngineError:
    """hanhua.llm 的中文错误 → EngineError：缺模型 / 缺引擎 / 显存不足 / Key 无效 = 致命（立即换下一个引擎）。"""
    msg = (str(e).splitlines() or ["本地模型出错"])[0][:160]
    fatal = any(k in msg for k in ("还没有下载", "缺少", "启动失败", "显存不足", "Key 无效", "还没有填写", "启动超时"))
    return EngineError(msg, fatal=fatal, retryable=not fatal)


class BundledLLM(BaseEngine):
    """软件自带的 llama.cpp + 按本机配置推荐的模型：离线、不限次数、整页上下文 + 术语表。"""

    id = "local"
    name = "本地大模型（内置，按配置推荐）"
    kind = "local"
    max_items = 40
    max_chars = 2400
    min_interval = 0.0
    placeholder_glossary = False

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(settings)
        from ..llm.engine import LocalLLMEngine

        self._impl = LocalLLMEngine(settings or {})

    @property
    def cache_id(self) -> str:
        try:
            path, _, _ = self._impl.resolve()
            return "local:" + os.path.basename(path)          # 换模型不串缓存
        except Exception:  # noqa: BLE001
            return "local"

    def probe(self) -> None:
        try:
            self._impl.resolve()
        except Exception as e:  # noqa: BLE001
            raise _llm_error(e) from e

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str, *,
                        context: Sequence[str] | None = None,
                        glossary: Sequence[tuple[str, str]] | None = None) -> list[str]:
        if not texts:
            return []
        try:
            return self._impl.translate_batch(list(texts), src, tgt, list(context or []),
                                              [{"src": s, "dst": d} for s, d in (glossary or [])])
        except Exception as e:  # noqa: BLE001
            raise _llm_error(e) from e


class ApiService(BaseEngine):
    """填 Key 的大模型 API（DeepSeek / 硅基流动 / OpenAI / Gemini / OpenRouter / 自定义 OpenAI 兼容）。"""

    id = "api"
    name = "API 服务（填 Key）"
    kind = "api"
    max_items = 40
    max_chars = 3000
    min_interval = 0.2
    placeholder_glossary = False

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(settings)
        from ..llm.engine import ApiLLMEngine

        self._impl = ApiLLMEngine(settings or {})

    @property
    def cache_id(self) -> str:
        return f"api:{self._impl.model}"

    def probe(self) -> None:
        if not self._impl.available():
            raise EngineError("还没有填写 API 服务的地址、模型和 Key", fatal=True)

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str, *,
                        context: Sequence[str] | None = None,
                        glossary: Sequence[tuple[str, str]] | None = None) -> list[str]:
        if not texts:
            return []
        try:
            return self._impl.translate_batch(list(texts), src, tgt, list(context or []),
                                              [{"src": s, "dst": d} for s, d in (glossary or [])])
        except Exception as e:  # noqa: BLE001
            raise _llm_error(e) from e


# ---- 注册表 -------------------------------------------------------------------------------------
ENGINE_CLASSES: dict[str, type[BaseEngine]] = {
    "local": BundledLLM,
    "api": ApiService,
    "google": GoogleFree,
    "edge": EdgeFree,
    "tencent": TencentFree,
    "llm": LocalLLM,
}

# 给界面显示的静态信息；recommended = 建议放进默认引擎列表（2026-10-30 实测）
ENGINE_INFO: list[dict[str, Any]] = [
    {"id": "local", "name": BundledLLM.name, "kind": "local", "batch": True, "recommended": True,
     "note": "默认首选：软件自带，按电脑配置自动推荐模型；离线、不限次数、整页上下文 + 术语表"},
    {"id": "api", "name": ApiService.name, "kind": "api", "batch": True, "recommended": False,
     "note": "第二选择：填 DeepSeek / 硅基流动 / OpenAI / Gemini 等的 Key"},
    {"id": "google", "name": GoogleFree.name, "kind": "free", "batch": True, "recommended": True,
     "note": "2026-10 实测可用；整页一次请求，速度快"},
    {"id": "edge", "name": EdgeFree.name, "kind": "free", "batch": True, "recommended": False,
     "note": "2026-10 实测授权接口已下线（404），暂不可用"},
    {"id": "tencent", "name": TencentFree.name, "kind": "free", "batch": True, "recommended": False,
     "note": "2026-10 实测服务器证书过期，暂不可用"},
    {"id": "llm", "name": LocalLLM.name, "kind": "local", "batch": True, "recommended": False,
     "note": "免费、离线、整页上下文 + 术语表；需自行安装 Ollama 或 LM Studio 并下载模型"},
]


def create_engine(engine_id: str, settings: dict | None) -> BaseEngine | None:
    cls = ENGINE_CLASSES.get(str(engine_id or "").strip().lower())
    if cls is None:
        return None
    try:
        return cls(settings)
    except Exception as e:  # noqa: BLE001 —— 引擎构造失败只影响它自己
        log.warning("创建翻译引擎 %s 失败：%s", engine_id, e)
        return None

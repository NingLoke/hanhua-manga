"""TranslationService：整页批量翻译、多引擎自动切换、限速 / 熔断 / 退避、缓存、文本前后处理。

每个文字块的流程：
  跳过 skip / 空原文 / 已有译文（人工修改）→ 纯标点直接搬运 → 规范化原文 → 摘下首尾装饰符号 →
  拟声词先查内置词典 → 去口吃 → 术语换 {0} 占位符 → [缓存 → 引擎批量请求] → 还原术语 →
  补回口吃 / 装饰符号 → 中文后处理（标点风格取 settings["typeset"]["punct_style"]）。

引擎按 settings["translate"]["engines"] 的顺序尝试，一条在前一个引擎失败就交给下一个。
所有引擎都连不上时抛 TranslationUnavailable（已经翻好的块照样写回），绝不把原文当译文。
"""
from __future__ import annotations

import copy
import logging
import random
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

from hanhua.types import Cancelled, TextBlock, TranslationUnavailable

from . import sfx_dict
from .cache import TranslationCache
from .engines import (ENGINE_CLASSES, ENGINE_INFO, BaseEngine, EngineError, SplitNeeded, count_mismatch,
                      create_engine, norm_source, norm_target, num, section)
from .textproc import (Glossary, apply_stutter, attach_affixes, is_punct_only, looks_untranslated,
                       normalize_source, postprocess_zh, punct_passthrough, split_affixes, split_stutter)

log = logging.getLogger("hanhua.translate")

DEFAULT_ENGINES = ("google",)
_TEST_TEXT = "今日はいい天気ですね。"
_CONTEXT_LINES = 6                     # 本地大模型每批带的前文条数
_PLACEHOLDER = re.compile(r"[{｛]\s*[0-9０-９]{1,2}\s*[}｝]")


def _has_letters(s: str) -> bool:
    return any(unicodedata.category(c)[0] == "L" for c in s)


# ---- 限速 / 熔断 --------------------------------------------------------------------------------
class _Limiter:
    """同一引擎两次请求之间的最小间隔（所有线程共享一个时间表）。"""

    def __init__(self, interval: float) -> None:
        self.interval = max(0.0, float(interval))
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self, cancel: threading.Event) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            at = max(now, self._next)
            self._next = at + self.interval
        delay = at - now
        if delay > 0 and cancel.wait(delay):
            raise Cancelled("已取消翻译")


class _Breaker:
    """熔断器：连续失败 threshold 次（或一次致命错误）→ 冷却 cooldown 秒内不再请求该引擎。

    冷却结束后处于「半开」状态：再失败一次立刻重新熔断，成功一次完全恢复。"""

    def __init__(self, threshold: int = 3, cooldown: float = 60.0) -> None:
        self.threshold = max(1, int(threshold))
        self.cooldown = max(0.0, float(cooldown))
        self.last_error = ""
        self._fails = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._open_until

    def success(self) -> None:
        with self._lock:
            self._fails = 0
            self._open_until = 0.0

    def failure(self, message: str, fatal: bool = False) -> bool:
        """记一次失败；返回 True 表示这次触发了熔断。"""
        with self._lock:
            self.last_error = message
            self._fails += 1
            if fatal or self._fails >= self.threshold:
                self._open_until = time.monotonic() + self.cooldown
                self._fails = self.threshold - 1
                return True
            return False


# ---- 待翻译条目 ---------------------------------------------------------------------------------
@dataclass(eq=False)
class _Item:
    raw: str                         # OCR 原文
    page: int = 0                    # 页序号（打包批次用）
    norm: str = ""                   # 规范化后的原文（后处理判断句号用）
    lead: str = ""                   # 首部装饰符号（♡ ♪ …）
    trail: str = ""                  # 尾部装饰符号
    stutter: int = 0                 # 口吃次数
    core: str = ""                   # 去掉口吃 / 装饰后的正文
    protected: str = ""              # 术语换成占位符后的正文（在线引擎发送 / 缓存用）
    dsts: list[str] = field(default_factory=list)   # 占位符序号 → 译名
    result: str | None = None        # 最终译文
    answered: bool = False           # 有引擎回应过（即使结果不合格）→ 不算「引擎全挂」


def _pack(queries: Sequence[str], pages: Sequence[int], max_items: int, max_chars: int) -> list[list[str]]:
    """按页聚合成批：整页尽量放进同一批；单页超限时逐条装。"""
    max_items, max_chars = max(1, int(max_items)), max(1, int(max_chars))
    groups: list[list[str]] = []
    last: object = object()
    for q, p in zip(queries, pages):
        if groups and p == last:
            groups[-1].append(q)
        else:
            groups.append([q])
            last = p
    chunks: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for g in groups:
        g_size = sum(len(q) for q in g)
        if cur and (len(cur) + len(g) > max_items or size + g_size > max_chars):
            chunks.append(cur)
            cur, size = [], 0
        if len(g) <= max_items and g_size <= max_chars:
            cur.extend(g)
            size += g_size
            continue
        for q in g:
            if cur and (len(cur) >= max_items or size + len(q) > max_chars):
                chunks.append(cur)
                cur, size = [], 0
            cur.append(q)
            size += len(q)
    if cur:
        chunks.append(cur)
    return chunks


# ---- 服务 ---------------------------------------------------------------------------------------
class TranslationService:
    """线程安全；构造一次、反复调用。"""

    def __init__(self, settings: dict, cache_path: str, *, engines: Sequence[BaseEngine] | None = None) -> None:
        """engines：直接注入引擎实例（测试 / 高级用途），给了就忽略 settings 里的引擎列表。"""
        self._settings: dict = copy.deepcopy(settings) if isinstance(settings, dict) else {}
        tr = section(self._settings, "translate")
        ts = section(self._settings, "typeset")
        self._target = norm_target(tr.get("target"))
        self._punct = ts.get("punct_style") if ts.get("punct_style") in ("cjk", "western") else "cjk"
        self._sfx_mode = str(tr.get("sfx_mode") or "translate")
        self._translate_free = bool(tr.get("translate_free_text", True))
        self._concurrency = int(num(tr.get("concurrency"), 3, 1, 8))
        self._max_retries = int(num(tr.get("max_retries"), 3, 0, 8))
        self._backoff_base = num(tr.get("backoff_base"), 1.0, 0.0, 30.0)
        self._backoff_max = num(tr.get("backoff_max"), 20.0, 0.0, 120.0)
        cooldown = num(tr.get("breaker_cooldown"), 60.0, 0.0, 3600.0)
        threshold = int(num(tr.get("breaker_threshold"), 3, 1, 20))
        try:
            self._glossary = Glossary(tr.get("glossary"))
        except Exception as e:  # noqa: BLE001 —— 术语表写坏了也不能影响翻译
            log.warning("术语表无法解析，已忽略：%s", e)
            self._glossary = Glossary(None)
        self._sem = threading.BoundedSemaphore(self._concurrency)
        self._engines: dict[str, BaseEngine] = {}
        if engines:
            for eng in engines:
                self._engines.setdefault(eng.id, eng)
        else:
            for eid in self._engine_order(tr):
                eng = create_engine(eid, self._settings)
                if eng is None:
                    continue
                if eid in ("local", "api"):          # 本地模型没下载 / API 没填 Key：直接跳过，用下一个
                    try:
                        eng.probe()
                    except EngineError as e:
                        log.info("翻译引擎 %s 暂不可用，跳过：%s", eid, e.message)
                        eng.close()
                        continue
                self._engines[eid] = eng
            if not self._engines and tr.get("online_fallback", True):
                for eid in DEFAULT_ENGINES:          # 本地模型没下载、也没填 Key：用免费在线引擎兜底
                    eng = create_engine(eid, self._settings)
                    if eng is not None:
                        self._engines[eid] = eng
        self._order = list(self._engines)
        self._breakers = {eid: _Breaker(threshold, cooldown) for eid in {*ENGINE_CLASSES, *self._engines}}
        self._limiters = {eid: _Limiter(getattr(eng, "min_interval", 0.0)) for eid, eng in self._engines.items()}
        self._cache = TranslationCache(cache_path or "")
        self._closed = False
        self._close_lock = threading.Lock()
        log.info("翻译服务就绪：引擎顺序 %s，目标 %s，并发 %d", self._order, self._target, self._concurrency)

    # ---- 公共接口 ------------------------------------------------------------------------------
    @staticmethod
    def engines() -> list[dict]:
        """[{"id","name","kind":"free"|"local","batch":bool,"recommended":bool,"note":str}]"""
        return [dict(info) for info in ENGINE_INFO]

    @property
    def engine_order(self) -> list[str]:
        return list(self._order)

    def translate_blocks(self, pages: list[list[TextBlock]], src_lang: str,
                         cancel: threading.Event | None = None) -> None:
        """为所有 (not skip) 且 src_text 非空且 translation=="" 的块填写译文（不覆盖人工修改）。

        所有引擎都失败时抛 TranslationUnavailable；cancel 被置位时抛 Cancelled。两种情况下已翻好的块都会写回。"""
        cancel = cancel if cancel is not None else threading.Event()
        src = norm_source(src_lang)
        jobs: list[tuple[Any, _Item]] = []
        for pi, page in enumerate(pages or []):
            for blk in page or []:
                try:
                    if not self._wants(blk):
                        continue
                    it = self._prepare(blk.src_text, src, self._target, bool(getattr(blk, "is_sfx", False)), pi)
                except Exception as e:  # noqa: BLE001 —— 单个坏块不影响整页
                    log.warning("跳过无法处理的文字块（第 %d 页）：%s", pi + 1, e)
                    continue
                if it is not None:
                    jobs.append((blk, it))
        if not jobs:
            return
        t0 = time.perf_counter()
        try:
            self._run([it for _, it in jobs], src, self._target, cancel)
        finally:
            done = 0
            for blk, it in jobs:
                try:
                    if it.result and not blk.translation:   # 再查一次：翻译期间编辑器可能已人工填写
                        blk.translation = it.result
                        done += 1
                except Exception as e:  # noqa: BLE001
                    log.warning("写回译文失败：%s", e)
            log.info("翻译写回 %d / %d 块，用时 %.1f 秒", done, len(jobs), time.perf_counter() - t0)

    def translate_texts(self, texts: list[str], src_lang: str, target: str | None = None) -> list[str]:
        """翻译一组文本，返回等长列表；空文本 / 拿不到可用译文的条目返回 ""。全部引擎不可用时抛 TranslationUnavailable。"""
        seq = list(texts or [])
        src = norm_source(src_lang)
        tgt = norm_target(target) if target else self._target
        out = [""] * len(seq)
        items: list[_Item] = []
        idx: list[int] = []
        for i, t in enumerate(seq):
            if not isinstance(t, str) or not t.strip():
                continue
            try:
                it = self._prepare(t, src, tgt, False, 0)
            except Exception as e:  # noqa: BLE001
                log.warning("跳过无法处理的文本：%s", e)
                continue
            if it is not None:
                items.append(it)
                idx.append(i)
        if items:
            self._run(items, src, tgt, threading.Event())
        for i, it in zip(idx, items):
            out[i] = it.result or ""
        return out

    def test_engine(self, engine_id: str) -> tuple[bool, str]:
        """真实请求一次（不走缓存、不受熔断限制）；返回 (是否可用, 中文说明)。不会抛异常。"""
        eid = str(engine_id or "").strip().lower()
        eng = self._engines.get(eid)
        temp = eng is None
        if temp:
            if eid not in ENGINE_CLASSES:
                return False, f"未知的翻译引擎：{engine_id}"
            eng = create_engine(eid, self._settings)
            if eng is None:
                return False, "引擎初始化失败"
        try:
            eng.probe()
            t0 = time.perf_counter()
            res = eng.translate_batch([_TEST_TEXT], "ja", self._target)
            ms = (time.perf_counter() - t0) * 1000
            if not isinstance(res, list) or len(res) != 1 or not self._valid(_TEST_TEXT, res[0], "ja"):
                return False, f"返回结果不对：{str(res)[:60]}"
            br = self._breakers.get(eid)
            if br is not None:
                br.success()                       # 测试通过：立刻解除熔断
            return True, f"可用（{ms:.0f} 毫秒）：{_TEST_TEXT} → {postprocess_zh(res[0], _TEST_TEXT, self._punct)}"
        except EngineError as e:
            return False, e.message
        except Exception as e:  # noqa: BLE001
            log.exception("测试翻译引擎 %s 出错", eid)
            return False, f"测试出错（{type(e).__name__}）"
        finally:
            if temp:
                eng.close()

    def cache_stats(self) -> dict[str, int]:
        return self._cache.stats()

    def clear_cache(self, engine_id: str | None = None) -> int:
        return self._cache.delete(engine_id)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        for eng in self._engines.values():
            try:
                eng.close()
            except Exception as e:  # noqa: BLE001
                log.debug("关闭引擎 %s 出错：%s", eng.id, e)
        self._cache.close()

    def __enter__(self) -> "TranslationService":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- 准备 / 收尾 ---------------------------------------------------------------------------
    @staticmethod
    def _engine_order(tr: dict) -> list[str]:
        raw = tr.get("engines")
        ids = raw if isinstance(raw, (list, tuple)) else list(DEFAULT_ENGINES)
        llm_on = bool(section(tr, "llm").get("enabled"))
        # 优先级：内置本地大模型（按配置推荐，默认开）→ 填 Key 的 API → 免费在线引擎兜底
        order: list[str] = []
        if section(tr, "local").get("enabled", True):
            order.append("local")
        if section(tr, "api").get("enabled"):
            order.append("api")
        online = bool(tr.get("online_fallback", True)) or not order
        for e in ids if online else []:
            if not isinstance(e, str):
                continue
            e = e.strip().lower()
            if e not in ENGINE_CLASSES or e in order or e in ("local", "api") or (e == "llm" and not llm_on):
                continue
            order.append(e)
        if llm_on and "llm" not in order:
            order.insert(len([o for o in order if o in ("local", "api")]), "llm")   # 自装 Ollama：排在内置 / API 之后
        if not order:
            log.warning("设置里没有可用的翻译引擎，改用默认 %s", DEFAULT_ENGINES)
            order = list(DEFAULT_ENGINES)
        return order

    def _wants(self, blk: Any) -> bool:
        if getattr(blk, "skip", False) or getattr(blk, "translation", ""):
            return False
        raw = getattr(blk, "src_text", "")
        if not isinstance(raw, str) or not raw.strip():
            return False
        is_sfx = bool(getattr(blk, "is_sfx", False))
        if is_sfx and self._sfx_mode != "translate":
            return False                         # 拟声词保留 / 只擦除：不需要译文
        if not is_sfx and getattr(blk, "kind", "bubble") == "free" and not self._translate_free:
            return False
        return True

    def _prepare(self, raw: str, src: str, tgt: str, is_sfx: bool, page: int) -> _Item | None:
        s = raw.strip()
        if not s:
            return None
        it = _Item(raw=s, page=page)
        if is_punct_only(s):
            it.result = punct_passthrough(s, self._punct)
            return it if it.result else None
        norm = normalize_source(s, src)
        if not norm:
            return None
        it.norm = norm
        it.lead, body, it.trail = split_affixes(norm)
        if not body.strip() or is_punct_only(body):
            it.result = attach_affixes(it.lead, punct_passthrough(body, self._punct), it.trail)
            return it if it.result else None
        if is_sfx:
            hit = sfx_dict.lookup(body, tgt)
            if hit:
                it.result = attach_affixes(it.lead, hit, it.trail)
                return it
        it.stutter, core = split_stutter(body)
        it.core = core or body
        it.protected, it.dsts = self._glossary.protect(it.core) if self._glossary else (it.core, [])
        if it.dsts and not _has_letters(_PLACEHOLDER.sub("", it.protected)):
            it.result = self._finish(it, it.protected, placeholder=True)   # 整句就是术语，不用请求
        return it

    def _finish(self, it: _Item, out: str, placeholder: bool) -> str | None:
        text = out
        if placeholder and it.dsts:
            text, ok = Glossary.restore(text, it.dsts)
            if not ok:
                return None                      # 占位符被引擎吞了：调用方改用直接替换重翻
        if self._glossary:
            text = self._glossary.correct(it.core, text)
        text = apply_stutter(it.stutter, text)
        text = attach_affixes(it.lead, text, it.trail)
        text = postprocess_zh(text, src=it.norm, punct_style=self._punct)
        return text or None

    @staticmethod
    def _valid(query: str, out: Any, src: str) -> bool:
        return isinstance(out, str) and bool(out.strip()) and not looks_untranslated(query, out, src)

    # ---- 主流程 --------------------------------------------------------------------------------
    def _run(self, items: list[_Item], src: str, tgt: str, cancel: threading.Event) -> None:
        pending = [it for it in items if it.result is None]
        if not pending:
            return
        if cancel.is_set():
            raise Cancelled("已取消翻译")
        errors: dict[str, str] = {}
        for eid in self._order:
            if not pending:
                break
            self._run_engine(self._engines[eid], pending, src, tgt, cancel, errors)
            pending = [it for it in pending if it.result is None]
        if not pending:
            return
        dead = [it for it in pending if not it.answered]
        if dead:
            raise TranslationUnavailable(self._unavailable_message(errors, len(dead)))
        for it in pending:
            log.warning("这句没有拿到可用译文，保留为空待人工处理：%r", it.raw[:60])

    def _run_engine(self, eng: BaseEngine, items: list[_Item], src: str, tgt: str,
                    cancel: threading.Event, errors: dict[str, str]) -> None:
        placeholder = eng.placeholder_glossary
        groups: dict[str, list[_Item]] = {}
        for it in items:
            groups.setdefault(it.protected if placeholder else it.core, []).append(it)
        found, answered, cancelled = self._resolve(eng, groups, src, tgt, cancel, errors)
        lost: list[_Item] = []
        for q, its in groups.items():
            if q in answered:
                for it in its:
                    it.answered = True
            out = found.get(q)
            if out is None:
                continue
            for it in its:
                res = self._finish(it, out, placeholder)
                if res:
                    it.result = res
                elif placeholder and it.dsts:
                    lost.append(it)
        if cancelled:
            raise Cancelled("已取消翻译")
        if not lost:
            return
        # 占位符被吞：把译名直接写进原文再翻一次（textproc 说明里的退路）
        direct: dict[str, list[_Item]] = {}
        for it in lost:
            direct.setdefault(self._glossary.direct(it.core), []).append(it)
        found2, _, cancelled2 = self._resolve(eng, direct, src, tgt, cancel, errors)
        for q, its in direct.items():
            out = found2.get(q)
            if out is None:
                continue
            for it in its:
                res = self._finish(it, out, placeholder=False)
                if res:
                    it.result = res
        if cancelled2:
            raise Cancelled("已取消翻译")

    def _resolve(self, eng: BaseEngine, groups: dict[str, list[_Item]], src: str, tgt: str,
                 cancel: threading.Event, errors: dict[str, str]) -> tuple[dict[str, str], set[str], bool]:
        """查询 → 引擎原始输出。返回 (合格输出, 有回应的查询, 是否被取消)。不抛 Cancelled。"""
        queries = list(groups)
        pages = [its[0].page for its in groups.values()]
        found: dict[str, str] = {}
        answered: set[str] = set()
        for q, out in self._cache.get_many(eng.cache_id, src, tgt, queries).items():
            if self._valid(q, out, src):
                found[q] = out
                answered.add(q)
        todo = [i for i, q in enumerate(queries) if q not in found]
        if not todo:
            return found, answered, False
        if cancel.is_set():
            return found, answered, True
        br = self._breakers[eng.id]
        if br.is_open():
            errors.setdefault(eng.id, br.last_error or "暂时停用（冷却中）")
            return found, answered, False
        chunks = _pack([queries[i] for i in todo], [pages[i] for i in todo], eng.max_items, eng.max_chars)
        pos = {q: i for i, q in enumerate(queries)}

        def job(chunk: list[str]) -> list[str | None]:
            ctx: list[str] | None = None
            gl: list[tuple[str, str]] | None = None
            if not eng.placeholder_glossary:       # 本地大模型：带前文 + 本批用到的术语
                start = pos[chunk[0]]
                ctx = queries[max(0, start - _CONTEXT_LINES):start]
                gl = [(e.src, e.dst) for e in self._glossary.relevant(chunk)] or None
            return self._call_split(eng, chunk, src, tgt, cancel, errors, ctx, gl)

        results: list[tuple[list[str], list[str | None]]] = []
        cancelled = False
        if len(chunks) == 1 or self._concurrency == 1:
            for c in chunks:
                try:
                    results.append((c, job(c)))
                except Cancelled:
                    cancelled = True
                    break
        else:
            with ThreadPoolExecutor(max_workers=min(self._concurrency, len(chunks)),
                                    thread_name_prefix="hanhua-tr") as ex:
                futs = [(c, ex.submit(job, c)) for c in chunks]
                for c, f in futs:
                    try:
                        results.append((c, f.result()))
                    except Cancelled:
                        cancelled = True
                    except Exception as e:  # noqa: BLE001 —— 理论上 _call_split 已兜底
                        log.exception("翻译批次意外失败：%s", e)
        to_cache: list[tuple[str, str]] = []
        for chunk, outs in results:
            for q, out in zip(chunk, outs):
                if out is None:
                    continue
                answered.add(q)
                if self._valid(q, out, src):
                    found[q] = out
                    to_cache.append((q, out))
                else:
                    log.debug("%s 返回的译文不合格（像没翻）：%r → %r", eng.id, q[:40], (out or "")[:40])
        if to_cache:
            self._cache.put_many(eng.cache_id, src, tgt, to_cache)
        return found, answered, cancelled

    def _call_split(self, eng: BaseEngine, texts: list[str], src: str, tgt: str, cancel: threading.Event,
                    errors: dict[str, str], ctx: list[str] | None,
                    gl: list[tuple[str, str]] | None) -> list[str | None]:
        """请求一批；条数对不上就二分。返回与 texts 等长：None = 引擎不可用，"" = 有回应但结果不可用。"""
        try:
            return self._call(eng, texts, src, tgt, cancel, ctx, gl)
        except SplitNeeded as e:
            if len(texts) <= 1:
                log.info("%s：这一句返回异常（%s），交给下一个引擎", eng.id, e.message)
                return [""] * len(texts)
            mid = len(texts) // 2
            log.debug("%s：%s，拆成 %d + %d 条重试", eng.id, e.message, mid, len(texts) - mid)
            return (self._call_split(eng, texts[:mid], src, tgt, cancel, errors, ctx, gl)
                    + self._call_split(eng, texts[mid:], src, tgt, cancel, errors, ctx, gl))
        except EngineError as e:
            errors[eng.id] = e.message
            return [None] * len(texts)

    def _call(self, eng: BaseEngine, texts: list[str], src: str, tgt: str, cancel: threading.Event,
              ctx: list[str] | None, gl: list[tuple[str, str]] | None) -> list[str]:
        """一次批量请求：限速 + 全局并发上限 + 429/5xx/超时指数退避 + 熔断。"""
        br = self._breakers[eng.id]
        lim = self._limiters.get(eng.id)
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            if cancel.is_set():
                raise Cancelled("已取消翻译")
            if br.is_open():
                raise EngineError(br.last_error or "暂时停用（冷却中）", retryable=False)
            if lim is not None:
                lim.wait(cancel)
            try:
                with self._sem:
                    if cancel.is_set():
                        raise Cancelled("已取消翻译")
                    t0 = time.perf_counter()
                    out = eng.translate_batch(list(texts), src, tgt, context=ctx, glossary=gl)
                    dt = time.perf_counter() - t0
            except Cancelled:
                raise
            except SplitNeeded:
                br.success()                     # 引擎能连上，只是这批要拆
                raise
            except EngineError as e:
                if br.failure(e.message, fatal=e.fatal):
                    log.warning("翻译引擎 %s 暂停使用 %.0f 秒：%s", eng.id, br.cooldown, e.message)
                if not e.retryable or attempt >= attempts - 1:
                    raise
                delay = self._backoff(attempt, e.retry_after)
                log.info("%s 请求失败（%s），%.1f 秒后第 %d 次重试", eng.id, e.message, delay, attempt + 1)
                if cancel.wait(delay):
                    raise Cancelled("已取消翻译")
                continue
            except Exception as e:  # noqa: BLE001 —— 引擎内部 bug：熔断该引擎，不让流程崩
                log.exception("翻译引擎 %s 出现意外错误", eng.id)
                err = EngineError(f"引擎内部错误（{type(e).__name__}）", fatal=True)
                br.failure(err.message, fatal=True)
                raise err from e
            br.success()
            if not isinstance(out, list) or len(out) != len(texts):
                raise count_mismatch(len(texts), len(out) if isinstance(out, list) else 0)
            log.debug("%s：%d 条，%.0f 毫秒", eng.id, len(texts), dt * 1000)
            return ["" if o is None else str(o) for o in out]
        raise EngineError("重试次数已用完", retryable=False)

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        delay = min(self._backoff_max, self._backoff_base * (2 ** attempt)) * (0.8 + 0.4 * random.random())
        if retry_after:
            delay = max(delay, min(retry_after, self._backoff_max))
        return delay

    def _unavailable_message(self, errors: dict[str, str], n: int) -> str:
        names = {info["id"]: info["name"] for info in ENGINE_INFO}
        detail = "；".join(f"{names.get(k, k)}：{v}" for k, v in errors.items() if v)
        msg = (f"翻译引擎暂时都不可用（可能是断网或被限流），还有 {n} 句没翻译。任务已暂停，不会输出未翻译的页面；"
               "请检查网络，稍等一两分钟后点「继续」。")
        return msg + (f"\n详情：{detail}" if detail else "")

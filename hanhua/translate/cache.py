"""翻译缓存：sqlite（WAL），线程安全。

key = (engine, src, tgt, 规范化原文) → 引擎原始输出（后处理之前的文本）。
- 存的是「发给引擎的文本」→「引擎返回的文本」，标点风格 / 口吃 / 术语还原都在取出后再做，
  所以改了 punct_style 或术语译名也不用清缓存；
- 缓存是锦上添花：数据库打不开 / 损坏 / 磁盘满时只记 warning，自动退到内存或直接失效，
  绝不让翻译流程因为缓存崩掉。
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
import unicodedata
from typing import Iterable, Sequence

log = logging.getLogger("hanhua.translate")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tr (
    engine TEXT NOT NULL,
    src    TEXT NOT NULL,
    tgt    TEXT NOT NULL,
    text   TEXT NOT NULL,
    out    TEXT NOT NULL,
    ts     INTEGER NOT NULL,
    PRIMARY KEY (engine, src, tgt, text)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS tr_ts ON tr(ts);
"""

_CHUNK = 400                 # 单条 SQL 里 IN (...) 的最大参数个数（远低于 sqlite 上限）
_DEFAULT_MAX_ROWS = 300_000  # 超过就按时间删掉最旧的，防止缓存无限长大


def cache_key_text(text: str) -> str:
    """缓存 key 用的原文规范化：NFC、去首尾空白、连续空白合一。"""
    if not text or not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()
    return re.sub(r"[ \t　]+", " ", s)


class TranslationCache:
    """线程安全的 sqlite 翻译缓存。path 为 "" 或 ":memory:" 时只用内存。"""

    def __init__(self, path: str = "", max_rows: int = _DEFAULT_MAX_ROWS) -> None:
        self.path = path or ":memory:"
        self.max_rows = max(1000, int(max_rows))
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._open()

    # ---- 打开 / 修复 ---------------------------------------------------------------------------
    def _connect(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=10, check_same_thread=False, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            if path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.execute("SELECT count(*) FROM tr LIMIT 1").fetchone()
        except Exception:
            conn.close()
            raise
        return conn

    def _open(self) -> None:
        if self.path != ":memory:":
            try:
                parent = os.path.dirname(os.path.abspath(self.path))
                os.makedirs(parent, exist_ok=True)
                self._conn = self._connect(self.path)
                self._prune()
                return
            except sqlite3.DatabaseError as e:
                log.warning("翻译缓存损坏（%s），改名备份后重建：%s", e, self.path)
                if self._quarantine():
                    try:
                        self._conn = self._connect(self.path)
                        return
                    except Exception as e2:  # noqa: BLE001 —— 缓存失败不能影响翻译
                        log.warning("重建翻译缓存失败：%s", e2)
            except Exception as e:  # noqa: BLE001
                log.warning("翻译缓存打不开（%s），本次只用内存缓存：%s", e, self.path)
        try:
            self._conn = self._connect(":memory:")
        except Exception as e:  # noqa: BLE001
            log.warning("内存缓存也开不了，缓存停用：%s", e)
            self._conn = None

    def _quarantine(self) -> bool:
        """把损坏的数据库（及 -wal / -shm）改名为 .corrupt-<时间戳>。"""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        ok = True
        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p):
                try:
                    os.replace(p, f"{p}.corrupt-{stamp}")
                except OSError as e:
                    log.warning("无法移走损坏的缓存文件 %s：%s", p, e)
                    ok = False
        return ok

    def _prune(self) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            n = conn.execute("SELECT count(*) FROM tr").fetchone()[0]
            if n > self.max_rows:
                drop = n - int(self.max_rows * 0.9)
                # WITHOUT ROWID 表没有 rowid，按主键元组删（sqlite >= 3.15 支持 row value）
                conn.execute("DELETE FROM tr WHERE (engine, src, tgt, text) IN "
                             "(SELECT engine, src, tgt, text FROM tr ORDER BY ts LIMIT ?)", (drop,))
                log.info("翻译缓存超过 %d 条，已清理最旧的 %d 条", self.max_rows, drop)
        except sqlite3.Error as e:
            log.warning("清理翻译缓存失败：%s", e)

    # ---- 读写 ----------------------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._conn is not None

    def get_many(self, engine: str, src: str, tgt: str, texts: Sequence[str]) -> dict[str, str]:
        """批量查询；返回 {原文: 引擎输出}（原文是调用方传入的原样字符串）。"""
        if self._conn is None or not texts:
            return {}
        keyed: dict[str, list[str]] = {}
        for t in texts:
            k = cache_key_text(t)
            if k:
                keyed.setdefault(k, []).append(t)
        if not keyed:
            return {}
        found: dict[str, str] = {}
        keys = list(keyed)
        try:
            with self._lock:
                conn = self._conn
                if conn is None:
                    return {}
                for i in range(0, len(keys), _CHUNK):
                    part = keys[i:i + _CHUNK]
                    marks = ",".join("?" * len(part))
                    rows = conn.execute(
                        f"SELECT text, out FROM tr WHERE engine=? AND src=? AND tgt=? AND text IN ({marks})",
                        (engine, src, tgt, *part)).fetchall()
                    for k, out in rows:
                        for orig in keyed.get(k, ()):
                            found[orig] = out
        except sqlite3.Error as e:
            log.warning("读翻译缓存失败（忽略）：%s", e)
            return {}
        return found

    def get(self, engine: str, src: str, tgt: str, text: str) -> str | None:
        return self.get_many(engine, src, tgt, [text]).get(text)

    def put_many(self, engine: str, src: str, tgt: str, pairs: Iterable[tuple[str, str]]) -> None:
        if self._conn is None:
            return
        now = int(time.time())
        rows = []
        for text, out in pairs:
            k = cache_key_text(text)
            if k and isinstance(out, str) and out.strip():
                rows.append((engine, src, tgt, k, out, now))
        if not rows:
            return
        try:
            with self._lock:
                conn = self._conn
                if conn is None:
                    return
                conn.execute("BEGIN")
                try:
                    conn.executemany(
                        "INSERT OR REPLACE INTO tr(engine, src, tgt, text, out, ts) VALUES (?,?,?,?,?,?)", rows)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
        except sqlite3.Error as e:
            log.warning("写翻译缓存失败（忽略）：%s", e)

    def put(self, engine: str, src: str, tgt: str, text: str, out: str) -> None:
        self.put_many(engine, src, tgt, [(text, out)])

    def delete(self, engine: str | None = None) -> int:
        """删除某个引擎（None = 全部）的缓存，返回删除条数。"""
        if self._conn is None:
            return 0
        try:
            with self._lock:
                if engine is None:
                    cur = self._conn.execute("DELETE FROM tr")
                else:
                    cur = self._conn.execute("DELETE FROM tr WHERE engine=?", (engine,))
                return max(0, cur.rowcount)
        except sqlite3.Error as e:
            log.warning("清除翻译缓存失败：%s", e)
            return 0

    def clear(self) -> int:
        return self.delete(None)

    def stats(self) -> dict[str, int]:
        """{引擎: 条数}，给设置页显示用。"""
        if self._conn is None:
            return {}
        try:
            with self._lock:
                rows = self._conn.execute("SELECT engine, count(*) FROM tr GROUP BY engine").fetchall()
            return {e: int(n) for e, n in rows}
        except sqlite3.Error as e:
            log.warning("统计翻译缓存失败：%s", e)
            return {}

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error as e:
                log.debug("关闭翻译缓存出错：%s", e)

    def __enter__(self) -> "TranslationCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

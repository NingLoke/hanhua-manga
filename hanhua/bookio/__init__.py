"""bookio：漫画「书」的读取与导出（合同见 ARCHITECTURE.md 5.6）。

    from hanhua.bookio import scan_inputs, open_book, Book, default_output_path, export_book

依赖方向：formats ← archive / pdf / epub ← export ← 本文件（对外入口 open_book / scan_inputs）。
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from ..types import HanhuaError, UnsupportedFormat
from .archive import ExtractedBook, ZipBook, nested_volumes, open_zip
from .epub import EpubBook
from .export import default_output_path, export_book, iter_output_kinds, output_kind
from .formats import (
    ARCHIVE_KINDS, EXTRACT_KINDS, INPUT_EXTS, LIST_SUFFIX, SUPPORTED_TEXT, ZIP_KINDS, Book, FolderBook,
    ImageListBook, find_origin, is_book_name, is_image_name, list_dir, make_image_list, resolve_kind,
    tmp_dir_or_default,
)
from .pdf import PdfBook

__all__ = [
    "scan_inputs", "open_book", "Book", "default_output_path", "export_book", "output_kind", "iter_output_kinds",
    "FolderBook", "ImageListBook", "ZipBook", "ExtractedBook", "PdfBook", "EpubBook",
    "INPUT_EXTS", "SUPPORTED_TEXT", "UnsupportedFormat",
]

log = logging.getLogger("hanhua.bookio")

MAX_SCAN_DEPTH = 8          # 拖进来的文件夹最多往下找 8 层
MAX_SCAN_DIRS = 20000       # 防止误拖整个磁盘时卡死
MAX_BOOKS = 5000
MAX_NESTED_DEPTH = 3        # 套娃最多拆 3 层
# 套娃卷名太「通用」（01、Vol.2、第3卷）时在前面加上最外层压缩包的标题，免得输出叫「01_汉化.cbz」
_GENERIC_TITLE = re.compile(
    r"^[\W_]*(?:vol(?:ume)?|v|ch(?:apter)?|ep|第)?[\W_]*\d+[\W_]*(?:卷|巻|话|話|集|册|冊)?[\W_]*$", re.I)


# ---- open_book -------------------------------------------------------------------------------------
def _open(p: str, tmp: str) -> Book:
    if os.path.isdir(p):
        return FolderBook(p)
    if not os.path.exists(p):
        raise UnsupportedFormat(f"找不到文件：{p}")
    if not os.path.isfile(p):
        raise UnsupportedFormat(f"这不是普通文件，无法打开：{p}")
    if p.lower().endswith(LIST_SUFFIX):
        return ImageListBook.from_list_file(p)
    if is_image_name(p):
        return ImageListBook.single(p)
    kind = resolve_kind(p)
    if kind in ZIP_KINDS:
        return open_zip(p, kind, tmp)
    if kind in EXTRACT_KINDS:
        return ExtractedBook(p, kind, tmp)
    if kind == "pdf":
        return PdfBook(p, tmp)
    if kind == "epub":
        return EpubBook(p)
    raise UnsupportedFormat(f"不支持这种文件：{os.path.basename(p)}（支持：{SUPPORTED_TEXT}）")


def _apply_origin(book: Book) -> None:
    """套娃展开出来的卷：默认输出放到最外层压缩包旁边；卷名太通用时加上外层标题。"""
    found = find_origin(book.path)
    if not found:
        return
    origin, outer_title = found
    book.origin_dir = os.path.dirname(os.path.abspath(origin)) or book.origin_dir
    if outer_title and outer_title not in book.title and _GENERIC_TITLE.match(book.title):
        book.title = f"{outer_title} {book.title}"


def open_book(path: str, tmp_dir: str) -> Book:
    """打开一本书。失败抛 UnsupportedFormat（中文说明）；任何意外异常也会包成 UnsupportedFormat，不让流程崩。"""
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str) or not path.strip():
        raise UnsupportedFormat("没有指定要打开的文件。")
    p = os.path.abspath(os.path.expanduser(path.strip().strip('"').strip()))
    tmp = tmp_dir_or_default(tmp_dir)
    try:
        book = _open(p, tmp)
    except HanhuaError:
        raise
    except MemoryError as e:
        raise UnsupportedFormat(f"「{os.path.basename(p)}」太大，内存不足，无法打开。") from e
    except Exception as e:  # noqa: BLE001 - 坏文件千奇百怪，统一给中文提示
        log.exception("打开失败：%s", p)
        raise UnsupportedFormat(f"无法打开「{os.path.basename(p)}」：文件可能已损坏（{type(e).__name__}）。") from e
    try:
        _apply_origin(book)
    except Exception:  # noqa: BLE001 - 只影响默认输出位置，不影响读取
        log.warning("读取套娃来源信息失败：%s", p, exc_info=True)
    log.info("已打开 %s：%s（%d 页）", book.kind, p, len(book.pages))
    return book


# ---- scan_inputs -----------------------------------------------------------------------------------
class _Scanner:
    def __init__(self, tmp_root: str | None) -> None:
        self.tmp_root = tmp_dir_or_default(tmp_root)
        self.slots: list[str | tuple[str, str]] = []      # 书路径，或 ("loose", 目录键) 占位（保持拖入顺序）
        self.loose: dict[str, list[str]] = {}
        self.seen: set[str] = set()
        self.dirs = 0

    def add_input(self, raw: Any) -> None:
        try:
            if isinstance(raw, os.PathLike):
                raw = os.fspath(raw)
            if not isinstance(raw, str) or not raw.strip():
                return
            p = os.path.abspath(os.path.expanduser(raw.strip().strip('"').strip()))
            if os.path.isdir(p):
                self._walk(p)
            elif os.path.isfile(p):
                self._file(p, direct=True)
            else:
                log.warning("拖进来的路径不存在，已跳过：%s", p)
        except Exception:  # noqa: BLE001 - 一个坏路径不能让整批失败
            log.exception("扫描输入时出错，已跳过：%r", raw)

    def _book(self, p: str) -> None:
        key = os.path.normcase(os.path.abspath(p))
        if key in self.seen:
            return
        if len(self.seen) >= MAX_BOOKS:
            log.warning("一次拖入的书超过 %d 本，后面的已忽略", MAX_BOOKS)
            return
        self.seen.add(key)
        self.slots.append(p)

    def _file(self, p: str, direct: bool = False, depth: int = 0) -> None:
        if p.lower().endswith(LIST_SUFFIX):
            self._book(p)
            return
        if is_image_name(p):
            if direct:                                    # 直接拖进来的散图：同目录的算一本
                d = os.path.normcase(os.path.dirname(p))
                if d not in self.loose:
                    self.loose[d] = []
                    self.slots.append(("loose", d))
                self.loose[d].append(p)
            return
        kind = resolve_kind(p) if (direct or is_book_name(p)) else None
        if kind is None:
            if direct:
                log.info("不支持的文件，已跳过：%s", p)
            return
        if kind in ARCHIVE_KINDS and depth < MAX_NESTED_DEPTH:
            vols = nested_volumes(p, kind, self.tmp_root)
            if vols:
                log.info("「%s」是套娃压缩包，展开为 %d 卷", os.path.basename(p), len(vols))
                for v in vols:
                    self._file(v, depth=depth + 1)
                return
        self._book(p)

    def _walk(self, root: str) -> None:
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            self.dirs += 1
            if self.dirs > MAX_SCAN_DIRS:
                log.warning("文件夹太多（超过 %d 个），后面的已忽略：%s", MAX_SCAN_DIRS, root)
                return
            files, subdirs = list_dir(d)
            if any(is_image_name(f) for f in files):
                self._book(d)                             # 直接含图片的文件夹 = 一本（子文件夹另算）
            for f in files:
                if is_book_name(f):
                    self._file(f)
            if depth < MAX_SCAN_DEPTH:
                stack.extend((s, depth + 1) for s in reversed(subdirs))

    def result(self) -> list[str]:
        out: list[str] = []
        for slot in self.slots:
            if isinstance(slot, str):
                out.append(slot)
                continue
            files = self.loose.get(slot[1]) or []
            try:
                if len(files) == 1:
                    path = files[0]                       # 单张图：open_book 直接按图片打开
                else:
                    path = make_image_list(files, self.tmp_root)
            except Exception:  # noqa: BLE001
                log.exception("散图分组失败，已跳过：%s", slot[1])
                continue
            key = os.path.normcase(os.path.abspath(path))
            if key not in self.seen:                      # 写清单失败会退回整个文件夹，可能和已有的书重复
                self.seen.add(key)
                out.append(path)
        return out


def scan_inputs(paths: list[str], tmp_root: str) -> list[str]:
    """把拖进来的路径展开成「书」的列表：压缩包 / pdf / epub / 含图片的文件夹 / 散图（同目录散图算一本）。

    套娃压缩包（zip 里全是 cbz）展开到 tmp_root 下，每卷一本。坏路径、不支持的文件只记日志并跳过。
    """
    scanner = _Scanner(tmp_root)
    items = paths if isinstance(paths, (list, tuple)) else [paths]
    for p in items:
        scanner.add_input(p)
    return scanner.result()

"""EPUB 书：META-INF/container.xml → OPF（manifest + spine）→ 按 spine 顺序打开每个 XHTML，
取 <img src> 和 SVG <image xlink:href> 指向的图片（去重）作为页面；其它条目全部是 extras（导出 epub 时原样复制）。

找不到 OPF / spine 里没有图时依次退回：manifest 里的图片顺序 → 所有图片按文件名自然排序。
"""
from __future__ import annotations

import html
import logging
import os
import posixpath
import re
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from ..types import UnsupportedFormat
from .formats import Book, ZipReader, clean_entry_name, decode_text, is_image_name, natural_key, title_from_name

log = logging.getLogger("hanhua.bookio")

CONTAINER = "META-INF/container.xml"
ENCRYPTION = "META-INF/encryption.xml"
MAX_DOC_BYTES = 16 << 20                       # 单个 XHTML / OPF 最多读 16MB，防恶意大文件
_DOC_TYPES = frozenset({"application/xhtml+xml", "text/html", "application/xml", "text/xml", "image/svg+xml"})
_DOC_EXTS = (".xhtml", ".html", ".htm", ".xml", ".svg")
_TAG = re.compile(r"<\s*(?:[A-Za-z][\w.-]*:)?(img|image)\b([^>]*)>", re.I | re.S)
_ATTR = re.compile(r"""([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.S)
_OPF_TAG = re.compile(r"<\s*(?:[A-Za-z][\w.-]*:)?(item|itemref|spine)\b([^>]*)>", re.I | re.S)
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _attrs(s: str) -> dict[str, str]:
    return {m.group(1).lower(): (m.group(2) if m.group(2) is not None else m.group(3) or "") for m in _ATTR.finditer(s)}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()


def resolve_href(base_dir: str, href: str) -> str:
    """XHTML / OPF 里的相对引用 → 包内条目名；外部链接、data: URI 返回空串。"""
    ref = unquote(html.unescape(href or "").strip().split("#", 1)[0].split("?", 1)[0])
    if not ref or _SCHEME.match(ref):
        return ""
    return clean_entry_name(posixpath.normpath(posixpath.join(base_dir, ref)))


def parse_opf(text: str) -> tuple[list[tuple[str, str, str]], list[str], str, str]:
    """(manifest [(id, href, media-type)], spine idref 列表, dc:title, page-progression-direction)。

    先用 ElementTree（不会去取外部实体）；XML 不规范时退回正则。
    """
    body = _XML_DECL.sub("", text, count=1)
    manifest: list[tuple[str, str, str]] = []
    spine: list[str] = []
    title = ppd = ""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        root = None
    if root is not None:
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            tag = _local(el.tag)
            a = {_local(k): v for k, v in el.attrib.items()}
            if tag == "item":
                manifest.append((a.get("id", ""), a.get("href", ""), a.get("media-type", "").lower()))
            elif tag == "itemref":
                spine.append(a.get("idref", ""))
            elif tag == "spine":
                ppd = a.get("page-progression-direction", "")
            elif tag == "title" and not title:
                title = (el.text or "").strip()
        return manifest, spine, title, ppd
    for m in _OPF_TAG.finditer(body):
        kind, a = m.group(1).lower(), _attrs(m.group(2))
        if kind == "item":
            manifest.append((a.get("id", ""), a.get("href", ""), a.get("media-type", "").lower()))
        elif kind == "itemref":
            spine.append(a.get("idref", ""))
        else:
            ppd = a.get("page-progression-direction", "")
    t = re.search(r"<\s*dc:title\b[^>]*>(.*?)<", body, re.I | re.S)
    if t:
        title = html.unescape(t.group(1)).strip()
    return manifest, spine, title, ppd


class EpubBook(Book):
    """EPUB：页面 = spine 顺序里出现的图片；entries = 全部条目（原顺序，导出 epub 时照抄）。"""

    def __init__(self, path: str) -> None:
        src = os.path.abspath(path)
        super().__init__(src, "epub", title_from_name(src), os.path.dirname(src))
        self._name = os.path.basename(src)
        self._reader = ZipReader(src)
        try:
            entries = self._reader.entries()
        except UnsupportedFormat as e:
            raise UnsupportedFormat(f"EPUB「{self._name}」已损坏或不是有效的 EPUB 文件。") from e
        self._infos = dict(entries)
        self.entries: list[str] = [n for n, _ in entries]
        self._lower: dict[str, str] = {}
        for n in self.entries:
            self._lower.setdefault(n.lower(), n)
        self.opf_path = ""
        self.meta_title = ""
        self.page_progression = ""
        pages = self._find_pages()
        if not pages:
            self.close()
            raise UnsupportedFormat(f"EPUB「{self._name}」里没有找到图片（可能是纯文字小说）。")
        self._check_drm(pages)
        page_set = set(pages)
        self._set_entries(pages, [n for n in self.entries if n not in page_set])

    # -- 解析 --
    def _lookup(self, name: str) -> str:
        if not name:
            return ""
        if name in self._infos:
            return name
        return self._lower.get(name.lower(), "")     # 有的书引用大小写和实际文件不一致

    def _text(self, name: str) -> str:
        info = self._infos[name]
        if info.file_size > MAX_DOC_BYTES:
            log.warning("EPUB 文档太大，跳过：%s", name)
            return ""
        return decode_text(self._reader.read(info, name))

    def _find_opf(self) -> str:
        cont = self._lookup(CONTAINER)
        if cont:
            try:
                text = self._text(cont)
            except UnsupportedFormat:
                text = ""
            for m in re.finditer(r"<\s*(?:[A-Za-z][\w.-]*:)?rootfile\b([^>]*)>", text, re.I | re.S):
                a = _attrs(m.group(1))
                mt = a.get("media-type", "application/oebps-package+xml").lower()
                full = self._lookup(resolve_href("", a.get("full-path", "")))
                if full and "oebps-package" in mt:
                    return full
        opfs = sorted((n for n in self.entries if n.lower().endswith(".opf")), key=natural_key)
        return opfs[0] if opfs else ""

    def _doc_images(self, doc: str) -> list[str]:
        try:
            text = _COMMENT.sub("", self._text(doc))
        except UnsupportedFormat as e:
            log.warning("EPUB 页面读取失败，跳过：%s（%s）", doc, e)
            return []
        base = posixpath.dirname(doc)
        out: list[str] = []
        for m in _TAG.finditer(text):
            a = _attrs(m.group(2))
            if m.group(1).lower() == "img":
                href = a.get("src", "")
            else:
                href = a.get("xlink:href") or a.get("href") or next((v for k, v in a.items() if k.endswith(":href")), "")
            full = self._lookup(resolve_href(base, href))
            if full:
                out.append(full)
        return out

    def _find_pages(self) -> list[str]:
        pages: list[str] = []
        seen: set[str] = set()

        def add(n: str) -> None:
            if n and n not in seen and is_image_name(n):
                seen.add(n)
                pages.append(n)

        opf = self._find_opf()
        if opf:
            self.opf_path = opf
            try:
                manifest, spine, self.meta_title, self.page_progression = parse_opf(self._text(opf))
            except UnsupportedFormat as e:
                log.warning("EPUB 的 OPF 读不了：%s", e)
                manifest, spine = [], []
            base = posixpath.dirname(opf)
            items: dict[str, tuple[str, str]] = {}
            for iid, href, mt in manifest:
                full = self._lookup(resolve_href(base, href))
                if iid and full:
                    items.setdefault(iid, (full, mt))
            for idref in spine:
                full, mt = items.get(idref, ("", ""))
                if not full:
                    continue
                if mt.startswith("image/") and mt != "image/svg+xml" or is_image_name(full):
                    add(full)                           # 有的书直接把图片放进 spine
                elif mt in _DOC_TYPES or full.lower().endswith(_DOC_EXTS):
                    for ref in self._doc_images(full):
                        add(ref)
            if not pages:
                log.info("EPUB spine 里没有图片，按 manifest 顺序取：%s", self._name)
                for full, mt in items.values():
                    if mt.startswith("image/") or is_image_name(full):
                        add(full)
        if not pages:
            for n in sorted(self.entries, key=natural_key):
                add(n)
        return pages

    def _check_drm(self, pages: list[str]) -> None:
        enc = self._lookup(ENCRYPTION)
        if not enc:
            return
        try:
            text = self._text(enc)
        except UnsupportedFormat:
            return
        uris = {self._lookup(resolve_href("", u)) for u in re.findall(r"URI\s*=\s*[\"']([^\"']+)[\"']", text)}
        if uris & set(pages):                          # 只加密字体（字体混淆）不算，图片被加密才是 DRM
            self.close()
            raise UnsupportedFormat(f"EPUB「{self._name}」有 DRM 加密（图片被加密），无法处理。请使用无 DRM 的版本。")

    # -- Book 接口 --
    def read_entry(self, name: str) -> bytes:
        """任意条目（页面或 extras）的原始字节。"""
        info = self._infos.get(name)
        if info is None:
            raise UnsupportedFormat(f"EPUB 里没有这个条目：{name}")
        return self._reader.read(info, name)

    def entry_compress_type(self, name: str) -> int:
        info = self._infos.get(name)
        return info.compress_type if info is not None else 8

    def read_page_bytes(self, name: str) -> bytes:
        self._need_page(name)
        return self.read_entry(name)

    def read_extra(self, name: str) -> bytes:
        self._need_extra(name)
        return self.read_entry(name)

    def close(self) -> None:
        self._reader.close()

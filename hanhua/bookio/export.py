"""导出：把翻译好的页面写成 文件夹 / zip / cbz / pdf / epub。

* 一律先写 <输出>.part（文件夹输出则是 <输出>.part/ 目录），全部成功后 os.replace 成最终名字；
  中途出任何异常（含用户取消）都删掉 .part，不留半成品；
* zip / cbz：图片 ZIP_STORED（本身已压缩，再 deflate 只会更慢）；extras 原样带走；
  ComicInfo.xml 标 LanguageISO=zh、标题加后缀（cbz 没有时新建一份）；
* pdf：轻量 writer，每页一个 DCTDecode JPEG XObject，逐页写盘，不把整本书放进内存；
* epub：源是 epub 时复制原包、只替换图片条目（图片保持原文件名和格式，XHTML 引用不会失效）；
  其它来源生成一本固定版式（pre-paginated）EPUB 3；
* image_format=same：沿用原扩展名；gif / bmp / tif / avif 这些不适合输出的改成 png。
"""
from __future__ import annotations

import datetime as _dt
import html
import io
import logging
import os
import posixpath
import re
import shutil
import time
import uuid
import zipfile
from typing import Any, Callable, Iterable
from xml.sax.saxutils import escape as _esc

from .. import imageio as hio
from ..types import HanhuaError, UnsupportedFormat
from .epub import EpubBook
from .formats import Book, clean_entry_name, decode_text, image_info, natural_key, safe_relpath, sanitize_filename

log = logging.getLogger("hanhua.bookio")

Progress = Callable[[int, int], None]
OUTPUT_KINDS = ("folder", "cbz", "zip", "pdf", "epub")
_SAME_OUTPUT = {
    "zip": "zip", "cbz": "cbz", "7z": "cbz", "cb7": "cbz", "rar": "cbz", "cbr": "cbz", "tar": "cbz", "cbt": "cbz",
    "pdf": "pdf", "epub": "epub", "folder": "folder", "images": "folder",
}
_FAMILY_OF_EXT = {
    ".jpg": "jpg", ".jpeg": "jpg", ".jfif": "jpg", ".png": "png", ".webp": "webp", ".gif": "gif",
    ".bmp": "bmp", ".tif": "tif", ".tiff": "tif", ".avif": "avif",
}
_KEEP_FAMILIES = frozenset({"jpg", "png", "webp"})          # image_format=same 时能原样沿用的格式
_MEDIA = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif",
          "bmp": "image/bmp", "tif": "image/tiff", "avif": "image/avif"}
_PIL_FORMAT = {"gif": "GIF", "tif": "TIFF", "avif": "AVIF"}
PDF_MAX_PT = 14400.0              # Acrobat 的页面边长上限（200 英寸）；更长的条漫等比缩小页面尺寸，图片像素不变
PRODUCER = "一键汉化漫画 4.0"
_EPUB_MIMETYPE = b"application/epub+zip"


# ---- 设置 ------------------------------------------------------------------------------------------
def _out(settings: Any) -> dict:
    o = settings.get("output") if isinstance(settings, dict) else None
    return o if isinstance(o, dict) else {}


def _quality(o: dict, key: str, default: int) -> int:
    try:
        q = int(o.get(key, default))
    except (TypeError, ValueError):
        q = default
    return max(50, min(100, q))


def _suffix(settings: Any) -> str:
    s = _out(settings).get("suffix")
    return "_汉化" if s is None else str(s)


def _is_rtl(settings: Any) -> bool:
    src = settings.get("source") if isinstance(settings, dict) else None
    src = src if isinstance(src, dict) else {}
    order = str(src.get("reading_order") or "auto").lower()
    if order in ("rtl", "ltr"):
        return order == "rtl"
    return str(src.get("lang") or "ja").lower() == "ja"        # 日漫右翻，韩漫 / 美漫左翻


def _target_lang(settings: Any) -> str:
    tr = settings.get("translate") if isinstance(settings, dict) else None
    target = str((tr or {}).get("target") or "zh-Hans") if isinstance(tr, dict) else "zh-Hans"
    return "zh-Hant" if target.lower() == "zh-hant" else "zh-Hans"


def output_kind(book: Book, settings: Any) -> str:
    """最终输出种类：folder | cbz | zip | pdf | epub（format=same 时按源格式映射，7z / rar / tar → cbz）。"""
    fmt = str(_out(settings).get("format") or "same").strip().lower()
    if fmt in OUTPUT_KINDS:
        return fmt
    return _SAME_OUTPUT.get(book.kind, "cbz")


def default_output_path(book: Book, settings: dict) -> str:
    """output.dir 为空则放在源文件旁边（套娃卷 = 最外层压缩包旁边）；文件名 = 标题 + output.suffix + 扩展名；重名加 (2)(3)…"""
    o = _out(settings)
    kind = output_kind(book, settings)
    base_dir = str(o.get("dir") or "").strip() or book.origin_dir or os.path.dirname(book.path) or os.getcwd()
    base_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(base_dir)))
    stem = sanitize_filename(f"{book.title}{_suffix(settings)}", "漫画_汉化")
    ext = "" if kind == "folder" else "." + kind
    cand = os.path.join(base_dir, stem + ext)
    k = 2
    while os.path.lexists(cand):
        if k > 9999:
            return os.path.join(base_dir, f"{stem} ({time.strftime('%Y%m%d-%H%M%S')}){ext}")
        cand = os.path.join(base_dir, f"{stem} ({k}){ext}")
        k += 1
    return cand


# ---- 页面数据 --------------------------------------------------------------------------------------
def _items(pages: Any) -> list[tuple[str, str | None]]:
    """[(页面名, 渲染结果路径)] 规范化：去重、跳过格式不对的条目（绝不因为一条坏数据崩掉）。"""
    out: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for it in pages or []:
        try:
            name, src = it[0], (it[1] if len(it) > 1 else None)
        except (TypeError, IndexError, KeyError):
            log.warning("忽略格式不对的页面条目：%r", it)
            continue
        if not isinstance(name, str) or not name or name in seen:
            continue
        try:
            src = os.fspath(src) if src else None
        except TypeError:
            log.warning("渲染结果路径格式不对，改用原图：%s（%r）", name, src)
            src = None
        seen.add(name)
        out.append((name, src))
    return out


def _load_rendered(book: Book, name: str, src: str | None) -> bytes:
    """渲染结果的字节；文件缺失 / 为空 / 读不了时退回原页面（记警告），保证导出的书不缺页。"""
    if src:
        try:
            with open(src, "rb") as f:
                data = f.read()
            if data:
                return data
            log.warning("渲染结果是空文件，改用原图：%s", src)
        except OSError as e:
            log.warning("读不到渲染结果（%s），改用原图：%s", e, name)
    else:
        log.warning("页面没有渲染结果，改用原图：%s", name)
    return book.read_page_bytes(name)


def _target(name: str, o: dict) -> tuple[str, str]:
    """(目标格式 jpg|png|webp, 输出扩展名)。沿用原格式时保留原扩展名的写法（.jpeg / .JPG）。"""
    fmt = str(o.get("image_format") or "same").strip().lower()
    ext = os.path.splitext(name)[1]
    fam = _FAMILY_OF_EXT.get(ext.lower())
    if fmt in ("jpg", "jpeg"):
        return "jpg", ext if fam == "jpg" else ".jpg"
    if fmt in ("png", "webp"):
        return fmt, ext if fam == fmt else "." + fmt
    if fam in _KEEP_FAMILIES:
        return fam, ext
    return "png", ".png"


def _encode_pil(img: Any, fam: str) -> bytes:
    buf = io.BytesIO()
    hio.to_pil(img).save(buf, _PIL_FORMAT[fam])
    return buf.getvalue()


def _convert(data: bytes, fam: str, o: dict, label: str) -> bytes:
    """解码再按目标格式编码；gif / tif / avif 只有 epub 原样替换时才会用到（Pillow 编码，失败退回 png）。"""
    try:
        img = hio.decode(data)
    except ValueError as e:
        raise UnsupportedFormat(f"页面图片已损坏，无法导出：{label}") from e
    try:
        if fam in _PIL_FORMAT:
            try:
                return _encode_pil(img, fam)
            except Exception as e:  # noqa: BLE001 - 编码器缺失等
                log.warning("无法编码成 %s（%s），改存 png 数据：%s", fam, e, label)
                return hio.encode(img, "png")
        q = _quality(o, "webp_quality", 90) if fam == "webp" else _quality(o, "jpeg_quality", 92)
        return hio.encode(img, fam, q)
    except ValueError as e:
        raise UnsupportedFormat(f"页面图片编码失败，无法导出：{label}") from e


def _page_bytes(book: Book, name: str, src: str | None, o: dict,
                keep_family: str | None = None) -> tuple[bytes, str, str]:
    """(图片字节, 输出条目名, 格式)。格式和目标一致时直接复制渲染结果，不重新编码（不掉画质）。"""
    raw = _load_rendered(book, name, src)
    if keep_family:                                   # epub 原样替换：保持原文件名和原格式
        fam, out_name = keep_family, name
    else:
        fam, ext = _target(name, o)
        root, old_ext = os.path.splitext(name)
        out_name = name if ext.lower() == old_ext.lower() else root + ext
    info = image_info(raw)
    if info is not None and info[0] == fam:
        return raw, out_name, fam
    return _convert(raw, fam, o, name), out_name, fam


def _pdf_jpeg(book: Book, name: str, src: str | None, o: dict) -> tuple[bytes, int, int, int]:
    """PDF 用的 JPEG：(字节, 宽, 高, 通道数 1|3)。8 位基线 / 渐进的灰度或彩色 JPEG 直接嵌入，其它重新编码。"""
    raw = _load_rendered(book, name, src)
    info = image_info(raw)
    if info is not None and info[0] == "jpg" and info[1] > 0 and info[2] > 0 and info[3] in (1, 3):
        return raw, info[1], info[2], info[3]
    try:
        img = hio.decode(raw)
        data = hio.encode(img, "jpg", _quality(o, "jpeg_quality", 92))
    except ValueError as e:
        raise UnsupportedFormat(f"页面图片已损坏，无法导出 PDF：{name}") from e
    h, w = img.shape[:2]
    return data, int(w), int(h), 3


def _dims(data: bytes) -> tuple[int, int]:
    info = image_info(data)
    if info is not None and info[1] > 0 and info[2] > 0:
        return info[1], info[2]
    try:
        h, w = hio.decode(data).shape[:2]
        return int(w), int(h)
    except ValueError:
        return 1000, 1500


def _unique(rel: str, used: set[str]) -> str:
    """条目名去重（不分大小写：Windows 文件系统和多数阅读器都不分）。"""
    if rel.lower() not in used:
        used.add(rel.lower())
        return rel
    stem, ext = os.path.splitext(rel)
    k = 2
    while f"{stem}_{k}{ext}".lower() in used:
        k += 1
    out = f"{stem}_{k}{ext}"
    used.add(out.lower())
    return out


def _tick(progress: Progress | None, done: int, total: int) -> None:
    if progress is not None:
        progress(done, total)                         # 回调抛出的异常（如 Cancelled）照常往外抛，由 export_book 清理


# ---- ComicInfo.xml ---------------------------------------------------------------------------------
_CI_CLOSE = re.compile(r"</\s*ComicInfo\s*>", re.I)


def _is_comicinfo(name: str) -> bool:
    return posixpath.basename(name).lower() == "comicinfo.xml"


def _ci_set(text: str, tag: str, value: Callable[[str], str]) -> str:
    """把 <tag>旧值</tag> 换成 value(旧值)；没有这个标签就插到 </ComicInfo> 前面。"""
    pat = re.compile(rf"<\s*{tag}\b[^>/]*/\s*>|<\s*{tag}\b[^>]*>(.*?)</\s*{tag}\s*>", re.I | re.S)
    m = pat.search(text)
    if m:
        old = html.unescape(m.group(1) or "").strip()
        return f"{text[:m.start()]}<{tag}>{_esc(value(old))}</{tag}>{text[m.end():]}"
    closes = list(_CI_CLOSE.finditer(text))
    if not closes:
        return text
    pos = closes[-1].start()
    return f"{text[:pos]}  <{tag}>{_esc(value(''))}</{tag}>\n{text[pos:]}"


def patch_comicinfo(data: bytes, title: str, suffix: str) -> bytes:
    """ComicInfo.xml：LanguageISO=zh，Title 加后缀（已有后缀不重复加）；其余内容和排版原样保留。"""
    try:
        text = decode_text(data)
        if not re.search(r"<\s*ComicInfo\b", text, re.I):
            return data
        text = re.sub(r"<\s*ComicInfo\b([^>]*?)/\s*>", r"<ComicInfo\1></ComicInfo>", text, count=1, flags=re.I)

        def new_title(old: str) -> str:
            base = old or title
            return base if (not suffix or base.endswith(suffix)) else base + suffix

        text = _ci_set(text, "Title", new_title)
        text = _ci_set(text, "LanguageISO", lambda _old: "zh")
        text = re.sub(r"""^(\s*<\?xml[^>]*?encoding\s*=\s*["'])[^"']*(["'])""", r"\1utf-8\2", text, count=1)
        return text.encode("utf-8")
    except Exception:  # noqa: BLE001 - 元数据改不了就原样带走，不影响导出
        log.warning("ComicInfo.xml 修改失败，原样保留", exc_info=True)
        return data


def new_comicinfo(title: str, pages: int, rtl: bool) -> bytes:
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<ComicInfo xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
        f"  <Title>{_esc(title)}</Title>",
        f"  <PageCount>{int(pages)}</PageCount>",
        "  <LanguageISO>zh</LanguageISO>",
    ]
    if rtl:
        lines.append("  <Manga>YesAndRightToLeft</Manga>")
    lines.append("</ComicInfo>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _extras(book: Book) -> list[str]:
    if book.kind == "epub":           # epub 的 xhtml / css / opf / 字体对 cbz / 文件夹没意义，只带 ComicInfo.xml
        return [x for x in book.extras if _is_comicinfo(x)]
    return list(book.extras)


def _read_extra(book: Book, name: str) -> bytes | None:
    try:
        return book.read_extra(name)
    except HanhuaError as e:
        log.warning("附加文件读不了，已跳过：%s（%s）", name, e)
        return None


# ---- 写出：zip / cbz / 文件夹 ----------------------------------------------------------------------
def _zinfo(name: str, compress: int) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
    zi.compress_type = compress
    zi.external_attr = 0o644 << 16
    return zi


def _needs_renumber(items: list[tuple[str, str | None]]) -> bool:
    """阅读顺序无法从文件名恢复（epub 的 spine、pdf 页序等）→ 输出成 zip / cbz / 文件夹时要按序号重命名，
    否则阅读器按文件名排序会把页面顺序打乱。"""
    names = [n for n, _ in items]
    return sorted(names, key=natural_key) != names


def _write_zip(book: Book, items: list[tuple[str, str | None]], part: str, o: dict, progress: Progress | None,
               title: str, suffix: str, comic: bool, rtl: bool) -> None:
    total = len(items)
    used: set[str] = set()
    renum = _needs_renumber(items)
    with zipfile.ZipFile(part, "w", allowZip64=True) as zf:
        for i, (name, src) in enumerate(items, 1):
            data, out_name, fam = _page_bytes(book, name, src, o)
            arc = _unique(f"{i:04d}.{fam}" if renum else (clean_entry_name(out_name) or f"{i:04d}.{fam}"), used)
            zf.writestr(_zinfo(arc, zipfile.ZIP_STORED), data)
            _tick(progress, i, total)
        has_ci = False
        for x in _extras(book):
            data = _read_extra(book, x)
            if data is None:
                continue
            if _is_comicinfo(x):
                data, has_ci = patch_comicinfo(data, title, suffix), True
            zf.writestr(_zinfo(_unique(clean_entry_name(x) or "extra", used), zipfile.ZIP_DEFLATED), data)
        if comic and not has_ci:
            zf.writestr(_zinfo(_unique("ComicInfo.xml", used), zipfile.ZIP_DEFLATED),
                        new_comicinfo(title + suffix, total, rtl))


def _write_file(root: str, rel: str, data: bytes) -> None:
    full = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)


def _write_folder(book: Book, items: list[tuple[str, str | None]], part: str, o: dict, progress: Progress | None,
                  title: str, suffix: str) -> None:
    os.makedirs(part, exist_ok=True)
    total = len(items)
    used: set[str] = set()
    renum = _needs_renumber(items)
    for i, (name, src) in enumerate(items, 1):
        data, out_name, fam = _page_bytes(book, name, src, o)
        rel = f"{i:04d}.{fam}" if renum else safe_relpath(out_name)       # 顺序可由文件名恢复时保持相对路径
        _write_file(part, _unique(rel, used), data)
        _tick(progress, i, total)
    for x in _extras(book):
        data = _read_extra(book, x)
        if data is None:
            continue
        if _is_comicinfo(x):
            data = patch_comicinfo(data, title, suffix)
        _write_file(part, _unique(safe_relpath(x), used), data)


# ---- 写出：pdf -------------------------------------------------------------------------------------
def _num(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s or "0"


def _pdf_text(s: str) -> bytes:
    return b"<FEFF" + s.encode("utf-16-be").hex().upper().encode("ascii") + b">"


class _PdfWriter:
    """最小 PDF 1.4 writer：对象逐个写盘，内存里只记 xref 偏移。"""

    def __init__(self, f: Any) -> None:
        self.f = f
        self.pos = 0
        self.offsets: dict[int, int] = {}

    def raw(self, b: bytes) -> None:
        self.f.write(b)
        self.pos += len(b)

    def obj(self, num: int, body: bytes, stream: bytes | None = None) -> None:
        self.offsets[num] = self.pos
        self.raw(b"%d 0 obj\n" % num + body)
        if stream is not None:
            self.raw(b"\nstream\n")
            self.raw(stream)
            self.raw(b"\nendstream")
        self.raw(b"\nendobj\n")

    def finish(self, root: int, info: int, size: int) -> None:
        xref = self.pos
        rows = [b"xref\n0 %d\n0000000000 65535 f \n" % size]
        for n in range(1, size):
            off = self.offsets.get(n)
            rows.append(b"%010d 00000 n \n" % off if off is not None else b"0000000000 65535 f \n")
        self.raw(b"".join(rows))
        fid = uuid.uuid4().hex.encode("ascii")
        self.raw(b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R /ID [<%s> <%s>] >>\nstartxref\n%d\n%%%%EOF\n"
                 % (size, root, info, fid, fid, xref))


def _write_pdf(book: Book, items: list[tuple[str, str | None]], part: str, o: dict, progress: Progress | None,
               title: str, rtl: bool) -> None:
    total = len(items)
    with open(part, "wb") as f:
        w = _PdfWriter(f)
        w.raw(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        kids: list[int] = []
        for i, (name, src) in enumerate(items):
            jpeg, pw, ph, comps = _pdf_jpeg(book, name, src, o)
            k = min(1.0, PDF_MAX_PT / max(pw, ph))     # 1px = 1pt；超长条漫等比缩小页面尺寸（像素不变）
            wpt, hpt = _num(pw * k), _num(ph * k)
            img_n, cont_n, page_n = 4 + 3 * i, 5 + 3 * i, 6 + 3 * i
            cs = "/DeviceGray" if comps == 1 else "/DeviceRGB"
            w.obj(img_n, (f"<< /Type /XObject /Subtype /Image /Width {pw} /Height {ph} /ColorSpace {cs} "
                          f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpeg)} >>").encode("ascii"), jpeg)
            content = f"q {wpt} 0 0 {hpt} 0 0 cm /Im0 Do Q".encode("ascii")
            w.obj(cont_n, b"<< /Length %d >>" % len(content), content)
            w.obj(page_n, (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {wpt} {hpt}] "
                           f"/Resources << /XObject << /Im0 {img_n} 0 R >> /ProcSet [/PDF /ImageB /ImageC] >> "
                           f"/Contents {cont_n} 0 R >>").encode("ascii"))
            kids.append(page_n)
            _tick(progress, i + 1, total)
        prefs = b" /ViewerPreferences << /Direction /R2L >>" if rtl else b""
        w.obj(1, b"<< /Type /Catalog /Pages 2 0 R" + prefs + b" >>")
        w.obj(2, f"<< /Type /Pages /Kids [{' '.join(f'{n} 0 R' for n in kids)}] /Count {len(kids)} >>".encode("ascii"))
        w.obj(3, b"<< /Title " + _pdf_text(title) + b" /Producer " + _pdf_text(PRODUCER)
              + time.strftime(" /CreationDate (D:%Y%m%d%H%M%S) >>").encode("ascii"))
        w.finish(root=1, info=3, size=4 + 3 * total)


# ---- 写出：epub ------------------------------------------------------------------------------------
_CONTAINER_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    '  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>\n'
    '</container>\n'
).encode("utf-8")


def _write_mimetype(zf: zipfile.ZipFile) -> None:
    """EPUB 规定 mimetype 必须是第一个条目、不压缩。"""
    zf.writestr(_zinfo("mimetype", zipfile.ZIP_STORED), _EPUB_MIMETYPE)


def _page_xhtml(img_href: str, w: int, h: int, i: int, lang: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}" lang="{lang}">\n'
        f'<head><meta charset="UTF-8"/><meta name="viewport" content="width={w}, height={h}"/><title>{i}</title>\n'
        f'<style>html,body{{margin:0;padding:0;width:{w}px;height:{h}px;overflow:hidden}}'
        f'img{{display:block;width:{w}px;height:{h}px}}</style></head>\n'
        f'<body><img src="../{img_href}" alt="{i}"/></body>\n</html>\n'
    ).encode("utf-8")


def _write_new_epub(book: Book, items: list[tuple[str, str | None]], part: str, o: dict, progress: Progress | None,
                    title: str, rtl: bool, lang: str) -> None:
    total = len(items)
    width = max(4, len(str(total)))
    manifest: list[str] = []
    spine: list[str] = []
    first_page = ""
    with zipfile.ZipFile(part, "w", allowZip64=True) as zf:
        _write_mimetype(zf)
        zf.writestr(_zinfo("META-INF/container.xml", zipfile.ZIP_DEFLATED), _CONTAINER_XML)
        for i, (name, src) in enumerate(items, 1):
            data, _, fam = _page_bytes(book, name, src, o)
            w, h = _dims(data)
            img = f"images/{i:0{width}d}.{fam}"
            page = f"xhtml/p{i:0{width}d}.xhtml"
            first_page = first_page or page
            zf.writestr(_zinfo("OEBPS/" + img, zipfile.ZIP_STORED), data)
            zf.writestr(_zinfo("OEBPS/" + page, zipfile.ZIP_DEFLATED), _page_xhtml(img, w, h, i, lang))
            cover = ' properties="cover-image"' if i == 1 else ""
            manifest.append(f'    <item id="img{i}" href="{img}" media-type="{_MEDIA[fam]}"{cover}/>')
            manifest.append(f'    <item id="p{i}" href="{page}" media-type="application/xhtml+xml"/>')
            spine.append(f'    <itemref idref="p{i}"/>')
            _tick(progress, i, total)
        t = _esc(title)
        nav = (
            '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n'
            f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}" lang="{lang}">\n'
            f'<head><meta charset="UTF-8"/><title>{t}</title></head>\n'
            f'<body><nav epub:type="toc" id="toc"><ol><li><a href="{first_page}">{t}</a></li></ol></nav></body>\n</html>\n'
        )
        modified = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        opf = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="{lang}" '
            'prefix="rendition: http://www.idpf.org/vocab/rendition/#">\n'
            '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
            f'    <dc:identifier id="bookid">urn:uuid:{uuid.uuid4()}</dc:identifier>\n'
            f'    <dc:title>{t}</dc:title>\n    <dc:language>{lang}</dc:language>\n'
            f'    <meta property="dcterms:modified">{modified}</meta>\n'
            '    <meta property="rendition:layout">pre-paginated</meta>\n'
            '    <meta property="rendition:spread">none</meta>\n'
            '    <meta name="cover" content="img1"/>\n'
            '  </metadata>\n  <manifest>\n'
            '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>\n'
            + "\n".join(manifest) + "\n  </manifest>\n"
            f'  <spine page-progression-direction="{"rtl" if rtl else "ltr"}">\n' + "\n".join(spine) + "\n  </spine>\n"
            "</package>\n"
        )
        zf.writestr(_zinfo("OEBPS/nav.xhtml", zipfile.ZIP_DEFLATED), nav.encode("utf-8"))
        zf.writestr(_zinfo("OEBPS/content.opf", zipfile.ZIP_DEFLATED), opf.encode("utf-8"))


def _patch_opf(data: bytes, suffix: str, lang: str) -> bytes:
    """原 epub 的 OPF：dc:title 加后缀、dc:language 改成中文；改不了就原样返回。"""
    try:
        text = decode_text(data)

        def title_sub(m: re.Match) -> str:
            old = html.unescape(m.group(2)).strip()
            new = old if (not suffix or not old or old.endswith(suffix)) else old + suffix
            return f"{m.group(1)}{_esc(new)}{m.group(3)}"

        text = re.sub(r"(<\s*dc:title\b[^>]*>)(.*?)(</\s*dc:title\s*>)", title_sub, text, count=1, flags=re.I | re.S)
        text = re.sub(r"(<\s*dc:language\b[^>]*>)(.*?)(</\s*dc:language\s*>)", rf"\g<1>{lang}\g<3>", text,
                      count=1, flags=re.I | re.S)
        text = re.sub(r"""^(\s*<\?xml[^>]*?encoding\s*=\s*["'])[^"']*(["'])""", r"\1utf-8\2", text, count=1)
        return text.encode("utf-8")
    except Exception:  # noqa: BLE001
        log.warning("OPF 元数据修改失败，原样保留", exc_info=True)
        return data


def _write_epub_copy(book: EpubBook, items: list[tuple[str, str | None]], part: str, o: dict,
                     progress: Progress | None, suffix: str, lang: str) -> None:
    page_set = set(book.pages)
    repl = {name: src for name, src in items if name in page_set}
    lost = [name for name, _ in items if name not in repl]
    if lost:
        log.warning("这些页面不在原 epub 里，导出 epub 时忽略：%s", lost[:5])
    total = len(repl)
    done = 0
    with zipfile.ZipFile(part, "w", allowZip64=True) as zf:
        _write_mimetype(zf)
        for name in book.entries:
            if name == "mimetype":
                continue
            if name in repl:
                fam = _FAMILY_OF_EXT.get(os.path.splitext(name)[1].lower(), "png")
                data, _, _ = _page_bytes(book, name, repl[name], o, keep_family=fam)
                zf.writestr(_zinfo(name, zipfile.ZIP_STORED), data)
                done += 1
                _tick(progress, done, total)
                continue
            try:
                data = book.read_entry(name)
            except HanhuaError as e:
                log.warning("epub 条目读不了，已跳过：%s（%s）", name, e)
                continue
            if name == book.opf_path:
                data = _patch_opf(data, suffix, lang)
            compress = zipfile.ZIP_STORED if book.entry_compress_type(name) == zipfile.ZIP_STORED else zipfile.ZIP_DEFLATED
            zf.writestr(_zinfo(name, compress), data)


# ---- 入口 ------------------------------------------------------------------------------------------
def _remove(path: str) -> None:
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.lexists(path):
            os.remove(path)
    except OSError as e:
        log.warning("清理临时输出失败：%s（%s）", path, e)


def _finalize(part: str, final: str, is_dir: bool) -> None:
    if is_dir:
        if os.path.isdir(final) and not os.path.islink(final):
            old = f"{final}.old-{int(time.time() * 1000)}"
            os.replace(final, old)                    # 覆盖上一次的导出：先挪开旧的，新的就位后再删
            try:
                os.replace(part, final)
            except OSError:
                os.replace(old, final)
                raise
            shutil.rmtree(old, ignore_errors=True)
            return
        if os.path.lexists(final):
            raise HanhuaError(f"输出位置已经有一个同名文件，无法写成文件夹：{final}")
    elif os.path.isdir(final):
        raise HanhuaError(f"输出位置已经有一个同名文件夹，无法写成文件：{final}")
    os.replace(part, final)


def export_book(book: Book, pages: list[tuple[str, str]], out_path: str, settings: dict,
                progress: Callable[[int, int], None] | None = None) -> str:
    """pages = [(页面名, 渲染好的图片文件路径)]；按 settings["output"] 写出；先写 .part 再改名；返回最终路径。

    渲染结果缺失的页面用原图代替（记警告）；中途异常（含 progress 回调抛出的取消）会删掉 .part 再往外抛。
    """
    o = _out(settings)
    kind = output_kind(book, settings)
    if not isinstance(out_path, (str, os.PathLike)) or not os.fspath(out_path).strip():
        raise HanhuaError("没有指定输出位置。")
    final = os.path.abspath(os.fspath(out_path))
    if os.path.normcase(final) == os.path.normcase(os.path.abspath(book.path)):
        raise HanhuaError("输出位置不能和源文件相同，请换一个文件名或输出文件夹。")
    items = _items(pages)
    if not items:
        raise HanhuaError("没有可以导出的页面。")
    suffix = _suffix(settings)
    title = f"{book.title}{suffix}"
    rtl = _is_rtl(settings)
    part = final + ".part"
    t0 = time.perf_counter()
    try:
        os.makedirs(os.path.dirname(final) or ".", exist_ok=True)
        _remove(part)                                 # 上次中断留下的半成品
        _tick(progress, 0, len(items))
        if kind == "folder":
            _write_folder(book, items, part, o, progress, book.title, suffix)
        elif kind in ("zip", "cbz"):
            _write_zip(book, items, part, o, progress, book.title, suffix, kind == "cbz", rtl)
        elif kind == "pdf":
            _write_pdf(book, items, part, o, progress, title, rtl)
        elif isinstance(book, EpubBook):
            _write_epub_copy(book, items, part, o, progress, suffix, _target_lang(settings))
        else:
            _write_new_epub(book, items, part, o, progress, title, rtl, _target_lang(settings))
        _finalize(part, final, kind == "folder")
    except BaseException as e:
        _remove(part)
        if isinstance(e, OSError):
            raise HanhuaError(f"写出失败：{final}（{e.strerror or e}）。请检查磁盘空间和输出文件夹的权限。") from e
        raise
    log.info("已导出 %s：%s（%d 页，%.1fs）", kind, final, len(items), time.perf_counter() - t0)
    return final


def iter_output_kinds() -> Iterable[str]:
    """界面下拉框用：所有可选输出格式（含 same）。"""
    return ("same",) + OUTPUT_KINDS

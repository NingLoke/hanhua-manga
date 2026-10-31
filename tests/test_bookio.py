"""bookio 测试：用合成小图造各种格式（中文文件名 / 目录），逐一 读 → 导出 → 再读，页数 / 顺序 / 尺寸一致。

每页尺寸都不同：顺序错了尺寸就对不上。素材写在 tests/out/bookio/<测试名>_测试目录/ 下，方便出问题时人工查看。
"""
from __future__ import annotations

import io
import os
import re
import shutil
import struct
import subprocess
import tarfile
import threading
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from hanhua import imageio as hio
from hanhua.bookio import Book, default_output_path, export_book, open_book, scan_inputs
from hanhua.bookio.archive import find_tar
from hanhua.bookio.epub import EpubBook
from hanhua.bookio.pdf import PdfBook
from hanhua.types import Cancelled, HanhuaError, UnsupportedFormat

_CJK = re.compile(r"[一-鿿]")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
needs_tar = pytest.mark.skipif(find_tar() is None, reason="系统没有 tar.exe（bsdtar）")


# ---- 素材 ------------------------------------------------------------------------------------------
def _img(i: int, w: int, h: int) -> np.ndarray:
    """第 i 页：底色各不相同 + 黑块，尺寸 w×h。"""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = ((40 * i + 30) % 256, (90 * i + 60) % 256, (150 + 20 * i) % 256)
    img[h // 4: h // 2, w // 4: 3 * w // 4] = 0
    return img


def _size(i: int) -> tuple[int, int]:
    return 60 + 8 * i, 90 + 6 * i


def _png(i: int) -> bytes:
    return hio.encode(_img(i, *_size(i)), "png")


def _settings(fmt: str = "same", image_format: str = "same", **out: object) -> dict:
    o = {"format": fmt, "image_format": image_format, "jpeg_quality": 92, "webp_quality": 90,
         "dir": "", "suffix": "_汉化"}
    o.update(out)
    return {"output": o, "source": {"lang": "ja", "reading_order": "auto"}, "translate": {"target": "zh-Hans"}}


@pytest.fixture
def work(out_dir: Path, request: pytest.FixtureRequest) -> Path:
    d = out_dir / f"{re.sub(r'[^0-9A-Za-z_]+', '_', request.node.name)}_测试目录"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def _tmp(work: Path) -> str:
    return str(work / "tmp 临时")


def _shapes(book: Book) -> list[tuple[int, int]]:
    return [tuple(book.read_page(n).shape[:2]) for n in book.pages]


def _render(book: Book, root: Path) -> list[tuple[str, str]]:
    """模拟引擎：每页反色后存 PNG，返回 [(页面名, 渲染结果路径)]。坏页跳过（真实引擎会标 error、导出时用原图）。"""
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i, name in enumerate(book.pages, 1):
        try:
            img = book.read_page(name)
        except UnsupportedFormat:
            continue
        p = root / f"{i:04d}.png"
        hio.write(str(p), 255 - img)
        out.append((name, str(p)))
    return out


def _roundtrip(book: Book, work: Path, fmt: str = "same", image_format: str = "same") -> Book:
    """渲染 → 导出 → 再读：页数 / 顺序（尺寸）一致、内容是渲染结果、进度回调完整、不留 .part。"""
    rendered = _render(book, work / f"rendered_{fmt}_{image_format}")
    settings = _settings(fmt, image_format)
    out = default_output_path(book, settings)
    calls: list[tuple[int, int]] = []
    final = export_book(book, rendered, out, settings, progress=lambda d, t: calls.append((d, t)))
    assert final == out and os.path.exists(final) and not os.path.exists(final + ".part")
    assert calls[0] == (0, len(book.pages)) and calls[-1] == (len(book.pages), len(book.pages))
    again = open_book(final, _tmp(work))
    assert len(again.pages) == len(book.pages)
    assert _shapes(again) == _shapes(book)
    for (name, src), new_name in zip(rendered, again.pages):
        diff = np.abs(again.read_page(new_name).astype(int) - hio.read(src).astype(int)).mean()
        assert diff < 6, f"{new_name} 内容不是渲染结果（平均差 {diff:.1f}）"
    return again


def _bsdtar_create(src: Path, out: Path, fmt: str) -> None:
    """用 tar.exe 造压缩包：cwd = 源目录、输出走 stdout（命令行里不出现中文路径）。"""
    with open(out, "wb") as fo:
        r = subprocess.run([find_tar(), "-c", "--format", fmt, "-f", "-", "."], cwd=src, stdout=fo,
                           stderr=subprocess.PIPE, timeout=60, creationflags=_NO_WINDOW)
    assert r.returncode == 0, r.stderr


def _rar4(files: list[tuple[str, bytes]]) -> bytes:
    """手工造 RAR 4.x 存储（不压缩）包；非 ASCII 名字 = UTF-8 + FHD_UNICODE 标志（WinRAR 3.x+ 的写法之一）。"""
    out = bytearray(b"Rar!\x1a\x07\x00")
    hdr = struct.pack("<BHH", 0x73, 0, 13) + b"\0" * 6
    out += struct.pack("<H", zlib.crc32(hdr) & 0xFFFF) + hdr
    for name, data in files:
        nb = name.encode("utf-8")
        flags = 0x8000 | (0x200 if not name.isascii() else 0)
        body = struct.pack("<BHH", 0x74, flags, 32 + len(nb))
        body += struct.pack("<IIBIIBBHI", len(data), len(data), 2, zlib.crc32(data) & 0xFFFFFFFF,
                            0x5A210000, 29, 0x30, len(nb), 0x20) + nb
        out += struct.pack("<H", zlib.crc32(body) & 0xFFFF) + body + data
    end = struct.pack("<BHH", 0x7B, 0x4000, 7)
    out += struct.pack("<H", zlib.crc32(end) & 0xFFFF) + end
    return bytes(out)


class _LegacyInfo(zipfile.ZipInfo):
    """模拟日文 Windows 的老压缩软件：文件名按 cp932 存、不设 UTF-8 标志。"""

    def _encodeFilenameFlags(self):  # noqa: N802 - 覆盖 zipfile 私有方法
        return self.filename.encode("cp932"), self.flag_bits & ~0x800


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, d in entries:
            zf.writestr(n, d)
    return buf.getvalue()


# ---- 文件夹 / 散图 ---------------------------------------------------------------------------------
def test_folder_book_sort_junk_comicinfo(work: Path) -> None:
    d = work / "第1话 中文"
    d.mkdir()
    for i, n in ((1, "1.png"), (2, "2.png"), (3, "10.png")):
        (d / n).write_bytes(_png(i))
    (d / "Thumbs.db").write_bytes(b"junk")
    (d / ".hidden.png").write_bytes(_png(9))
    (d / "__MACOSX").mkdir()
    (d / "__MACOSX" / "x.png").write_bytes(_png(9))
    (d / "ComicInfo.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<ComicInfo>\n  <Title>原标题</Title>\n'
        "  <LanguageISO>ja</LanguageISO>\n</ComicInfo>\n", encoding="utf-8")
    book = open_book(str(d), _tmp(work))
    assert book.kind == "folder" and book.title == "第1话 中文"
    assert book.pages == ["1.png", "2.png", "10.png"] and book.extras == ["ComicInfo.xml"]

    out_folder = _roundtrip(book, work, "same")                  # 文件夹 → 文件夹
    assert os.path.basename(out_folder.path) == "第1话 中文_汉化"
    ci = out_folder.read_extra("ComicInfo.xml").decode("utf-8")
    assert "<Title>原标题_汉化</Title>" in ci and "<LanguageISO>zh</LanguageISO>" in ci

    cbz = _roundtrip(book, work, "cbz")                          # 文件夹 → cbz
    with zipfile.ZipFile(cbz.path) as zf:
        infos = {i.filename: i for i in zf.infolist()}
        assert all(infos[n].compress_type == zipfile.ZIP_STORED for n in book.pages)
        ci = zf.read("ComicInfo.xml").decode("utf-8")
    assert ci.count("<Title>") == 1 and "原标题_汉化" in ci and "<LanguageISO>zh</LanguageISO>" in ci
    cbz.close()
    # 重名加 (2)：第一次导出的文件夹已经存在
    assert default_output_path(book, _settings("folder")).endswith("第1话 中文_汉化 (2)")


def test_loose_images_one_book_and_single(work: Path) -> None:
    d = work / "散图 目录"
    d.mkdir()
    (d / "a 10.webp").write_bytes(hio.encode(_img(3, *_size(3)), "webp", 95))
    (d / "a 2.jpg").write_bytes(hio.encode(_img(2, *_size(2)), "jpg", 95))
    (d / "a 1.png").write_bytes(_png(1))
    files = [str(d / n) for n in ("a 10.webp", "a 1.png", "a 2.jpg")]
    found = scan_inputs(files, _tmp(work))
    assert len(found) == 1 and found[0].endswith(".hanhua-images.json")
    assert scan_inputs(files, _tmp(work)) == found                 # 同一批图 → 同一个清单（可续跑）
    book = open_book(found[0], _tmp(work))
    assert book.kind == "images" and book.pages == ["a 1.png", "a 2.jpg", "a 10.webp"]
    again = _roundtrip(book, work, "same")
    assert os.path.dirname(again.path) == str(d)                  # 散图默认输出在图片所在目录
    assert again.pages == ["a 1.png", "a 2.jpg", "a 10.webp"]     # image_format=same 沿用原扩展名

    single = scan_inputs([str(d / "a 1.png")], _tmp(work))
    assert single == [str(d / "a 1.png")]
    one = open_book(single[0], _tmp(work))
    assert one.pages == ["a 1.png"] and one.title == "a 1"


def test_image_format_mapping(work: Path) -> None:
    d = work / "格式"
    d.mkdir()
    (d / "1.gif").write_bytes(b"")
    hio.to_pil(_img(1, *_size(1))).save(str(d / "1.gif"), "GIF")
    (d / "2.bmp").write_bytes(hio.encode(_img(2, *_size(2)), "bmp"))
    (d / "3.jpeg").write_bytes(hio.encode(_img(3, *_size(3)), "jpg"))
    book = open_book(str(d), _tmp(work))
    same = _roundtrip(book, work, "same")
    assert same.pages == ["1.png", "2.png", "3.jpeg"]              # gif / bmp → png，jpeg 保留原写法
    as_jpg = _roundtrip(book, work, "zip", "jpg")
    assert as_jpg.pages == ["1.jpg", "2.jpg", "3.jpeg"]
    assert all(as_jpg.read_page_bytes(n)[:3] == b"\xff\xd8\xff" for n in as_jpg.pages)
    as_jpg.close()


# ---- zip / 套娃 ------------------------------------------------------------------------------------
def test_zip_shift_jis_names(work: Path) -> None:
    path = work / "日文 老压缩包.zip"
    names = ["第1話/本文 10.png", "第1話/表紙.png", "第1話/本文 2.png"]
    with zipfile.ZipFile(path, "w") as zf:
        for i, n in enumerate(names, 1):
            zf.writestr(_LegacyInfo(n, date_time=(2020, 1, 1, 0, 0, 0)), _png(i))
        zf.writestr(_LegacyInfo("__MACOSX/第1話/._表紙.png", date_time=(2020, 1, 1, 0, 0, 0)), b"junk")
    with zipfile.ZipFile(path) as zf:                              # 不修复时 zipfile 读出来是乱码
        assert not any("第1話" in n for n in zf.namelist())
    book = open_book(str(path), _tmp(work))
    assert book.kind == "zip"
    assert book.pages == ["第1話/本文 2.png", "第1話/本文 10.png", "第1話/表紙.png"]
    again = _roundtrip(book, work, "same")
    assert again.kind == "zip" and again.pages == book.pages      # 输出用 UTF-8 文件名
    again.close()
    folder = _roundtrip(book, work, "folder")                     # 文件夹输出保持相对路径
    assert os.path.isfile(os.path.join(folder.path, "第1話", "本文 2.png"))
    book.close()


def test_zip_concurrent_reads(work: Path) -> None:
    path = work / "并发.cbz"
    path.write_bytes(_zip_bytes([(f"{i:03d}.png", _png(i)) for i in range(1, 9)]))
    book = open_book(str(path), _tmp(work))
    expect = {n: _png(i) for i, n in enumerate(book.pages, 1)}
    errors: list[str] = []

    def worker() -> None:
        for _ in range(5):
            for n in book.pages:
                if book.read_page_bytes(n) != expect[n]:
                    errors.append(n)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    book.close()


def test_nested_zip_volumes(work: Path) -> None:
    src = work / "源"
    src.mkdir()
    outer = src / "合集 全2卷.zip"
    vol1 = _zip_bytes([("001.png", _png(1)), ("002.png", _png(2))])
    vol2 = _zip_bytes([("001.png", _png(3)), ("002.png", _png(4)), ("003.png", _png(5))])
    outer.write_bytes(_zip_bytes([("Vol.02.cbz", vol2), ("01.cbz", vol1), ("说明.txt", b"hi")]))
    found = scan_inputs([str(outer)], _tmp(work))
    assert [os.path.basename(p) for p in found] == ["01.cbz", "Vol.02.cbz"]
    assert scan_inputs([str(outer)], _tmp(work)) == found          # 重复扫描不重复展开
    books = [open_book(p, _tmp(work)) for p in found]
    assert [len(b.pages) for b in books] == [2, 3]
    assert [b.title for b in books] == ["合集 全2卷 01", "合集 全2卷 Vol.02"]   # 卷名太通用 → 加外层标题
    assert all(b.origin_dir == str(src) for b in books)            # 默认输出在最外层压缩包旁边
    out = default_output_path(books[0], _settings())
    assert out == os.path.join(str(src), "合集 全2卷 01_汉化.cbz")
    again = _roundtrip(books[0], work, "same")
    assert again.kind == "cbz"
    for b in books + [again]:
        b.close()


# ---- 7z / tar / rar（tar.exe） ---------------------------------------------------------------------
@needs_tar
def test_7z_made_by_tar_exe(work: Path) -> None:
    src = work / "源 文件夹"
    (src / "子目录").mkdir(parents=True)
    for i, n in ((1, "第01页.png"), (2, "第02页.png"), (3, "子目录/第10页.png")):
        (src / n).write_bytes(_png(i))
    (src / "ComicInfo.xml").write_text("<ComicInfo><Title>七z</Title></ComicInfo>", encoding="utf-8")
    path = work / "测试 七z卷.7z"
    _bsdtar_create(src, path, "7zip")
    book = open_book(str(path), _tmp(work))
    assert book.kind == "7z"
    assert book.pages == ["第01页.png", "第02页.png", "子目录/第10页.png"] and book.extras == ["ComicInfo.xml"]
    again_open = open_book(str(path), _tmp(work))                 # 第二次打开复用解压结果
    assert again_open._root == book._root
    again = _roundtrip(book, work, "same")
    assert again.kind == "cbz" and again.path.endswith("测试 七z卷_汉化.cbz")    # 7z → cbz
    with zipfile.ZipFile(again.path) as zf:
        ci = zf.read("ComicInfo.xml").decode("utf-8")
    assert "七z_汉化" in ci and "<LanguageISO>zh</LanguageISO>" in ci
    again.close()


@pytest.mark.parametrize("fname,mode,fmt", [
    ("gnu 格式.cbt", "w", tarfile.GNU_FORMAT),
    ("pax 格式.tar.gz", "w:gz", tarfile.PAX_FORMAT),
])
def test_tar_family(work: Path, fname: str, mode: str, fmt: int) -> None:
    path = work / fname
    with tarfile.open(path, mode, format=fmt, encoding="utf-8") as tf:
        for i, n in ((2, "第1话/02.png"), (1, "第1话/01.png"), (3, "第1话/10.png")):
            data = _png(i)
            ti = tarfile.TarInfo(n)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        evil = tarfile.TarInfo("../../逃出去.png")               # 路径穿越：只能落在解压目录里
        evil.size = 3
        tf.addfile(evil, io.BytesIO(b"bad"))
    book = open_book(str(path), _tmp(work))
    assert book.kind in ("cbt", "tar")
    # 穿越条目被 bsdtar 去掉 ../ 后落在解压根目录（根目录文件排在子目录前），只核对第1话的自然顺序
    assert [n for n in book.pages if n.startswith("第1话/")] == ["第1话/01.png", "第1话/02.png", "第1话/10.png"]
    assert not (work / "逃出去.png").exists() and not (work.parent / "逃出去.png").exists()
    pages = [n for n in book.pages if n.startswith("第1话/")]
    rendered = _render(book, work / "r")
    final = export_book(book, [r for r in rendered if r[0] in pages], default_output_path(book, _settings()),
                        _settings())
    again = open_book(final, _tmp(work))
    assert again.pages == pages and _shapes(again) == [tuple(book.read_page(n).shape[:2]) for n in pages]
    again.close()


@needs_tar
def test_rar_extract_and_broken(work: Path) -> None:
    path = work / "手工 RAR4.cbr"
    path.write_bytes(_rar4([("第1话/002.png", _png(2)), ("第1话/001.png", _png(1)), ("ComicInfo.xml", b"<ComicInfo/>")]))
    book = open_book(str(path), _tmp(work))
    assert book.kind == "cbr" and book.pages == ["第1话/001.png", "第1话/002.png"]
    assert book.extras == ["ComicInfo.xml"]
    again = _roundtrip(book, work, "same")
    assert again.kind == "cbz"
    again.close()

    bad = work / "坏掉的.cbr"
    bad.write_bytes(b"Rar!\x1a\x07\x00" + os.urandom(300))
    with pytest.raises(UnsupportedFormat) as ei:
        open_book(str(bad), _tmp(work))
    msg = str(ei.value)
    assert "坏掉的.cbr" in msg and ("损坏" in msg or "无法识别" in msg)
    leftovers = [n for n in os.listdir(_tmp(work)) if n.endswith(".part")]
    assert not leftovers                                           # 失败不留半个解压目录


# ---- PDF -------------------------------------------------------------------------------------------
def _encrypted_pdf() -> bytes:
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >>",
        b"<< /Filter /Standard /V 1 /R 2 /O <" + b"11" * 32 + b"> /U <" + b"22" * 32 + b"> /P -44 >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, body in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += (b"trailer\n<< /Size %d /Root 1 0 R /Encrypt 4 0 R /ID [<%s><%s>] >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, b"ab" * 16, b"ab" * 16, xref))
    return bytes(out)


def test_pdf_roundtrip_and_errors(work: Path) -> None:
    d = work / "PDF 源"
    d.mkdir()
    for i in range(1, 4):
        (d / f"{i:02d}.png").write_bytes(_png(i))
    folder = open_book(str(d), _tmp(work))
    pdf = _roundtrip(folder, work, "pdf")                          # 文件夹 → pdf：单图页按原始像素渲染
    assert isinstance(pdf, PdfBook) and pdf.pages == ["0001.png", "0002.png", "0003.png"]
    cache = os.path.join(pdf._cache, "0001.png")
    assert os.path.isfile(cache)                                   # PNG 缓存
    assert pdf.read_page_bytes("0001.png") == open(cache, "rb").read()
    again = _roundtrip(pdf, work, "same")                          # pdf → pdf
    assert again.path.endswith("PDF 源_汉化_汉化.pdf")
    pdf.close()
    again.close()

    import pypdfium2 as pdfium                                     # 矢量页（无图）：长边约 2000px
    vec = work / "矢量页.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 300)
    doc.save(str(vec))
    doc.close()
    vb = open_book(str(vec), _tmp(work))
    assert vb.read_page(vb.pages[0]).shape[:2] == (2000, 1334)
    vb.close()

    enc = work / "有密码.pdf"
    enc.write_bytes(_encrypted_pdf())
    with pytest.raises(UnsupportedFormat, match="密码"):
        open_book(str(enc), _tmp(work))
    broken = work / "损坏的.pdf"
    broken.write_bytes(b"%PDF-1.4\n" + os.urandom(200))
    with pytest.raises(UnsupportedFormat, match="损坏"):
        open_book(str(broken), _tmp(work))


def test_real_pages_jpeg_passthrough_pdf(work: Path, real_pages: list) -> None:
    pages = real_pages[:3]
    path = work / "真实页面.cbz"
    path.write_bytes(_zip_bytes([(p.name, p.read_bytes()) for p in pages]))
    book = open_book(str(path), _tmp(work))
    out = default_output_path(book, _settings("pdf"))
    final = export_book(book, [(p.name, str(p)) for p in pages], out, _settings("pdf"))
    blob = open(final, "rb").read()
    assert all(p.read_bytes() in blob for p in pages)              # JPEG 原样嵌入，不重新编码
    again = open_book(final, _tmp(work))
    assert _shapes(again) == _shapes(book)
    again.close()
    book.close()


# ---- EPUB ------------------------------------------------------------------------------------------
def _make_epub(path: Path) -> None:
    opf = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">x</dc:identifier><dc:title>测试书</dc:title><dc:language>ja</dc:language>
  </metadata>
  <manifest>
    <item id="p1" href="Text/p1.xhtml" media-type="application/xhtml+xml"/>
    <item id="p2" href="Text/p2.xhtml" media-type="application/xhtml+xml"/>
    <item id="p3" href="Text/p3.xhtml" media-type="application/xhtml+xml"/>
    <item id="a" href="Images/a.png" media-type="image/png"/>
    <item id="b" href="Images/b%20b.jpg" media-type="image/jpeg"/>
    <item id="c" href="Images/c.png" media-type="image/png"/>
    <item id="u" href="Images/unused.png" media-type="image/png"/>
    <item id="css" href="Styles/s.css" media-type="text/css"/>
  </manifest>
  <spine page-progression-direction="rtl"><itemref idref="p1"/><itemref idref="p2"/><itemref idref="p3"/></spine>
</package>"""
    page = ('<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml" '
            'xmlns:xlink="http://www.w3.org/1999/xlink"><head><link href="../Styles/s.css" rel="stylesheet"/>'
            "</head><body>{}</body></html>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), b"application/epub+zip")
        zf.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" '
                    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                    '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                    "</rootfiles></container>")
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/Text/p1.xhtml", page.format('<img src="../Images/c.png" alt=""/>'))
        zf.writestr("OEBPS/Text/p2.xhtml", page.format(
            '<!-- <img src="../Images/unused.png"/> --><img src="../Images/a.png"/><img src="../Images/c.png"/>'))
        zf.writestr("OEBPS/Text/p3.xhtml", page.format(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><image width="10" height="10" '
            'xlink:href="../Images/b%20b.jpg"/></svg>'))
        zf.writestr("OEBPS/Styles/s.css", "img{max-width:100%}")
        zf.writestr("OEBPS/Images/c.png", _png(1))
        zf.writestr("OEBPS/Images/a.png", _png(2))
        zf.writestr("OEBPS/Images/b b.jpg", hio.encode(_img(3, *_size(3)), "jpg", 95))
        zf.writestr("OEBPS/Images/unused.png", _png(4))


def test_epub_spine_order_copy_and_new(work: Path) -> None:
    path = work / "测试 电子书.epub"
    _make_epub(path)
    book = open_book(str(path), _tmp(work))
    assert isinstance(book, EpubBook)
    assert book.pages == ["OEBPS/Images/c.png", "OEBPS/Images/a.png", "OEBPS/Images/b b.jpg"]   # spine 顺序 + 去重
    assert "OEBPS/Images/unused.png" in book.extras and "OEBPS/Text/p1.xhtml" in book.extras

    again = _roundtrip(book, work, "same")                         # epub → epub：复制原包只换图
    assert isinstance(again, EpubBook) and again.pages == book.pages
    with zipfile.ZipFile(again.path) as zf:
        first = zf.infolist()[0]
        assert first.filename == "mimetype" and first.compress_type == zipfile.ZIP_STORED
        assert zf.read("OEBPS/Text/p2.xhtml") == book.read_extra("OEBPS/Text/p2.xhtml")
        assert zf.read("OEBPS/Images/unused.png") == _png(4)
        opf = zf.read("OEBPS/content.opf").decode("utf-8")
    assert "<dc:title>测试书_汉化</dc:title>" in opf and "<dc:language>zh-Hans</dc:language>" in opf
    again.close()

    cbz = _roundtrip(book, work, "cbz")                            # epub → cbz：只要页面（+ ComicInfo）
    with zipfile.ZipFile(cbz.path) as zf:
        assert sorted(zf.namelist()) == sorted(cbz.pages + ["ComicInfo.xml"])
    cbz.close()

    d = work / "新建 epub 源"
    d.mkdir()
    for i in range(1, 4):
        (d / f"{i}.png").write_bytes(_png(i))
    new = _roundtrip(open_book(str(d), _tmp(work)), work, "epub")  # 文件夹 → 新建固定版式 epub
    assert isinstance(new, EpubBook) and new.page_progression == "rtl"
    new.close()
    book.close()


# ---- 导出健壮性 ------------------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["folder", "cbz", "pdf", "epub"])
def test_export_cancel_leaves_nothing(work: Path, fmt: str) -> None:
    d = work / "取消 源"
    d.mkdir()
    for i in range(1, 5):
        (d / f"{i}.png").write_bytes(_png(i))
    book = open_book(str(d), _tmp(work))
    rendered = _render(book, work / "r")
    out = default_output_path(book, _settings(fmt))

    def cancel(done: int, total: int) -> None:
        if done == 2:
            raise Cancelled("用户取消")

    with pytest.raises(Cancelled):
        export_book(book, rendered, out, _settings(fmt), progress=cancel)
    assert not os.path.exists(out) and not os.path.exists(out + ".part")
    # 渲染结果缺失 / 路径为空的页面用原图代替，导出照常完成
    rendered[1] = (rendered[1][0], str(work / "不存在.png"))
    rendered[2] = (rendered[2][0], None)
    final = export_book(book, rendered, out, _settings(fmt))
    again = open_book(final, _tmp(work))
    assert len(again.pages) == 4 and _shapes(again) == _shapes(book)
    orig = book.read_page(book.pages[1]).astype(int)
    assert np.abs(again.read_page(again.pages[1]).astype(int) - orig).mean() < 6
    again.close()


def test_default_output_path_rules(work: Path) -> None:
    fake = Book(str(work / "某书.7z"), "7z", "某书", str(work))
    assert default_output_path(fake, _settings()) == str(work / "某书_汉化.cbz")       # 7z → cbz
    (work / "某书_汉化.cbz").write_bytes(b"x")
    (work / "某书_汉化 (2).cbz").write_bytes(b"x")
    assert default_output_path(fake, _settings()) == str(work / "某书_汉化 (3).cbz")
    other = work / "输出 目录"
    assert default_output_path(fake, _settings("pdf", dir=str(other))) == str(other / "某书_汉化.pdf")
    assert default_output_path(fake, _settings("folder", suffix="")) == str(work / "某书")
    for bad in (None, {}, {"output": None}, {"output": {"format": "???", "suffix": None}}):
        assert default_output_path(fake, bad).endswith(".cbz")      # 设置缺失 / 损坏也不崩
    weird = Book(str(work / "x.zip"), "zip", 'a<b>:c"?*|', str(work))
    # 标题 a<b>:c"?*| 净化成 a_b__c____（4 个），再接后缀「_汉化」自带的 1 个下划线
    assert os.path.basename(default_output_path(weird, _settings())) == "a_b__c_____汉化.zip"


def test_bad_inputs_never_crash(work: Path) -> None:
    (work / "空文件夹").mkdir()
    (work / "坏.zip").write_bytes(b"PK\x03\x04" + os.urandom(100))
    (work / "只有文字.zip").write_bytes(_zip_bytes([("readme.txt", b"hello")]))
    (work / "怪.xyz").write_bytes(b"???")
    (work / "空.epub").write_bytes(b"")
    cases = ["", "   ", str(work / "不存在.cbz"), str(work / "空文件夹"), str(work / "坏.zip"),
             str(work / "只有文字.zip"), str(work / "怪.xyz"), str(work / "空.epub")]
    for c in cases:
        with pytest.raises(UnsupportedFormat) as ei:
            open_book(c, _tmp(work))
        assert _CJK.search(str(ei.value)), f"{c!r} 的错误信息不是中文：{ei.value}"
    with pytest.raises(UnsupportedFormat):
        open_book(None, _tmp(work))  # type: ignore[arg-type]
    found = scan_inputs([None, "", str(work / "不存在"), 123, str(work / "怪.xyz"), str(work)], _tmp(work))  # type: ignore[list-item]
    assert all(isinstance(p, str) for p in found)
    d = work / "导出源"
    d.mkdir()
    (d / "1.png").write_bytes(_png(1))
    book = open_book(str(d), _tmp(work))
    with pytest.raises(HanhuaError):
        export_book(book, [], str(work / "空.cbz"), _settings())
    with pytest.raises(HanhuaError):
        export_book(book, [("1.png", None)], str(d), _settings("folder"))   # 输出不能覆盖源
    assert (d / "1.png").exists()


def test_scan_folder_tree(work: Path) -> None:
    root = work / "系列 根目录"
    for ch in ("第1话", "第2话", "第10话"):
        (root / ch).mkdir(parents=True)
        (root / ch / "001.png").write_bytes(_png(1))
    (root / "第2话" / "番外").mkdir()
    (root / "第2话" / "番外" / "001.png").write_bytes(_png(2))
    (root / "卷3.cbz").write_bytes(_zip_bytes([("001.png", _png(3))]))
    (root / "Thumbs.db").write_bytes(b"x")
    (root / "空目录").mkdir()
    found = scan_inputs([str(root)], _tmp(work))
    rel = [os.path.relpath(p, root).replace("\\", "/") for p in found]
    assert rel == ["卷3.cbz", "第1话", "第2话", "第2话/番外", "第10话"]
    assert [len(open_book(p, _tmp(work)).pages) for p in found] == [1, 1, 1, 1, 1]

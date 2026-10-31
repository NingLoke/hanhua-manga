"""压缩包类的书。

* zip / cbz：zipfile 按条目直接读（不全量解压），文件名乱码由 formats.zip_entries 修复；
  zipfile 读不了（中央目录损坏、Deflate64 / PPMd / zstd 等压缩方式）时退回「解压后按文件夹读」。
* 7z / cb7、rar / cbr、tar / cbt：解压到 tmp_dir/x_<源文件签名哈希>/ 后按文件夹读，重复打开直接复用。
  - tar 家族先用标准库 tarfile（自己控制文件名解码和路径安全），失败再交给 tar.exe；
  - 7z / rar 用 Windows 自带 tar.exe（bsdtar / libarchive），失败且装了 7-Zip 时用 7z.exe 兜底；
  - bsdtar 的 argv 走 ANSI 代码页，中文路径会变成 ?（已实测）：所以命令行里不出现任何路径——
    压缩包作为子进程 stdin 喂进去（Python 用宽字符 API 打开），解压目标用子进程 cwd（CreateProcessW 宽字符）。
* 套娃：压缩包里没有图片、只有一卷卷的 cbz / zip / rar / pdf …… → nested_volumes 展开到 tmp_root，每卷一本。
"""
from __future__ import annotations

import functools
import json
import locale
import logging
import os
import shutil
import subprocess
import tarfile
import threading
import time
import zipfile

from .. import imageio as hio
from ..types import UnsupportedFormat
from .formats import (
    EXTRACT_KINDS, MAX_META_BYTES, ORIGIN_MARKER, ZIP_KINDS, Book, ZipReader, ZipUnreadable, clean_entry_name,
    file_signature, find_origin, guess_legacy_encoding, is_book_name, is_image_name, is_junk_path, is_meta_name,
    iter_dir_files, natural_key, safe_relpath, short_hash, title_from_name, tmp_dir_or_default, zip_entries,
)

log = logging.getLogger("hanhua.bookio")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DONE_MARKER = ".hanhua_extracted.json"          # 以 . 开头：iter_dir_files 会当隐藏文件跳过
ZIP_METHODS_OK = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA})
TAR_KINDS = frozenset({"tar", "cbt"})
_CHUNK = 1 << 20

_KEY_LOCKS: dict[str, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()


def _key_lock(key: str) -> threading.Lock:
    """同一个解压目标同一时间只允许一个线程写（多线程同时打开同一本书时不互相踩）。"""
    with _KEY_LOCKS_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = _KEY_LOCKS[key] = threading.Lock()
        return lock


# ---- 外部解压程序 ----------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def find_tar() -> str | None:
    """Windows 10 1803+ 自带的 bsdtar。不用 PATH 里 Git 的 GNU tar：它不认 7z / rar。"""
    for root in filter(None, (os.environ.get("SystemRoot"), os.environ.get("windir"), r"C:\Windows")):
        for sub in ("System32", "Sysnative"):
            p = os.path.join(root, sub, "tar.exe")
            if os.path.isfile(p):
                return p
    return shutil.which("bsdtar")


@functools.lru_cache(maxsize=1)
def find_7z() -> str | None:
    """用户装了 7-Zip 时作为兜底（tar.exe 不支持的 RAR 特性、Deflate64 等）。"""
    cands = [shutil.which("7z"), shutil.which("7za")]
    for env in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if base:
            cands.append(os.path.join(base, "7-Zip", "7z.exe"))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def _timeout_for(path: str) -> float:
    """按文件大小给超时：至少 2 分钟，每 MB 再加 1.5 秒，最多 1 小时。"""
    try:
        mb = os.path.getsize(path) / (1 << 20)
    except OSError:
        mb = 0.0
    return float(min(3600.0, max(120.0, 60.0 + mb * 1.5)))


def _decode_output(data: bytes) -> str:
    try:
        return data.decode(locale.getpreferredencoding(False) or "utf-8", "replace")
    except LookupError:
        return data.decode("utf-8", "replace")


def _run(args: list[str], cwd: str, stdin_path: str | None, timeout: float) -> tuple[int, str, str]:
    """跑外部程序：(退出码, stdout, stderr)。stdin_path 给出时把该文件句柄直接作为子进程 stdin（可 seek）。"""
    fh = None
    try:
        fh = open(stdin_path, "rb") if stdin_path else None
        proc = subprocess.run(
            args, cwd=cwd, stdin=fh if fh is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as e:
        label = os.path.basename(stdin_path or args[-1])
        raise UnsupportedFormat(f"解压「{label}」超时（超过 {int(timeout)} 秒），文件可能太大或已损坏。") from e
    except OSError as e:
        return -1, "", f"无法启动解压程序：{e}"
    finally:
        if fh is not None:
            fh.close()
    return proc.returncode, _decode_output(proc.stdout or b""), _decode_output(proc.stderr or b"")


def _looks_encrypted(detail: str) -> bool:
    low = detail.lower()
    return any(k in low for k in ("encrypt", "passphrase", "password"))


def explain_failure(detail: str, name: str) -> str:
    """把解压程序的英文报错翻成给客户看的中文。"""
    low = detail.lower()
    if any(k in low for k in ("no space", "disk full", "not enough space")):
        return f"磁盘空间不足，无法解压「{name}」。请清理磁盘后重试。"
    if _looks_encrypted(detail):
        return f"压缩包「{name}」有密码，暂不支持加密压缩包。请先用解压软件输入密码解压，再把解压出来的文件夹拖进来。"
    if any(k in low for k in ("unrecognized archive", "can not open the file as archive",
                              "cannot open the file as archive", "is not archive")):
        return f"无法识别「{name}」的压缩格式（文件可能已损坏，或扩展名和真实格式不符）。"
    if any(k in low for k in ("truncated", "damaged", "corrupt", "crc", "bad rar", "unexpected end",
                              "invalid", "data error", "headers error")):
        return f"压缩包「{name}」已损坏或不完整（可能没下载完），无法解压。"
    if any(k in low for k in ("unsupported", "not supported", "not implemented")):
        return f"压缩包「{name}」用了暂不支持的压缩方式。可以先用 7-Zip / WinRAR 解压，再把文件夹拖进来。"
    if "tar.exe" in low and "没有找到" in detail:
        return (f"系统里没有找到 tar.exe（Windows 10 1803 及以上自带），无法解压「{name}」。"
                "可以先用解压软件解压，再把文件夹拖进来。")
    return f"解压「{name}」失败。可以先用 7-Zip / WinRAR 解压，再把文件夹拖进来。"


# ---- 解压 ------------------------------------------------------------------------------------------
def _rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.lexists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _has_files(root: str) -> bool:
    for _, _, files in os.walk(root):
        if files:
            return True
    return False


def _tar_member_name(m: tarfile.TarInfo) -> str:
    """tarfile 按 UTF-8 + surrogateescape 解码；不是 UTF-8 的老名字（GBK / Shift-JIS）再猜一次编码。"""
    try:
        m.name.encode("utf-8")
        return m.name
    except UnicodeEncodeError:
        raw = m.name.encode("utf-8", "surrogateescape")
    enc = guess_legacy_encoding([raw])
    if enc:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace")


def _extract_tar_py(path: str, dest: str) -> int:
    """标准库解 tar / tar.gz / tar.bz2 / tar.xz：只取普通文件，路径清洗防穿越，链接 / 设备一律跳过。"""
    count = 0
    with tarfile.open(path, "r:*") as tf:
        for m in tf:                                  # 顺序读：压缩流不会回退重解
            if not m.isfile():
                continue
            rel = clean_entry_name(_tar_member_name(m))
            if not rel or is_junk_path(rel):
                continue
            target = os.path.join(dest, *safe_relpath(rel).split("/"))
            if os.path.exists(target):               # 重复条目只取第一个
                continue
            src = tf.extractfile(m)
            if src is None:
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, _CHUNK)
            count += 1
    return count


def _extract_bsdtar(path: str, dest: str) -> tuple[bool, str]:
    tar = find_tar()
    if not tar:
        return False, "系统里没有找到 tar.exe"
    code, _, err = _run([tar, "-x", "-f", "-"], dest, path, _timeout_for(path))
    return code == 0, err


def _extract_7z(path: str, dest: str) -> tuple[bool, str]:
    exe = find_7z()
    if not exe:
        return False, ""
    # 7-Zip 是 Unicode 程序，参数里的中文路径没问题；-p_ 防止遇到加密包时等待输入密码
    code, out, err = _run([exe, "x", "-y", "-bd", "-p_", "-o" + dest, path], dest, None, _timeout_for(path))
    return code == 0, (err or out)


def _extract_into(src: str, kind: str, staging: str) -> str | None:
    """解压到 staging。成功返回 None，失败返回报错详情（英文原文，交给 explain_failure 翻译）。"""
    details: list[str] = []
    if kind in TAR_KINDS:
        try:
            if _extract_tar_py(src, staging) > 0:
                return None
        except Exception as e:  # noqa: BLE001 - zstd、怪异 tar 变体等交给 bsdtar 再试
            details.append(f"tarfile: {e}")
            log.debug("tarfile 解不开「%s」（%s），改用 tar.exe", src, e)
        _rmtree(staging)
        os.makedirs(staging, exist_ok=True)
    ok, err = _extract_bsdtar(src, staging)
    if ok:
        return None
    details.append(err)
    if find_7z():
        alt = staging + "7"
        _rmtree(alt)
        os.makedirs(alt, exist_ok=True)
        ok7, err7 = _extract_7z(src, alt)
        if ok7 or (_has_files(alt) and not _has_files(staging)):
            _rmtree(staging)
            os.replace(alt, staging)
            return None
        _rmtree(alt)
        details.append(err7)
    detail = "\n".join(d for d in details if d)
    if _has_files(staging) and not _looks_encrypted(detail):
        log.warning("「%s」有部分条目解压失败，已取到其余内容：%s", os.path.basename(src), detail.strip()[:300])
        return None
    return detail or "unknown error"


def _is_done(dest: str, sig: str) -> bool:
    try:
        with open(os.path.join(dest, DONE_MARKER), "r", encoding="utf-8") as f:
            return json.load(f).get("source") == sig
    except (OSError, ValueError, AttributeError):
        return False


def extract_archive(path: str, kind: str, tmp_dir: str | None) -> str:
    """解压到 tmp_dir/x_<签名哈希>/ 并返回该目录；已经完整解压过（标记对得上）就直接复用。失败抛 UnsupportedFormat。"""
    src = os.path.abspath(path)
    name = os.path.basename(src)
    if not os.path.isfile(src):
        raise UnsupportedFormat(f"找不到文件：{src}")
    sig = file_signature(src)
    dest = os.path.join(tmp_dir_or_default(tmp_dir), f"x_{short_hash(sig)}")
    with _key_lock(os.path.normcase(dest)):
        if _is_done(dest, sig):
            return dest
        staging = dest + ".part"
        _rmtree(staging)
        try:
            os.makedirs(staging, exist_ok=True)
        except OSError as e:
            raise UnsupportedFormat(f"无法创建临时解压文件夹：{staging}（{e.strerror or e}）") from e
        t0 = time.perf_counter()
        try:
            detail = _extract_into(src, kind, staging)
        except BaseException:
            _rmtree(staging)
            raise
        if detail is not None:
            _rmtree(staging)
            log.warning("解压失败「%s」：%s", name, detail.strip()[:500])
            raise UnsupportedFormat(explain_failure(detail, name))
        try:
            with open(os.path.join(staging, DONE_MARKER), "w", encoding="utf-8") as f:
                json.dump({"source": sig, "time": time.time()}, f, ensure_ascii=False)
            _rmtree(dest)
            os.replace(staging, dest)
        except OSError as e:                          # 改名失败（被占用等）：直接用 staging，下次重新解压
            log.warning("解压目录改名失败（%s），本次直接使用：%s", e, staging)
            return staging
        log.info("已解压「%s」→ %s（%.1fs）", name, dest, time.perf_counter() - t0)
        return dest


def list_archive_names(path: str) -> list[str] | None:
    """用 tar.exe -t 列出条目名（非 ASCII 字符可能显示成 ?，只用来看扩展名）。列不出返回 None。"""
    tar = find_tar()
    if not tar:
        return None
    src = os.path.abspath(path)
    try:
        code, out, _ = _run([tar, "-t", "-f", "-"], os.path.dirname(src) or None, src, _timeout_for(src))
    except UnsupportedFormat:
        return None
    if code != 0 and not out.strip():
        return None
    names = []
    for line in out.splitlines():
        n = line.strip()
        n = n[2:] if n.startswith("./") else n
        if n and not n.endswith("/") and not is_junk_path(n):
            names.append(n)
    return names


# ---- 书 --------------------------------------------------------------------------------------------
def _no_image_message(name: str, names: list[str]) -> str:
    if any(is_book_name(n) for n in names):
        return f"「{name}」里装的是一卷卷的压缩包 / PDF（套娃），请把它重新拖进来，软件会自动按卷拆开。"
    return f"「{name}」里没有找到图片。"


class ZipBook(Book):
    """zip / cbz：按条目直接读，不全量解压。"""

    def __init__(self, path: str, kind: str = "zip") -> None:
        src = os.path.abspath(path)
        super().__init__(src, kind, title_from_name(src), os.path.dirname(src))
        self._reader = ZipReader(src)
        entries = self._reader.entries()
        self._infos: dict[str, zipfile.ZipInfo] = dict(entries)
        names = [n for n, _ in entries]
        pages = sorted((n for n in names if is_image_name(n)), key=natural_key)
        if not pages:
            self.close()
            raise UnsupportedFormat(_no_image_message(os.path.basename(src), names))
        if any(self._infos[n].flag_bits & 0x1 for n in pages):
            self.close()
            raise UnsupportedFormat(explain_failure("encrypted", os.path.basename(src)))
        extras = [n for n, i in entries if not is_image_name(n) and is_meta_name(n) and i.file_size <= MAX_META_BYTES]
        self._set_entries(pages, sorted(extras, key=natural_key))
        # zipfile 解不了的压缩方式：open_zip 会改走解压
        self.needs_extract = any(self._infos[n].compress_type not in ZIP_METHODS_OK for n in pages)

    def read_page_bytes(self, name: str) -> bytes:
        self._need_page(name)
        return self._reader.read(self._infos[name], name)

    def read_extra(self, name: str) -> bytes:
        self._need_extra(name)
        return self._reader.read(self._infos[name], name)

    def close(self) -> None:
        self._reader.close()


class ExtractedBook(Book):
    """7z / rar / tar（以及 zipfile 读不了的 zip）：解压到 tmp_dir 后按相对路径读取。"""

    def __init__(self, path: str, kind: str, tmp_dir: str | None) -> None:
        src = os.path.abspath(path)
        super().__init__(src, kind, title_from_name(src), os.path.dirname(src))
        self._root = extract_archive(src, kind, tmp_dir)
        rels = [rel for rel, _ in iter_dir_files(self._root)]
        pages = sorted((r for r in rels if is_image_name(r)), key=natural_key)
        if not pages:
            raise UnsupportedFormat(_no_image_message(os.path.basename(src), rels))
        extras = sorted((r for r in rels if not is_image_name(r) and is_meta_name(r) and self._small(r)), key=natural_key)
        self._set_entries(pages, extras)

    def _full(self, name: str) -> str:
        return os.path.join(self._root, *name.split("/"))

    def _small(self, rel: str) -> bool:
        try:
            return os.path.getsize(self._full(rel)) <= MAX_META_BYTES
        except OSError:
            return False

    def _read(self, name: str) -> bytes:
        try:
            with open(self._full(name), "rb") as f:
                return f.read()
        except OSError as e:
            raise UnsupportedFormat(f"读取「{os.path.basename(self.path)}」里的「{name}」失败（临时文件可能被清理了，请重新打开）。") from e

    def read_page_bytes(self, name: str) -> bytes:
        self._need_page(name)
        return self._read(name)

    def read_extra(self, name: str) -> bytes:
        self._need_extra(name)
        return self._read(name)


def open_zip(path: str, kind: str, tmp_dir: str | None) -> Book:
    """zip / cbz：优先 zipfile 直读；打不开或压缩方式不支持时退回解压。"""
    try:
        book = ZipBook(path, kind)
    except ZipUnreadable as e:
        log.info("zipfile 读不了「%s」（%s），改用解压方式", path, e)
        return ExtractedBook(path, kind, tmp_dir)
    if book.needs_extract:
        book.close()
        log.info("「%s」用了 zipfile 不支持的压缩方式，改用解压方式", path)
        return ExtractedBook(path, kind, tmp_dir)
    return book


# ---- 套娃压缩包 ------------------------------------------------------------------------------------
def _write_origin(folder: str, src: str) -> None:
    """在展开目录里记下最外层压缩包（套娃里的套娃沿用最外层），find_origin 靠它决定输出位置和书名。"""
    outer = find_origin(src)
    meta = {"origin": outer[0] if outer else src, "title": outer[1] if outer else title_from_name(src)}
    payload = json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")
    marker = os.path.join(folder, ORIGIN_MARKER)
    try:
        with open(marker, "rb") as f:
            if f.read() == payload:
                return
    except OSError:
        pass
    hio.write_bytes_atomic(marker, payload)


def _copy_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: str) -> None:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    part = target + ".part"
    try:
        with zf.open(info) as src, open(part, "wb") as out:
            shutil.copyfileobj(src, out, _CHUNK)
        os.replace(part, target)
    except BaseException:
        _rmtree(part)
        raise
    try:                                              # 修改时间固定 → 文件签名稳定 → 引擎的断点续跑能认出同一卷
        ts = time.mktime(tuple(info.date_time) + (0, 0, -1))
        os.utime(target, (ts, ts))
    except (OverflowError, ValueError, OSError):
        pass


def _nested_from_zip(path: str, tmp_root: str | None) -> list[str] | None:
    try:
        zf = zipfile.ZipFile(path)
    except Exception:  # noqa: BLE001 - 打不开就不是套娃，交给 open_book 报错
        return None
    with zf:
        entries = zip_entries(zf)
        if any(is_image_name(n) for n, _ in entries):
            return None
        vols = sorted(((n, i) for n, i in entries if is_book_name(n)), key=lambda t: natural_key(t[0]))
        if not vols or any(i.flag_bits & 0x1 or i.compress_type not in ZIP_METHODS_OK for _, i in vols):
            return None
        src = os.path.abspath(path)
        dest = os.path.join(tmp_dir_or_default(tmp_root), f"nested_{short_hash(file_signature(src))}")
        os.makedirs(dest, exist_ok=True)
        _write_origin(dest, src)
        out: list[str] = []
        for n, info in vols:
            target = os.path.join(dest, *safe_relpath(n).split("/"))
            if not (os.path.isfile(target) and os.path.getsize(target) == info.file_size):
                _copy_member(zf, info, target)
            out.append(target)
        return out


def _nested_from_extract(path: str, kind: str, tmp_root: str | None) -> list[str] | None:
    names = list_archive_names(path)
    if not names or any(is_image_name(n) for n in names) or not any(is_book_name(n) for n in names):
        return None
    root = extract_archive(path, kind, tmp_root)
    _write_origin(root, os.path.abspath(path))
    vols = [(rel, full) for rel, full in iter_dir_files(root) if is_book_name(rel)]
    vols.sort(key=lambda t: natural_key(t[0]))
    return [full for _, full in vols] or None


def nested_volumes(path: str, kind: str, tmp_root: str | None) -> list[str] | None:
    """套娃压缩包（没有图片、只有一卷卷的书）→ 展开到 tmp_root 并按阅读顺序返回各卷路径；不是套娃返回 None。

    展开出错也返回 None（记日志），由 open_book 去报具体的中文错误。
    """
    try:
        if kind in ZIP_KINDS:
            return _nested_from_zip(path, tmp_root)
        if kind in EXTRACT_KINDS:
            return _nested_from_extract(path, kind, tmp_root)
    except UnsupportedFormat as e:
        log.warning("套娃压缩包展开失败「%s」：%s", path, e)
    except Exception:  # noqa: BLE001 - 扫描阶段绝不能因为一个坏文件崩掉
        log.exception("套娃压缩包展开出错：%s", path)
    return None

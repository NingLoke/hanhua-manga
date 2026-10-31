"""PDF 书：pypdfium2 渲染页面。

* 单图页（扫描版、本软件导出的 PDF）：按那张图的原始像素渲染，不放大不缩小；
* 其它页（矢量 / 文字 / 多图拼版）：按长边约 2000px 渲染；
* 渲染结果 PNG 缓存到 tmp_dir/pdf_<源文件签名哈希>/，同一页第二次读直接取缓存；
* PDFium 不是线程安全的：所有文档的所有调用共用一把全局锁 PDFIUM_LOCK。
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import cv2
import numpy as np

from .. import imageio as hio
from ..types import UnsupportedFormat
from .formats import Book, file_signature, short_hash, title_from_name, tmp_dir_or_default

log = logging.getLogger("hanhua.bookio")

try:  # pypdfium2 在依赖白名单里；仍做防御，缺失时给中文错误而不是 ImportError
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c
except Exception:  # noqa: BLE001
    pdfium = None
    pdfium_c = None

PDFIUM_LOCK = threading.RLock()
TARGET_LONG_SIDE = 2000          # 非单图页的渲染长边（px）
MAX_LONG_SIDE = 30000            # 渲染长边上限（超长条漫也够用）
MAX_PIXELS = 150_000_000         # 渲染像素上限（≈450MB BGR），超过就等比缩小
SINGLE_IMAGE_COVER = 0.80        # 唯一一张「大图」覆盖页面 ≥80% 面积才算单图页
SMALL_IMAGE_AREA = 0.05          # 小于页面 5% 的图（logo、水印像素）不参与单图判定
_ERR_PASSWORD = int(getattr(pdfium_c, "FPDF_ERR_PASSWORD", 4))
_ERR_SECURITY = int(getattr(pdfium_c, "FPDF_ERR_SECURITY", 5))


def explain_pdf_error(e: BaseException, name: str) -> str:
    code = getattr(e, "err_code", None)
    text = str(e).lower()
    if code == _ERR_PASSWORD or "password" in text:
        return f"PDF「{name}」有打开密码，暂不支持。请先用 PDF 软件输入密码后另存为无密码版本，再拖进来。"
    if code == _ERR_SECURITY or "security" in text:
        return f"PDF「{name}」使用了不支持的加密方式（DRM），无法打开。"
    return f"PDF「{name}」已损坏或不是有效的 PDF 文件，无法打开。"


def _to_bgr(arr: Any) -> np.ndarray:
    """PDFium 位图（BGR / BGRx / BGRA / L）→ 独立内存的 BGR uint8。"""
    a = np.asarray(arr)
    if a.ndim == 2:
        return cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_GRAY2BGR)
    if a.shape[2] == 1:
        return cv2.cvtColor(np.ascontiguousarray(a[:, :, 0]), cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(a[:, :, :3]).copy()


def _native_scale(page: Any, w_pt: float, h_pt: float) -> float | None:
    """单图页：返回让那张图按原始像素显示的缩放比例；不是单图页返回 None。"""
    try:
        objs = list(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3))
    except Exception:  # noqa: BLE001 - 结构异常的页面就按普通页渲染
        return None
    page_area = w_pt * h_pt
    big: list[tuple[float, float, int, int]] = []
    for img in objs:
        try:
            px_w, px_h = img.get_px_size()
            left, bottom, right, top = img.get_bounds()
        except Exception:  # noqa: BLE001
            continue
        bw, bh = abs(right - left), abs(top - bottom)
        if bw * bh >= SMALL_IMAGE_AREA * page_area:
            big.append((bw, bh, int(px_w), int(px_h)))
    if len(big) != 1:
        return None
    bw, bh, px_w, px_h = big[0]
    if min(px_w, px_h) <= 0 or min(bw, bh) <= 1e-3 or bw * bh < SINGLE_IMAGE_COVER * page_area:
        return None
    return max(px_w, px_h) / max(bw, bh)


def _snap(scale: float, w_pt: float, h_pt: float) -> float:
    """render 的位图尺寸是 ceil(pt × scale)：两边都接近整数时略微下调，保证不多出 1px（导出再读尺寸严格一致）。"""
    tw, th = w_pt * scale, h_pt * scale
    rw, rh = round(tw), round(th)
    if rw >= 1 and rh >= 1 and abs(tw - rw) < 0.05 and abs(th - rh) < 0.05:
        return min((rw - 0.002) / w_pt, (rh - 0.002) / h_pt)
    return scale


class PdfBook(Book):
    """PDF：页面名 0001.png、0002.png …（渲染结果是 PNG）。"""

    def __init__(self, path: str, tmp_dir: str | None) -> None:
        src = os.path.abspath(path)
        super().__init__(src, "pdf", title_from_name(src), os.path.dirname(src))
        self._name = os.path.basename(src)
        if pdfium is None:
            raise UnsupportedFormat("缺少 PDF 组件（pypdfium2），无法打开 PDF。请重新安装本软件。")
        if not os.path.isfile(src):
            raise UnsupportedFormat(f"找不到文件：{src}")
        self._cache = os.path.join(tmp_dir_or_default(tmp_dir), f"pdf_{short_hash(file_signature(src))}")
        self._doc: Any = None
        with PDFIUM_LOCK:
            n = len(self._document())
        if n <= 0:
            self.close()
            raise UnsupportedFormat(f"PDF「{self._name}」里没有任何页面。")
        width = max(4, len(str(n)))
        self._set_entries([f"{i + 1:0{width}d}.png" for i in range(n)], [])
        self._index = {name: i for i, name in enumerate(self.pages)}

    # -- PDFium（调用方必须持有 PDFIUM_LOCK） --
    def _document(self) -> Any:
        if self._doc is None:
            try:
                self._doc = pdfium.PdfDocument(self.path)
            except pdfium.PdfiumError as e:
                raise UnsupportedFormat(explain_pdf_error(e, self._name)) from e
            except OSError as e:
                raise UnsupportedFormat(f"读取 PDF「{self._name}」失败（{e.strerror or e}）。") from e
            except Exception as e:  # noqa: BLE001 - pypdfium2 其它异常
                raise UnsupportedFormat(explain_pdf_error(e, self._name)) from e
        return self._doc

    def _scale_for(self, page: Any) -> float:
        w_pt, h_pt = (max(float(v), 1.0) for v in page.get_size())
        long_pt = max(w_pt, h_pt)
        native = _native_scale(page, w_pt, h_pt)
        scale = native if native else TARGET_LONG_SIDE / long_pt
        cap = min(MAX_LONG_SIDE / long_pt, (MAX_PIXELS / (w_pt * h_pt)) ** 0.5)
        if scale > cap:
            log.info("PDF 页面太大，按上限缩小渲染：%s", self._name)
            return max(cap, 0.01)
        return max(_snap(scale, w_pt, h_pt) if native else scale, 0.01)

    def _render(self, index: int) -> np.ndarray:
        with PDFIUM_LOCK:
            doc = self._document()
            try:
                page = doc[index]
            except Exception as e:  # noqa: BLE001
                raise UnsupportedFormat(f"PDF「{self._name}」第 {index + 1} 页已损坏，无法读取。") from e
            try:
                bitmap = page.render(scale=self._scale_for(page), may_draw_forms=True)
                try:
                    return _to_bgr(bitmap.to_numpy())
                finally:
                    bitmap.close()
            except UnsupportedFormat:
                raise
            except MemoryError as e:
                raise UnsupportedFormat(f"PDF「{self._name}」第 {index + 1} 页太大，内存不足，无法渲染。") from e
            except Exception as e:  # noqa: BLE001
                raise UnsupportedFormat(f"PDF「{self._name}」第 {index + 1} 页渲染失败：{e}") from e
            finally:
                page.close()

    # -- 缓存 --
    def _cache_path(self, name: str) -> str:
        return os.path.join(self._cache, name)

    def _cached(self, name: str) -> bytes | None:
        try:
            with open(self._cache_path(name), "rb") as f:
                data = f.read()
            return data or None
        except OSError:
            return None

    def _render_and_cache(self, name: str) -> tuple[np.ndarray, bytes]:
        img = self._render(self._index[name])
        data = hio.encode(img, "png")
        try:
            hio.write_bytes_atomic(self._cache_path(name), data)
        except OSError as e:
            log.warning("PDF 页面缓存写入失败（不影响使用）：%s", e)
        return img, data

    # -- Book 接口 --
    def read_page_bytes(self, name: str) -> bytes:
        self._need_page(name)
        return self._cached(name) or self._render_and_cache(name)[1]

    def read_page(self, name: str) -> np.ndarray:
        self._need_page(name)
        data = self._cached(name)
        if data is not None:
            try:
                return hio.decode(data)
            except ValueError:
                log.warning("PDF 页面缓存损坏，重新渲染：%s", name)
        return self._render_and_cache(name)[0]

    def read_extra(self, name: str) -> bytes:
        self._need_extra(name)
        return b""

    def close(self) -> None:
        with PDFIUM_LOCK:
            if self._doc is not None:
                try:
                    self._doc.close()
                except Exception:  # noqa: BLE001
                    pass
                self._doc = None

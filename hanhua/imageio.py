"""中文路径安全的图片读写。全项目禁止直接 cv2.imread / cv2.imwrite（Windows 中文路径会失败）。

内部图片统一为 BGR uint8 (H, W, 3)。
"""
from __future__ import annotations

import io
import os
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}
WRITE_EXTS = {"jpg", "png", "webp"}


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def _normalize(arr: np.ndarray) -> np.ndarray:
    """任意 cv2 解码结果 → BGR uint8；透明背景合成到白底（漫画页透明处一般是白纸）。"""
    if arr.dtype != np.uint8:
        if arr.dtype == np.uint16:
            arr = (arr / 257.0).round().astype(np.uint8)
        else:
            a = arr.astype(np.float32)
            mx = float(a.max()) if a.size else 1.0
            arr = np.clip(a * (255.0 / mx if mx > 1.0 else 255.0), 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        return np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])


def decode(data: bytes) -> np.ndarray:
    """bytes → BGR uint8。cv2 解不了的（AVIF / CMYK JPEG / 部分 GIF）交给 Pillow。"""
    buf = np.frombuffer(data, np.uint8)
    arr = None
    if buf.size:
        try:
            arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        except cv2.error:
            arr = None
    if arr is not None and arr.size:
        return _normalize(arr)
    try:
        im = Image.open(io.BytesIO(data))
        im.seek(0)
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA", "L", "LA"):
            im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
        a = np.asarray(im)
        if a.ndim == 3 and a.shape[2] == 3:
            a = a[:, :, ::-1]
        elif a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, [2, 1, 0, 3]]
        elif a.ndim == 3 and a.shape[2] == 2:  # LA
            a = np.dstack([a[:, :, 0]] * 3 + [a[:, :, 1]])
        return _normalize(np.ascontiguousarray(a))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"无法解码图片：{e}") from e


def read(path: str) -> np.ndarray:
    return decode(np.fromfile(path, dtype=np.uint8).tobytes())


def encode(img: np.ndarray, fmt: str = "png", quality: int = 92) -> bytes:
    """BGR → bytes。fmt: jpg | png | webp（png 用快速压缩档）。"""
    fmt = fmt.lower().lstrip(".")
    if fmt in ("jpg", "jpeg", "jfif"):
        ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    elif fmt == "webp":
        ok, b = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
    elif fmt == "png":
        ok, b = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif fmt == "bmp":
        ok, b = cv2.imencode(".bmp", img)
    else:
        raise ValueError(f"不支持的输出格式：{fmt}")
    if not ok:
        raise ValueError(f"图片编码失败：{fmt}")
    return b.tobytes()


def write_bytes_atomic(path: str, data: bytes) -> None:
    """先写同目录临时文件再替换：断电 / 崩溃不会留下半个文件（断点续跑依赖这一点）。"""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write(path: str, img: np.ndarray, quality: int = 92) -> None:
    fmt = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    write_bytes_atomic(path, encode(img, fmt, quality))


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def from_pil(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def thumbnail(img: np.ndarray, max_side: int = 360) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)

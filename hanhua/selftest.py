"""无界面自检：打包后用 exe 跑一遍完整流水线（不联网），确认模型 / DLL / 字体 / 界面依赖都齐全。

一键汉化漫画.exe --selftest 报告.json   → 退出码 0 = 通过
"""
from __future__ import annotations

import json
import os
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import runtime
from .config import DEFAULTS


def _test_page() -> np.ndarray:
    """白纸 + 两个椭圆气泡 + 竖排日文（用自带的 Noto Sans SC，含假名）。"""
    w, h = 1000, 1400
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(os.path.join(runtime.fonts_dir(), "NotoSansSC-VF.ttf"), 34)
    try:
        font.set_variation_by_axes([700])
    except Exception:  # noqa: BLE001
        pass
    bubbles = [((560, 120, 860, 560), ["そんなこと", "ないよ！"]), ((150, 700, 450, 1150), ["今日は", "いい天気だね"])]
    for (x1, y1, x2, y2), cols in bubbles:
        d.ellipse((x1, y1, x2, y2), fill=(255, 255, 255), outline=(0, 0, 0), width=5)
        cx = (x1 + x2) // 2 + 20 * (len(cols) - 1)
        for ci, col in enumerate(cols):
            x = cx - ci * 44
            y = (y1 + y2) // 2 - len(col) * 38 // 2
            for ch in col:
                d.text((x, y), ch, font=font, fill=(0, 0, 0), anchor="mt")
                y += 38
    # 一点网点背景，让 LaMa 也有机会被调用
    arr = np.asarray(im).copy()
    arr[1250:1400, :, :] = 200
    arr[1250:1400:6, ::6, :] = 90
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _webview2_version() -> str:
    try:
        import winreg

        keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"),
            (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"),
        ]
        for hive, path in keys:
            try:
                with winreg.OpenKey(hive, path) as k:
                    v = str(winreg.QueryValueEx(k, "pv")[0])
                    if v and v != "0.0.0.0":
                        return v
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return ""


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    report_path = argv[0] if argv else os.path.join(runtime.data_dir(), "selftest.json")
    rep: dict = {"ok": False, "version": runtime.VERSION, "build": runtime.build_info(), "steps": []}

    def step(name: str, t0: float, **extra) -> None:
        rep["steps"].append({"name": name, "ms": round((time.perf_counter() - t0) * 1000), **extra})

    try:
        hw = runtime.detect_hardware()
        rep["hardware"] = hw.to_dict()
        rep["webview2"] = _webview2_version()
        try:
            import clr  # noqa: F401  pythonnet：pywebview 在 Windows 上靠它驱动 WebView2
            import webview  # noqa: F401

            rep["gui_deps"] = "ok"
        except Exception as e:  # noqa: BLE001
            rep["gui_deps"] = f"失败：{e}"

        t = time.perf_counter()
        rt = runtime.Runtime("auto")
        prof = runtime.resolve_profile("balanced", rt.device)
        rep["device"], rep["label"], rep["profile"] = rt.device, rt.label, prof.to_dict()
        step("runtime", t)

        from .detect import TextDetector
        from .inpaint import Inpainter
        from .ocr import OCREngine
        from .typeset import FontRegistry, Typesetter

        img = _test_page()
        settings = json.loads(json.dumps(DEFAULTS))

        t = time.perf_counter()
        blocks, bubbles = TextDetector(rt, prof).detect(img, "rtl")
        step("detect", t, blocks=len(blocks), bubbles=len(bubbles))

        t = time.perf_counter()
        OCREngine(rt, prof, "ja").recognize(img, blocks)
        step("ocr", t, texts=[b.src_text for b in blocks])

        for b in blocks:
            b.translation = "自检通过！"
        t = time.perf_counter()
        clean, _mask = Inpainter(rt, prof, settings).clean(img, blocks)
        step("inpaint", t)

        t = time.perf_counter()
        fonts = FontRegistry(runtime.fonts_dir(), [os.path.join(runtime.data_dir(), "fonts")])
        out = Typesetter(settings, fonts).render(clean, blocks)
        step("typeset", t, fonts=len(fonts.list()))

        t = time.perf_counter()  # 确保 LaMa 在当前设备上真的能跑
        rt.model("lama").run({"image": np.random.rand(1, 3, 256, 256).astype(np.float32),
                              "mask": np.ones((1, 1, 256, 256), np.float32)})
        step("lama", t)

        rep["model_providers"] = {k: v["provider"] for k, v in rt.stats().items()}
        rep["output_shape"] = list(out.shape)
        problems = []
        if not blocks:
            problems.append("测试页没有检测到文字")
        if blocks and not any(b.src_text for b in blocks):
            problems.append("OCR 没有识别出文字")
        if rep.get("gui_deps") != "ok":
            problems.append("界面依赖缺失")
        if rt.device != "cpu" and any(p == "CPUExecutionProvider" for p in rep["model_providers"].values()):
            problems.append("有模型退回了 CPU")
        rep["problems"] = problems
        rep["ok"] = not problems
    except Exception:  # noqa: BLE001
        rep["error"] = traceback.format_exc()

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return 0 if rep["ok"] else 1

"""下载 / 校验 4.0 需要的模型和字体（全部为 Apache-2.0 / MIT / OFL 可商用授权）。

用法:
    python build/fetch_assets.py            # 下载全部
    python build/fetch_assets.py --list     # 只列清单
    python build/fetch_assets.py --only models/ppocr

中文路径安全；支持断点续传（.part 文件）、多线程、失败重试。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co"
GF = "https://github.com/google/fonts/raw/main/ofl"

# (相对 4.0/ 的目标路径, 下载地址或 ("gh_release", owner/repo, 资产文件名))
ASSETS: list[tuple[str, object]] = [
    # 气泡 + 文字检测 RT-DETR-v2（Apache-2.0）ogkalu/comic-text-and-bubble-detector
    ("models/detector/detector.onnx", f"{HF}/ogkalu/comic-text-and-bubble-detector/resolve/main/detector.onnx"),
    ("models/detector/detector_int8.onnx", f"{HF}/ogkalu/comic-text-and-bubble-detector/resolve/main/detector_int8.onnx"),
    ("models/detector/detector-v4-s_int8.onnx", f"{HF}/ogkalu/comic-text-and-bubble-detector/resolve/main/detector-v4-s_int8.onnx"),
    ("models/detector/config.json", f"{HF}/ogkalu/comic-text-and-bubble-detector/resolve/main/config.json"),
    ("models/detector/README.md", f"{HF}/ogkalu/comic-text-and-bubble-detector/resolve/main/README.md"),
    # 日文 OCR manga-ocr（Apache-2.0）onnx-community/manga-ocr-base-ONNX + 原版词表
    ("models/manga-ocr/encoder_model.onnx", f"{HF}/onnx-community/manga-ocr-base-ONNX/resolve/main/onnx/encoder_model.onnx"),
    ("models/manga-ocr/decoder_model.onnx", f"{HF}/onnx-community/manga-ocr-base-ONNX/resolve/main/onnx/decoder_model.onnx"),
    ("models/manga-ocr/encoder_model_int8.onnx", f"{HF}/onnx-community/manga-ocr-base-ONNX/resolve/main/onnx/encoder_model_int8.onnx"),
    ("models/manga-ocr/decoder_model_int8.onnx", f"{HF}/onnx-community/manga-ocr-base-ONNX/resolve/main/onnx/decoder_model_int8.onnx"),
    ("models/manga-ocr/config.json", f"{HF}/onnx-community/manga-ocr-base-ONNX/resolve/main/config.json"),
    ("models/manga-ocr/generation_config.json", f"{HF}/onnx-community/manga-ocr-base-ONNX/resolve/main/generation_config.json"),
    ("models/manga-ocr/preprocessor_config.json", f"{HF}/onnx-community/manga-ocr-base-ONNX/resolve/main/preprocessor_config.json"),
    ("models/manga-ocr/README.md", f"{HF}/onnx-community/manga-ocr-base-ONNX/resolve/main/README.md"),
    ("models/manga-ocr/vocab.txt", f"{HF}/kha-white/manga-ocr-base/resolve/main/vocab.txt"),
    ("models/manga-ocr/tokenizer_config.json", f"{HF}/kha-white/manga-ocr-base/resolve/main/tokenizer_config.json"),
    ("models/manga-ocr/special_tokens_map.json", f"{HF}/kha-white/manga-ocr-base/resolve/main/special_tokens_map.json"),
    # 去字修复 LaMa（漫画微调，动态尺寸；Apache-2.0，原权重 MIT）
    ("models/lama/lama-manga-dynamic.onnx", f"{HF}/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx"),
    ("models/lama/README.md", f"{HF}/ogkalu/lama-manga-onnx-dynamic/resolve/main/README.md"),
    # 韩 / 英 / 中 / 日横排 OCR：PP-OCRv5 mobile（Apache-2.0，PaddlePaddle 官方 ONNX）
    ("models/ppocr/det.onnx", f"{HF}/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx"),
    ("models/ppocr/det.yml", f"{HF}/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/main/inference.yml"),
    ("models/ppocr/rec_multi.onnx", f"{HF}/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx"),
    ("models/ppocr/rec_multi.yml", f"{HF}/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.yml"),
    ("models/ppocr/rec_korean.onnx", f"{HF}/PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx"),
    ("models/ppocr/rec_korean.yml", f"{HF}/PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.yml"),
    ("models/ppocr/rec_en.onnx", f"{HF}/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx"),
    ("models/ppocr/rec_en.yml", f"{HF}/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.yml"),
    # 字体（全部 SIL OFL 1.1，可商用、可随软件分发；授权文件一起带上）
    ("fonts/NotoSansSC-VF.ttf", f"{GF}/notosanssc/NotoSansSC%5Bwght%5D.ttf"),
    ("fonts/licenses/NotoSansSC-OFL.txt", f"{GF}/notosanssc/OFL.txt"),
    ("fonts/NotoSansTC-VF.ttf", f"{GF}/notosanstc/NotoSansTC%5Bwght%5D.ttf"),
    ("fonts/licenses/NotoSansTC-OFL.txt", f"{GF}/notosanstc/OFL.txt"),
    ("fonts/ZCOOLKuaiLe-Regular.ttf", f"{GF}/zcoolkuaile/ZCOOLKuaiLe-Regular.ttf"),
    ("fonts/licenses/ZCOOLKuaiLe-OFL.txt", f"{GF}/zcoolkuaile/OFL.txt"),
    ("fonts/MaShanZheng-Regular.ttf", f"{GF}/mashanzheng/MaShanZheng-Regular.ttf"),
    ("fonts/licenses/MaShanZheng-OFL.txt", f"{GF}/mashanzheng/OFL.txt"),
    ("fonts/LXGWWenKai-Regular.ttf", ("gh_release", "lxgw/LxgwWenKai", "LXGWWenKai-Regular.ttf")),
    ("fonts/licenses/LXGWWenKai-OFL.txt", "https://raw.githubusercontent.com/lxgw/LxgwWenKai/main/OFL.txt"),
]

UA = {"User-Agent": "hanhua-fetch/4.0"}

# llama.cpp 本地翻译引擎（MIT），每个发行版只打包自己的后端；版本固定，保证可复现
LLAMA_BUILD = "b12001"
LLAMA_ZIPS: list[tuple[str, str]] = [
    ("runtimes/llama/cuda", f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-win-cuda-13.4-x64.zip"),
    ("runtimes/llama/vulkan", f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip"),
    ("runtimes/llama/cpu", f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip"),
]


def fetch_zip(rel_dir: str, url: str) -> str:
    """下载 llama.cpp 发行 zip 并解压到 rel_dir（已有 llama-server.exe 就跳过）。"""
    import zipfile

    dest = os.path.join(ROOT, rel_dir)
    if os.path.exists(os.path.join(dest, "llama-server.exe")):
        return f"skip  {rel_dir}"
    zrel = os.path.join("runtimes", "_downloads", os.path.basename(url))
    fetch(zrel, url)
    with zipfile.ZipFile(os.path.join(ROOT, zrel)) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or name.startswith("/") or ".." in name.split("/"):
                continue                       # 防路径穿越
            target = os.path.join(dest, os.path.basename(name))   # 发行包是平铺的 exe/dll
            os.makedirs(dest, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    with open(os.path.join(dest, "LLAMA_BUILD.txt"), "w", encoding="utf-8") as f:
        f.write(LLAMA_BUILD + "\n" + url + "\n")
    return f"ok    {rel_dir}（llama.cpp {LLAMA_BUILD}）"


def _resolve(src: object) -> str:
    if isinstance(src, str):
        return src
    kind, repo, name = src  # type: ignore[misc]
    assert kind == "gh_release"
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/latest", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        rel = json.load(r)
    for a in rel.get("assets", []):
        if a.get("name") == name:
            return a["browser_download_url"]
    raise RuntimeError(f"{repo} 最新 release 里没有 {name}")


def fetch(rel_path: str, src: object, retries: int = 5) -> str:
    dest = os.path.join(ROOT, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return f"skip  {rel_path} ({os.path.getsize(dest) / 1e6:.1f} MB)"
    part = dest + ".part"
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            url = _resolve(src)
            have = os.path.getsize(part) if os.path.exists(part) else 0
            headers = dict(UA)
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # 服务器不支持续传，从头来
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(part, dest)
            return f"ok    {rel_path} ({os.path.getsize(dest) / 1e6:.1f} MB)"
        except Exception as e:  # noqa: BLE001 - 网络错误统一重试
            last_err = e
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"下载失败 {rel_path}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="", help="只处理以此前缀开头的路径")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    items = [(p, s) for p, s in ASSETS if p.startswith(args.only)]
    if args.list:
        for p, s in items:
            print(p, "<-", s if isinstance(s, str) else "/".join(s[1:]))
        return 0
    zips = [(d, u) for d, u in LLAMA_ZIPS if d.startswith(args.only) or args.only.startswith("runtimes")]
    if args.only and not args.only.startswith("runtimes"):
        zips = [(d, u) for d, u in zips if d.startswith(args.only)]
    failed = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, p, s): p for p, s in items}
        futs.update({ex.submit(fetch_zip, d, u): d for d, u in zips})
        for fu in cf.as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL ", e, flush=True)
    print(f"完成：{len(items) - failed}/{len(items)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

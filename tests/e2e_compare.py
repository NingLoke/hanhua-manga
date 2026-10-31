"""端到端对比：用真实 Engine 把 tests/fixtures/real 当一本书跑完，输出 原图 | 3.0 | 4.0 三联图。

用法（4.0 目录下）：
    envs/cuda/Scripts/python.exe -X utf8 tests/e2e_compare.py [--device auto|cuda|dml|cpu] [--profile balanced]
输出：tests/out/e2e/<device>/compare_pXXX.jpg、report.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

V3_ZIP = os.path.join(os.path.dirname(ROOT), "文字版（熟肉3.0）_汉化版.zip")
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "real")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--llm-backend", default="auto", help="本地翻译后端 auto|cuda|vulkan|cpu（模拟不同发行版）")
    ap.add_argument("--llm-model", default="auto")
    args = ap.parse_args()

    real_llm = os.path.join(os.environ.get("APPDATA", ""), "HanhuaManga", "models", "llm")
    data = tempfile.mkdtemp(prefix="hanhua_e2e_")
    os.environ["HANHUA_DATA"] = data
    if os.path.isdir(real_llm):
        os.environ.setdefault("HANHUA_LLM_DIR", real_llm)     # 共用已下载的本地翻译模型
    out_root = os.path.join(ROOT, "tests", "out", "e2e", args.device)
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root, exist_ok=True)

    import cv2
    import numpy as np

    from hanhua import imageio
    from hanhua.config import SettingsStore
    from hanhua.engine import Engine

    # 把 fixtures 复制到临时“书”目录（避免输出写进 fixtures）
    book = os.path.join(data, "测试书")
    os.makedirs(book)
    for n in sorted(os.listdir(FIXTURES)):
        if n.lower().endswith(".jpg"):
            shutil.copy2(os.path.join(FIXTURES, n), os.path.join(book, n))

    store = SettingsStore()
    store.update({
        "hardware": {"device": args.device, "profile": args.profile},
        "translate": {"local": {"backend": args.llm_backend, "model": args.llm_model}},
        "source": {"lang": "ja"},
        "output": {"dir": out_root, "format": "folder"},
    })
    events: list[dict] = []
    done = threading.Event()

    def on_event(ev: dict) -> None:
        events.append(ev)
        if ev.get("type") == "job" and ev["job"].get("status") in ("done", "error"):
            done.set()
        if ev.get("type") == "log":
            print(f"[{ev.get('level')}] {ev.get('message')}", flush=True)

    eng = Engine(store, on_event)
    t0 = time.perf_counter()
    jobs = eng.add_paths([book])
    eng.start()
    if not done.wait(args.timeout):
        print("超时！", flush=True)
    elapsed = time.perf_counter() - t0
    job = next(j for j in eng.list_jobs() if j["id"] == jobs[0]["id"])
    print(json.dumps(job, ensure_ascii=False, indent=1), flush=True)

    v3 = zipfile.ZipFile(V3_ZIP) if os.path.exists(V3_ZIP) else None
    pages = []
    for i in range(job["total"]):
        page = eng.get_page(job["id"], i)
        name = os.path.basename(page["name"])
        src = imageio.read(os.path.join(book, name))
        out_p = eng.page_image_path(job["id"], i, "out")
        out = imageio.read(out_p) if out_p and os.path.exists(out_p) else np.zeros_like(src)
        h, w = src.shape[:2]
        tiles = [src]
        if v3 is not None:
            try:
                tiles.append(cv2.resize(imageio.decode(v3.read(name.lstrip("p"))), (w, h)))
            except KeyError:
                pass
        tiles.append(cv2.resize(out, (w, h)))
        sep = np.full((h, 8, 3), (0, 0, 255), np.uint8)
        row = tiles[0]
        for t in tiles[1:]:
            row = np.hstack([row, sep, t])
        row = cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA)
        imageio.write(os.path.join(out_root, f"compare_{os.path.splitext(name)[0]}.jpg"), row, 88)
        pages.append({
            "name": name, "stage": page["stage"], "error": page.get("error", ""),
            "timings": page.get("timings", {}),
            "blocks": [{"kind": b["kind"], "src": b["src_text"], "zh": b["translation"],
                        "font": b.get("render_font_size"), "overflow": b.get("overflow")} for b in page["blocks"]],
        })
    report = {"device": args.device, "profile": args.profile, "elapsed_s": round(elapsed, 1),
              "per_page_s": round(elapsed / max(1, job["total"]), 2), "job": job, "pages": pages}
    with open(os.path.join(out_root, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    eng.shutdown()
    print(f"完成：{job['total']} 页，用时 {elapsed:.1f}s，输出 {out_root}", flush=True)
    return 0 if job.get("status") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())

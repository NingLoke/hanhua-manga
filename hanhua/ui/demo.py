"""演示用假引擎 DemoEngine：方法签名与 hanhua.engine.Engine 一致，但不加载任何模型。

用途：
  - ``python -m hanhua.app --demo``：窗口演示；``--demo --no-window``：只起服务器，浏览器里调界面；
  - tests/test_server.py：端到端测试 Api + AppServer。

素材：tests/fixtures/real 的页面（打包后没有素材时现场合成简单页面）。
"成品"做法：启发式找浅色气泡 → 擦掉气泡里的原文 → 用内置字体写上演示译文，
足够展示前后对比、缩略图和校对编辑器；每页的阶段推进由定时器模拟（流水线式错开）。
"""
from __future__ import annotations

import collections
import copy
import hashlib
import logging
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from hanhua import imageio
from hanhua.runtime import DEVICE_LABELS, data_dir, detect_hardware, fonts_dir
from hanhua.types import STAGES, Box, Cancelled, HanhuaError, PageData, TextBlock, box_iou, clamp_box, stage_at_least

log = logging.getLogger("hanhua.demo")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FIXTURES = os.path.join(_PROJECT_ROOT, "tests", "fixtures", "real")

# 演示用原文 / 译文 / 重译（原创短句）
_SRC_LINES = (
    "今日はいい天気だね。", "待って！どこへ行くの？", "ずっと待ってたんだ…", "この試合、絶対に勝つ！",
    "そういうことか。", "ありがとう、本当に。", "まさか…あいつが？", "早く行こうぜ！",
)
_DST_LINES = (
    "今天天气真好啊。", "等一下！你要去哪里？", "我一直在等你……", "这场比赛我们一定要赢！",
    "原来是这样啊。", "谢谢你，真的。", "难道说……是那家伙？", "快走吧！",
)
_ALT_LINES = (
    "今天的天气真不错。", "等等！你要去哪儿？", "我一直都在等你……", "这场比赛，绝对要赢！",
    "原来如此。", "真的，谢谢你。", "不会吧……是他？", "赶紧走吧！",
)
_KINDS = {
    ".zip": "zip", ".cbz": "cbz", ".7z": "7z", ".cb7": "cb7", ".rar": "rar", ".cbr": "cbr",
    ".tar": "tar", ".cbt": "tar", ".pdf": "pdf", ".epub": "epub",
}
# (标题, 类型, 页数, 状态, 已完成页数, 消息)
_SEED_JOBS = (
    ("星辰物语 第3卷", "cbz", 18, "done", 18, ""),
    ("夏日球场 第12话", "zip", 64, "running", 9, ""),
    ("海风与少年 Vol.2", "pdf", 32, "queued", 0, ""),
    ("灯塔下的约定", "folder", 14, "paused", 5, "已暂停（进度已保存，随时可以继续）"),
    ("旧硬盘里的漫画", "rar", 0, "error", 0, "压缩包已损坏或需要密码，无法读取"),
)
_DEMO_FAIL_PAGE = 13          # 演示"单页出错"：进行中的那本书第 14 页
_IN_FLIGHT = 4                # 流水线同时在途的页数


# --------------------------------------------------------------------------------------
# 启发式气泡 + 演示渲染
# --------------------------------------------------------------------------------------
def _fill_holes(comp: np.ndarray) -> np.ndarray:
    """0/1 连通域 → 填上内部的洞（气泡里的文字笔画就是洞）。"""
    pad = np.pad(comp.astype(np.uint8), 1, constant_values=0)
    mask = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(pad, mask, (0, 0), 2)
    return (pad[1:-1, 1:-1] != 2).astype(np.uint8)


def find_bubbles(img: np.ndarray, max_n: int = 6) -> list[dict]:
    """找"浅色、比较实、里面有深色笔画"的封闭区域当气泡（只为演示，阈值全部相对页面）。"""
    h, w = img.shape[:2]
    scale = min(1.0, 900.0 / max(h, w))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else gray
    sh, sw = small.shape
    page_area = float(sh * sw)
    cands: list[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    # 气泡常带灰色 / 夕阳色和渐变；单一白色阈值会漏掉整页对白。
    for threshold in (225, 210, 195, 180, 165):
        n, labels, stats, _ = cv2.connectedComponentsWithStats((small >= threshold).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            x, y, bw, bh, _area = (int(v) for v in stats[i])
            if x <= 1 or y <= 1 or x + bw >= sw - 1 or y + bh >= sh - 1:
                continue                                    # 碰到页边的是留白 / 分镜间隙
            box_area = bw * bh
            if not (0.006 * page_area <= box_area <= 0.12 * page_area) or not (0.2 <= bw / bh <= 3.5):
                continue
            comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
            filled = _fill_holes(comp)
            fa = int(filled.sum())
            if fa < 0.55 * box_area:
                continue
            holes = filled & (1 - comp)
            ratio = float(holes.sum()) / max(1, fa)
            if not 0.03 <= ratio <= 0.6:
                continue
            ys, xs = np.nonzero(holes)
            cands.append((fa, (x, y, bw, bh), (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)))
    cands.sort(key=lambda c: -c[0])
    inv = 1.0 / scale
    out: list[dict] = []
    for _fa, (x, y, bw, bh), (tx1, ty1, tx2, ty2) in cands:
        pad = max(2, int(round(0.04 * min(bw, bh))))
        bubble = clamp_box((int((x - pad) * inv), int((y - pad) * inv),
                            int((x + bw + pad) * inv), int((y + bh + pad) * inv)), w, h)
        if any(box_iou(bubble, item["bubble_box"]) >= 0.5 for item in out):
            continue
        # 二值化找到的是深色笔画，外侧描边 / 抗锯齿也要纳入擦除窗口。
        px = max(2, int(round(0.08 * (tx2 - tx1))))
        py = max(2, int(round(0.04 * (ty2 - ty1))))
        text = clamp_box((int((x + tx1 - px) * inv), int((y + ty1 - py) * inv),
                          int((x + tx2 + px) * inv), int((y + ty2 + py) * inv)), w, h)
        vertical = (text[3] - text[1]) >= (text[2] - text[0]) * 1.1
        out.append({"bubble_box": bubble, "box": text, "direction": "v" if vertical else "h"})
        if len(out) >= max_n:
            break
    # 阅读顺序（日漫）：从上到下、从右到左
    out.sort(key=lambda d: (d["bubble_box"][1] // max(1, h // 6), -d["bubble_box"][0]))
    return out


class _Fonts:
    """演示和正式排版共用字体库。"""

    def __init__(self) -> None:
        self._reg: Any = None
        try:
            from hanhua.typeset.fonts import FontRegistry

            self._reg = FontRegistry(fonts_dir(), [os.path.join(data_dir(), "fonts")])
        except Exception as e:  # noqa: BLE001
            log.warning("演示：字体库不可用：%s", e)


def render_page(img: np.ndarray, blocks: list[TextBlock], fonts: _Fonts) -> tuple[np.ndarray, np.ndarray]:
    """返回 (去字补丁 BGRA：alpha=修改过的像素, 成品 BGR)；同时回写块的 render_box 等。"""
    from hanhua.inpaint import Inpainter
    from hanhua.typeset import Typesetter

    h, w = img.shape[:2]
    # 与正式引擎共用擦字和气泡轮廓排版；不加载推理模型。
    clean, mask = Inpainter(None, None, {}).clean(img, blocks)
    if fonts._reg is None:
        raise HanhuaError("演示嵌字失败：字体库不可用")
    out = Typesetter({}, fonts._reg).render(clean, blocks)
    patch = np.zeros((h, w, 4), np.uint8)
    patch[..., :3] = clean
    patch[..., 3] = mask
    return patch, out


def _synth_page(path: str, k: int) -> None:
    """没有真实素材时合成一张"漫画页"：分镜框 + 几个带假文字的白气泡。"""
    rng = random.Random(1000 + k)
    w, h = 1200, 1700
    img = np.full((h, w, 3), 250, np.uint8)
    for (x1, y1, x2, y2) in ((40, 40, 1160, 620), (40, 660, 590, 1660), (630, 660, 1160, 1660)):
        shade = rng.randint(150, 215)
        img[y1:y2, x1:x2] = (shade, shade - 6, shade - 14)
        cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 6)
    for cx, cy in ((330, 300), (880, 360), (300, 1000), (900, 1200)):
        ax, ay = rng.randint(90, 130), rng.randint(150, 200)
        cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (20, 20, 20), 4)
        for col in range(3):
            x = cx + 40 - col * 40
            for row in range(rng.randint(4, 7)):
                y = cy - 110 + row * 36
                cv2.rectangle(img, (x - 12, y), (x + 12, y + 24), (30, 30, 30), -1)
    imageio.write(path, img)


def _sample_sources(cache_dir: str) -> list[str]:
    d = os.environ.get("HANHUA_DEMO_PAGES") or _FIXTURES
    files: list[str] = []
    try:
        files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if imageio.is_image_name(f)]
    except OSError:
        files = []
    if files:
        return files
    os.makedirs(cache_dir, exist_ok=True)
    for k in range(4):
        p = os.path.join(cache_dir, f"synth_{k}.jpg")
        if not os.path.isfile(p):
            _synth_page(p, k)
        files.append(p)
    return files


class _SampleBank:
    """素材页 → (块、去字补丁、成品、缩略图)，每张素材只生成一次并落盘缓存。"""

    def __init__(self, root: str) -> None:
        self.dir = os.path.join(root, "samples")
        os.makedirs(self.dir, exist_ok=True)
        self.sources = _sample_sources(self.dir)
        self._lock = threading.Lock()
        self._built: dict[int, dict] = {}
        self.fonts = _Fonts()

    def __len__(self) -> int:
        return len(self.sources)

    def source(self, k: int) -> str:
        return self.sources[k % len(self.sources)]

    def get(self, k: int) -> dict:
        k %= len(self.sources)
        with self._lock:
            info = self._built.get(k)
            if info is None:
                info = self._build(k)
                self._built[k] = info
            return info

    def _build(self, k: int) -> dict:
        t = time.perf_counter()
        img = imageio.read(self.sources[k])
        h, w = img.shape[:2]
        blocks: list[TextBlock] = []
        for j, f in enumerate(find_bubbles(img)):
            line = (k * 3 + j) % len(_DST_LINES)
            blocks.append(TextBlock(
                id=j, box=f["box"], kind="bubble", score=round(0.86 + 0.02 * (j % 5), 2), bubble_box=f["bubble_box"],
                direction=f["direction"], src_text=_SRC_LINES[line], ocr_conf=0.93,
                translation=_DST_LINES[line], text_color=(22, 22, 22),
            ))
        patch, out = render_page(img, blocks, self.fonts)
        d = os.path.join(self.dir, f"s{k:02d}")
        info = {
            "w": w, "h": h, "blocks": blocks,
            "clean": os.path.join(d, "clean.png"), "out": os.path.join(d, "out.jpg"), "thumb": os.path.join(d, "thumb.jpg"),
        }
        imageio.write_bytes_atomic(info["clean"], imageio.encode(patch, "png"))
        imageio.write(info["out"], out, 90)
        imageio.write(info["thumb"], imageio.thumbnail(out, 360), 85)
        log.debug("演示素材 %d 生成完毕：%d 个块，%.0fms", k, len(blocks), (time.perf_counter() - t) * 1000)
        return info


def _device_label() -> str:
    try:
        hw = detect_hardware()
        gpu = hw.best_gpu
        if gpu and gpu.vendor == "nvidia" and "CUDAExecutionProvider" in hw.providers:
            return f"{gpu.name} · {DEVICE_LABELS['cuda']}"
        if gpu and "DmlExecutionProvider" in hw.providers:
            return f"{gpu.name} · {DEVICE_LABELS['dml']}"
        return f"CPU · {hw.cpu_name}"
    except Exception:  # noqa: BLE001
        return "CPU"


def _safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    return "".join("_" if c in bad or ord(c) < 32 else c for c in s).strip(" .") or "未命名"


# --------------------------------------------------------------------------------------
# 假任务 + 假引擎
# --------------------------------------------------------------------------------------
@dataclass
class _Job:
    id: str
    source: str
    title: str
    kind: str
    workdir: str
    out_path: str
    status: str
    message: str
    settings: dict
    pages: list[PageData]
    samples: list[int]
    created: float
    started: float = 0.0
    finished: float = 0.0
    done_times: collections.deque = field(default_factory=lambda: collections.deque(maxlen=48))
    edits: dict[int, list[TextBlock]] = field(default_factory=dict)
    fail_at: set[int] = field(default_factory=set)
    updated: float = 0.0


_DEMO_API_MODELS = (
    ("deepseek", ("deepseek-chat", "deepseek-reasoner")),
    ("siliconflow", ("Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3", "THUDM/glm-4-9b-chat")),
    ("openai", ("gpt-4o-mini", "gpt-4o")),
    ("googleapis", ("gemini-2.0-flash", "gemini-1.5-pro")),
    ("openrouter", ("deepseek/deepseek-chat", "google/gemini-2.0-flash-001")),
)


class DemoModels:
    """演示：本地大模型的假实现（接口同 hanhua.app.LocalModels）。推荐按本机真实硬件算，
    下载只推进假进度（大约 DOWNLOAD_S 秒完成，可取消、可续传），不联网、不写盘。"""

    DOWNLOAD_S = 14.0
    FAKE_SPEED = 41.5e6          # 显示用的「下载速度」（B/s）

    def __init__(self, speed: float = 1.0) -> None:
        from hanhua.llm.catalog import CATALOG, recommend

        self._catalog = CATALOG
        self._speed = max(0.05, float(speed))
        self._lock = threading.Lock()
        self._installed: set[str] = set()
        self._partial: dict[str, int] = {}
        self._custom: dict[str, dict] = {}
        try:
            self._rec = recommend(detect_hardware(), ["cuda", "vulkan", "cpu"])
        except Exception:  # noqa: BLE001
            log.debug("演示：硬件识别失败", exc_info=True)
            self._rec = None

    def _model(self, model_id: str) -> Any:
        return next((m for m in self._catalog if m.id == model_id), None)

    def installed_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._installed) + sorted(self._custom)

    def status(self) -> dict:
        from hanhua.app import LocalModels

        try:
            hw = LocalModels.hardware()
        except Exception:  # noqa: BLE001
            hw = {"gpu": "", "vram_gb": 0.0, "ram_gb": 16.0, "cpu": "CPU", "cpu_threads": 8}
        return {"hardware": hw, "recommend": self._rec.to_dict() if self._rec else None,
                "backends": ["cuda", "vulkan", "cpu"], "models": self._rows()}

    def _rows(self) -> list[dict]:
        rows = []
        with self._lock:
            for m in self._catalog:
                d = m.to_dict()
                d.update(installed=m.id in self._installed, recommended=bool(self._rec and self._rec.model.id == m.id),
                         partial_bytes=self._partial.get(m.id, 0))
                rows.append(d)
            rows += [dict(c) for c in self._custom.values()]
        return rows

    def info(self, model_id: str) -> Optional[dict]:
        return next((r for r in self._rows() if r.get("id") == model_id and not r.get("custom")), None)

    def download(self, model_id: str, progress: Callable[[int, int, float], None],
                 cancel: Optional[threading.Event] = None) -> str:
        m = self._model(model_id)
        if m is None:
            raise HanhuaError("没有这个模型")
        total = int(m.size)
        with self._lock:
            done = self._partial.get(m.id, 0)
        tick = 0.2
        rate = total / max(0.3, self.DOWNLOAD_S / self._speed)
        while done < total:
            if cancel is not None:
                if cancel.wait(tick):
                    with self._lock:
                        self._partial[m.id] = done
                    raise Cancelled("已取消下载")
            else:
                time.sleep(tick)
            done = min(total, done + int(rate * tick))
            progress(done, total, self.FAKE_SPEED * (0.85 + 0.3 * random.random()))
        with self._lock:
            self._installed.add(m.id)
            self._partial.pop(m.id, None)
        return f"demo://{m.file}"

    def delete(self, model_id: str) -> None:
        with self._lock:
            self._installed.discard(model_id)
            self._partial.pop(model_id, None)
            self._custom.pop(model_id, None)

    def import_file(self, path: str) -> dict:
        p = str(path or "")
        if not p.lower().endswith(".gguf"):
            raise HanhuaError("请选择 .gguf 格式的模型文件")
        name = os.path.basename(p)
        size = os.path.getsize(p) if os.path.isfile(p) else int(4.2e9)
        entry = {"id": "custom:" + name, "name": name, "path": p, "size_gb": round(size / 1024**3, 2),
                 "installed": True, "custom": True}
        with self._lock:
            self._custom[entry["id"]] = entry
        return {"id": entry["id"], "name": name, "path": p}

    @staticmethod
    def presets() -> list[dict]:
        from hanhua.llm.engine import API_PRESETS

        return [{"id": k, **v} for k, v in API_PRESETS.items()]

    def api_models(self, base_url: str, api_key: str) -> list[str]:
        time.sleep(0.4 / self._speed)
        if not api_key:
            raise HanhuaError("获取模型列表失败：请先填写 API Key（演示）")
        u = base_url.lower()
        return list(next((names for key, names in _DEMO_API_MODELS if key in u), ("demo-model-large", "demo-model-small")))


class DemoEngine:
    """与 Engine 同样的方法签名；所有方法线程安全，内部错误一律转成中文 HanhuaError。"""

    TICK_S = 0.35

    def __init__(
        self,
        store: Any,
        on_event: Callable[[dict], None],
        work_root: Optional[str] = None,
        components: Any = None,
        *,
        speed: float = 1.0,
        seed: bool = True,
        autostart: bool = True,
    ) -> None:
        self.store = store
        self._on_event = on_event
        self.root = os.path.abspath(work_root or os.path.join(data_dir(), "demo"))
        os.makedirs(self.root, exist_ok=True)
        self._speed = max(0.05, float(speed))
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}          # 插入顺序 = 队列顺序
        self._rng = random.Random(7)
        self._stop = threading.Event()
        self.samples = _SampleBank(self.root)
        self.label = _device_label()
        self.local_models = DemoModels(self._speed)     # Api.llm_* 用的假本地模型
        self._running = False
        if seed:
            self._seed()
            self._running = bool(autostart)
            threading.Thread(target=self._prebuild, name="demo-prebuild", daemon=True).start()
        self._thread = threading.Thread(target=self._loop, name="demo-engine", daemon=True)
        self._thread.start()

    # ---- 内部：任务 ------------------------------------------------------------------
    def _new_job(self, source: str, title: str, kind: str, n: int, status: str = "queued", message: str = "") -> _Job:
        jid = hashlib.sha1(f"{source}|demo".encode("utf-8")).hexdigest()[:12]
        offset = len(self._jobs) * 2
        job = _Job(
            id=jid, source=source, title=title, kind=kind, workdir=os.path.join(self.root, "jobs", jid), out_path="",
            status=status, message=message or self._default_message(status), settings=self._settings(),
            pages=[PageData(index=i, name=f"{i + 1:03d}.jpg") for i in range(n)],
            samples=[(i + offset) % max(1, len(self.samples)) for i in range(n)], created=time.time(),
        )
        job.updated = job.created
        self._jobs[jid] = job
        return job

    def _settings(self) -> dict:
        try:
            return self.store.get()
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _default_message(status: str) -> str:
        return {"queued": "排队中", "paused": "已暂停（进度已保存）", "canceled": "已取消", "done": "全部完成"}.get(status, "")

    def _seed(self) -> None:
        now = time.time()
        for title, kind, n, status, done, msg in _SEED_JOBS:
            source = os.path.join("D:\\漫画", title if kind == "folder" else f"{title}.{kind}")
            job = self._new_job(source, title, kind, n, status, msg)
            for p in job.pages[:done]:
                p.stage, p.version = "rendered", 1
            if status == "running":
                for off, st in enumerate(("cleaned", "translated", "recognized", "detected")):
                    if done + off < n:
                        job.pages[done + off].stage = st
                job.started = now - done * 1.8
                job.done_times.extend(now - (done - j) * 1.8 for j in range(done))
                job.fail_at = {_DEMO_FAIL_PAGE}
            elif status == "done":
                job.started, job.finished = now - 840, now - 588
                job.message = "全部完成，用时 4 分 12 秒"
                job.out_path = self._output_dir(job)
            elif status == "paused":
                job.started = now - 3600
        self._emit_all()

    def _prebuild(self) -> None:
        """后台把素材都生成好（每张 0.1~0.3s），之后取图零等待；演示里"已完成"的书顺便写出成品文件夹。"""
        for k in range(len(self.samples)):
            if self._stop.is_set():
                return
            try:
                self.samples.get(k)
            except Exception:  # noqa: BLE001
                log.warning("演示素材 %d 生成失败", k, exc_info=True)
        with self._lock:
            done_jobs = [j for j in self._jobs.values() if j.status == "done" and j.out_path]
        for j in done_jobs:
            try:
                self._write_output(j)
            except Exception:  # noqa: BLE001
                log.warning("演示成品写出失败", exc_info=True)

    def _job(self, job_id: str) -> _Job:
        job = self._jobs.get(str(job_id))
        if job is None:
            raise HanhuaError("找不到这个任务（可能已被删除）")
        return job

    @staticmethod
    def _page(job: _Job, index: Any) -> PageData:
        try:
            i = int(index)
        except (TypeError, ValueError):
            raise HanhuaError("页码不对") from None
        if not 0 <= i < len(job.pages):
            raise HanhuaError(f"页码超出范围（这本书共 {len(job.pages)} 页）")
        return job.pages[i]

    def _summary(self, job: _Job) -> dict:
        counts = {s: 0 for s in STAGES}
        counts["error"] = 0
        for p in job.pages:
            counts[p.stage if p.stage in counts else "error"] += 1
        total, done, failed = len(job.pages), counts["rendered"], counts["error"]
        now = time.time()
        speed = 0.0
        if job.status == "running":
            recent = [t for t in job.done_times if now - t <= 90]
            if len(recent) >= 2 and recent[-1] > recent[0]:
                speed = (len(recent) - 1) / (recent[-1] - recent[0]) * 60
        elif job.status == "done" and job.finished > job.started:
            speed = done / (job.finished - job.started) * 60
        remaining = total - done - failed
        eta = int(remaining / speed * 60) if job.status == "running" and speed > 0 and remaining > 0 else None
        return {
            "id": job.id, "title": job.title, "source": job.source, "kind": job.kind, "status": job.status,
            "message": job.message, "total": total, "done": done, "failed": failed, "stage_counts": counts,
            "speed_ppm": round(speed, 1), "eta_s": eta, "out_path": job.out_path, "cover": 0, "updated": job.updated,
        }

    def _emit(self, ev: dict) -> None:
        try:
            self._on_event(ev)
        except Exception:  # noqa: BLE001
            log.exception("演示引擎：事件回调出错")

    def _emit_all(self) -> None:
        with self._lock:
            evs = [{"type": "job", "job": self._summary(j)} for j in self._jobs.values()]
        for e in evs:
            self._emit(e)
        self._emit_engine()

    def _emit_engine(self) -> None:
        self._emit({"type": "engine", "running": self._running, "device": self.label})

    def _touch(self, job: _Job) -> dict:
        job.updated = time.time()
        return {"type": "job", "job": self._summary(job)}

    # ---- 内部：流水线模拟 ------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(self.TICK_S / self._speed):
            try:
                with self._lock:
                    events = self._tick()
            except Exception:  # noqa: BLE001
                log.exception("演示引擎出错")
                events = []
            for e in events:
                self._emit(e)

    def _current(self) -> Optional[_Job]:
        for j in self._jobs.values():
            if j.status == "running":
                return j
        for j in self._jobs.values():
            if j.status == "queued":
                j.status, j.started = "running", j.started or time.time()
                return j
        return None

    def _tick(self) -> list[dict]:
        if not self._running:
            return []
        job = self._current()
        if job is None:
            return []
        events: list[dict] = []
        in_flight, prev_rank = 0, len(STAGES) - 1
        first_open: Optional[int] = None
        for p in job.pages:
            if p.stage == "error":
                continue
            rank = STAGES.index(p.stage) if p.stage in STAGES else 0
            if rank >= len(STAGES) - 1:
                prev_rank = rank
                continue
            if first_open is None:
                first_open = p.index
            if in_flight >= _IN_FLIGHT:
                break
            in_flight += 1
            if (rank + 1 < prev_rank or prev_rank == len(STAGES) - 1) and self._rng.random() < 0.6:
                rank += 1
                if p.index in job.fail_at and STAGES[rank] == "translated":
                    p.stage, p.error = "error", f"第 {p.index + 1} 页图片损坏，导出时将使用原图"
                    events.append({"type": "page", "job_id": job.id, "index": p.index, "stage": "error", "version": p.version})
                    events.append({"type": "log", "level": "warning", "message": f"《{job.title}》{p.error}"})
                    continue
                p.stage = STAGES[rank]
                if p.stage == "rendered":
                    p.version += 1
                    job.done_times.append(time.time())
                events.append({"type": "page", "job_id": job.id, "index": p.index, "stage": p.stage, "version": p.version})
            prev_rank = rank
        if first_open is None:          # 全部页面都结束了
            self._finish(job, events)
        else:
            stage_name = {"pending": "检测", "detected": "识别", "recognized": "翻译", "translated": "去字", "cleaned": "嵌字"}
            st = job.pages[first_open].stage
            job.message = f"正在处理第 {first_open + 1} 页 · {stage_name.get(st, '处理')}"
        if events or first_open is None:
            events.append(self._touch(job))
        return events

    def _finish(self, job: _Job, events: list[dict]) -> None:
        job.status, job.finished = "done", time.time()
        failed = sum(1 for p in job.pages if p.stage == "error")
        mins, secs = divmod(int(job.finished - job.started), 60)
        job.message = f"全部完成，用时 {mins} 分 {secs} 秒" + (f"（{failed} 页出错，导出时使用原图）" if failed else "")
        job.out_path = self._output_dir(job)
        events.append({"type": "log", "level": "info", "message": f"《{job.title}》汉化完成"})
        threading.Thread(target=self._write_output_safe, args=(job,), daemon=True).start()

    # ---- 内部：文件 ------------------------------------------------------------------
    def _output_dir(self, job: _Job) -> str:
        return os.path.join(self.root, "output", f"{_safe_name(job.title)}_汉化")

    def _override_path(self, job: _Job, index: int, kind: str) -> str:
        sub, ext = {"clean": ("clean", "png"), "out": ("out", "jpg"), "thumb": ("thumbs", "jpg")}[kind]
        return os.path.join(job.workdir, sub, f"{index + 1:04d}.{ext}")

    def _write_output_safe(self, job: _Job) -> None:
        try:
            self._write_output(job)
        except Exception:  # noqa: BLE001
            log.warning("演示成品写出失败：%s", job.title, exc_info=True)

    def _write_output(self, job: _Job) -> str:
        out_dir = self._output_dir(job)
        os.makedirs(out_dir, exist_ok=True)
        with self._lock:
            pages = [(p.index, p.stage) for p in job.pages]
        for index, stage in pages:
            src = self.page_image_path(job.id, index, "out") if stage == "rendered" else self.page_image_path(job.id, index, "src")
            if src:
                ext = os.path.splitext(src)[1] or ".jpg"
                shutil.copyfile(src, os.path.join(out_dir, f"{index + 1:03d}{ext}"))
        return out_dir

    def _blocks_for(self, job: _Job, index: int) -> list[TextBlock]:
        with self._lock:
            edits = job.edits.get(index)
            k = job.samples[index]
        if edits is not None:
            return copy.deepcopy(edits)
        return copy.deepcopy(self.samples.get(k)["blocks"])

    def _render_override(self, job: _Job, index: int, blocks: list[TextBlock]) -> None:
        img = imageio.read(self.samples.source(job.samples[index]))
        patch, out = render_page(img, blocks, self.samples.fonts)
        imageio.write_bytes_atomic(self._override_path(job, index, "clean"), imageio.encode(patch, "png"))
        imageio.write(self._override_path(job, index, "out"), out, 90)
        imageio.write(self._override_path(job, index, "thumb"), imageio.thumbnail(out, 360), 85)

    def _commit_page(self, job: _Job, index: int, blocks: Optional[list[TextBlock]]) -> dict:
        """保存（或清除）人工修改并重渲染本页，版本号 +1，推送事件，返回最新页面数据。"""
        if blocks is not None:
            self._render_override(job, index, blocks)
        with self._lock:
            p = job.pages[index]
            if blocks is None:
                job.edits.pop(index, None)
                for kind in ("clean", "out", "thumb"):
                    try:
                        os.remove(self._override_path(job, index, kind))
                    except OSError:
                        pass
            else:
                job.edits[index] = copy.deepcopy(blocks)
            if p.stage != "error":
                p.stage = "rendered"
            p.version += 1
            evs = [{"type": "page", "job_id": job.id, "index": index, "stage": p.stage, "version": p.version}, self._touch(job)]
        for e in evs:
            self._emit(e)
        return self.get_page(job.id, index)

    # ---- 公开接口（与 Engine 一致）---------------------------------------------------
    def add_paths(self, paths: list[str]) -> list[dict]:
        out: list[dict] = []
        with self._lock:
            for raw in paths or []:
                if not isinstance(raw, str) or not raw.strip():
                    continue
                src = os.path.abspath(raw.strip())
                jid = hashlib.sha1(f"{src}|demo".encode("utf-8")).hexdigest()[:12]
                job = self._jobs.get(jid)
                if job is not None:                  # 同一本书再拖进来 = 续跑
                    if job.status in ("paused", "canceled", "error"):
                        job.status, job.message = "queued", "排队中（继续上次的进度）"
                    out.append(self._touch(job)["job"])
                    continue
                base = os.path.basename(src.rstrip("\\/")) or src
                stem, ext = os.path.splitext(base)
                kind = _KINDS.get(ext.lower(), "images" if imageio.is_image_name(base) else "folder")
                n = 1 if kind == "images" else self._rng.randint(16, 40)
                if os.path.isdir(src):
                    try:
                        n = max(1, sum(1 for f in os.listdir(src) if imageio.is_image_name(f)))
                    except OSError:
                        pass
                job = self._new_job(src, stem if kind not in ("folder",) else base, kind, n)
                out.append(self._touch(job)["job"])
        for j in out:
            self._emit({"type": "job", "job": j})
        return out

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [self._summary(j) for j in self._jobs.values()]

    def list_pages(self, job_id: str) -> list[dict]:
        with self._lock:
            job = self._job(job_id)
            return [{"index": p.index, "name": p.name, "stage": p.stage, "version": p.version, "error": p.error} for p in job.pages]

    def start(self) -> None:
        with self._lock:
            self._running = True
        self._emit_engine()

    def pause(self, job_id: Optional[str] = None) -> None:
        with self._lock:
            targets = [self._job(job_id)] if job_id else list(self._jobs.values())
            evs = []
            for j in targets:
                if j.status in ("running", "queued"):
                    j.status, j.message = "paused", "已暂停（进度已保存）"
                    evs.append(self._touch(j))
            if not job_id:
                self._running = False
        for e in evs:
            self._emit(e)
        self._emit_engine()

    def resume(self, job_id: Optional[str] = None) -> None:
        with self._lock:
            targets = [self._job(job_id)] if job_id else list(self._jobs.values())
            evs = []
            for j in targets:
                if j.status in ("paused", "canceled") and j.pages:
                    j.status, j.message = "queued", "排队中"
                    evs.append(self._touch(j))
            self._running = True
        for e in evs:
            self._emit(e)
        self._emit_engine()

    def cancel(self, job_id: str) -> None:
        with self._lock:
            j = self._job(job_id)
            if j.status in ("running", "queued", "paused"):
                j.status, j.message = "canceled", "已取消（已完成的页面会保留）"
            ev = self._touch(j)
        self._emit(ev)

    def remove(self, job_id: str, delete_files: bool = False) -> None:
        with self._lock:
            j = self._job(job_id)
            del self._jobs[j.id]
        if delete_files:
            wd = os.path.abspath(j.workdir)
            if os.path.commonpath([wd, self.root]) == self.root and os.path.isdir(wd):
                shutil.rmtree(wd, ignore_errors=True)

    def retry_failed(self, job_id: str) -> None:
        with self._lock:
            j = self._job(job_id)
            if not j.pages:
                raise HanhuaError("这个文件读不出页面，请检查文件是否完整后重新添加")
            for p in j.pages:
                if p.stage == "error":
                    p.stage, p.error = "pending", ""
            j.fail_at.clear()
            if j.status in ("error", "done", "canceled", "paused"):
                j.status, j.message = "queued", "排队中（重试失败的页面）"
            self._running = True
            ev = self._touch(j)
        self._emit(ev)
        self._emit_engine()

    def get_page(self, job_id: str, index: int) -> dict:
        with self._lock:
            job = self._job(job_id)
            p = self._page(job, index)
            d = p.to_dict()
            stage, k = p.stage, job.samples[p.index]
        info = self.samples.get(k)
        d["width"], d["height"] = info["w"], info["h"]
        d["job_id"] = job.id
        if stage_at_least(stage, "detected") or stage == "error":
            blocks = self._blocks_for(job, p.index)
            for b in blocks:
                if not stage_at_least(stage, "recognized"):
                    b.src_text = ""
                if not stage_at_least(stage, "translated"):
                    b.translation = ""
                if stage != "rendered":
                    b.render_box, b.render_font_size = None, None
            d["blocks"] = [b.to_dict() for b in blocks]
            d["bubbles"] = [list(b.bubble_box) for b in blocks if b.bubble_box]
        return d

    def update_page(self, job_id: str, index: int, blocks: list[dict]) -> dict:
        with self._lock:
            job = self._job(job_id)
            p = self._page(job, index)
            if not stage_at_least(p.stage, "translated"):
                raise HanhuaError("这一页还没翻译完，稍后再来校对")
        if not isinstance(blocks, list):
            raise HanhuaError("文字块数据格式不对")
        try:
            tbs = [TextBlock.from_dict(b) for b in blocks if isinstance(b, dict)]
        except Exception as e:  # noqa: BLE001
            raise HanhuaError(f"文字块数据格式不对：{e}") from e
        return self._commit_page(job, p.index, tbs)

    def retranslate_page(self, job_id: str, index: int) -> dict:
        with self._lock:
            job = self._job(job_id)
            p = self._page(job, index)
            if not stage_at_least(p.stage, "recognized"):
                raise HanhuaError("这一页还没识别完原文，稍后再试")
        time.sleep(0.5 / self._speed)          # 模拟联网翻译
        blocks = self._blocks_for(job, p.index)
        for b in blocks:
            if b.skip or not b.src_text:
                continue
            if b.translation in _DST_LINES:
                b.translation = _ALT_LINES[_DST_LINES.index(b.translation)]
            elif b.translation in _ALT_LINES:
                b.translation = _DST_LINES[_ALT_LINES.index(b.translation)]
        return self._commit_page(job, p.index, blocks)

    def redetect_page(self, job_id: str, index: int) -> dict:
        with self._lock:
            job = self._job(job_id)
            p = self._page(job, index)
        time.sleep(0.6 / self._speed)          # 模拟检测 + OCR + 翻译
        return self._commit_page(job, p.index, None)

    def rerender_all(self, job_id: str) -> None:
        with self._lock:
            job = self._job(job_id)
            evs = []
            for p in job.pages:
                if p.stage == "rendered":
                    p.version += 1
                    evs.append({"type": "page", "job_id": job.id, "index": p.index, "stage": p.stage, "version": p.version})
            evs.append(self._touch(job))
        for e in evs:
            self._emit(e)

    def export(self, job_id: str) -> str:
        with self._lock:
            job = self._job(job_id)
            if not any(p.stage == "rendered" for p in job.pages):
                raise HanhuaError("还没有完成的页面，暂时无法导出")
        time.sleep(0.4 / self._speed)
        path = self._write_output(job)
        with self._lock:
            job.out_path = path
            ev = self._touch(job)
        self._emit(ev)
        return path

    def page_image_path(self, job_id: str, index: int, kind: str) -> Optional[str]:
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None or not 0 <= int(index) < len(job.pages) or kind not in ("src", "clean", "out", "thumb"):
                return None
            p = job.pages[int(index)]
            stage, k, edited = p.stage, job.samples[p.index], p.index in job.edits
        if kind == "src":
            return self.samples.source(k)
        if not stage_at_least(stage, "cleaned" if kind == "clean" else "rendered"):
            return None
        if edited:
            path = self._override_path(job, int(index), kind)
            if os.path.isfile(path):
                return path
        return self.samples.get(k)[kind]

    def test_engine(self, engine_id: str) -> tuple[bool, str]:
        """演示：模拟测试结果（真实程序由 Api 调 TranslationService.test_engine）。"""
        time.sleep(0.35 / self._speed)
        if engine_id == "local":
            if not self.local_models.installed_ids():
                return False, "本地翻译模型还没有下载（演示：在上面的模型表里点「下载」）"
            time.sleep(0.8 / self._speed)            # 模拟启动本地引擎 + 翻一句
            return True, "可用（演示，1260 毫秒）：お疲れ様です → 辛苦了"
        if engine_id == "api":
            cfg = self._settings()["translate"].get("api") or {}
            if not str(cfg.get("api_key") or "").strip():
                return False, "还没有填写 API Key"
            return True, f"可用（演示，820 毫秒）：{cfg.get('model') or '默认模型'} 已连通"
        if engine_id in ("llm", "ollama"):
            return False, "连接不上 http://127.0.0.1:11434（演示：请确认 Ollama / LM Studio 已启动）"
        if engine_id in ("edge", "tencent"):
            return False, "暂不可用：该服务 2026-10 实测已失效"
        return True, "可以使用（演示模式模拟结果）"

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            self._running = False
        self._emit_engine()

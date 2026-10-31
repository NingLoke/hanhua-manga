"""设置：默认值、读写、深度合并、变更通知。

设置是一个嵌套 dict（方便和网页界面直接交换 JSON）。所有模块都通过 key 路径读取，
例如 settings["translate"]["engines"]。新增设置项只需在 DEFAULTS 里加默认值，
旧版本保存的文件会自动补齐。
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
from typing import Any, Callable

from .runtime import data_dir

log = logging.getLogger("hanhua.config")

SCHEMA_VERSION = 1

DEFAULTS: dict[str, Any] = {
    "schema": SCHEMA_VERSION,
    "general": {
        "theme": "auto",               # auto | light | dark
        "auto_resume": True,           # 启动时自动恢复上次没跑完的任务
        "notify_done": True,           # 完成时系统通知
    },
    "hardware": {
        "device": "auto",              # auto | cuda | dml | cpu
        "profile": "balanced",         # fast | balanced | quality
        "cpu_threads": 0,              # 0 = 自动
        "gpu_id": 0,
    },
    "source": {
        "lang": "ja",                  # ja | ko | en | zh | auto
        "reading_order": "ltr",        # 竖排列与气泡阅读方向；auto 兼容旧任务
    },
    "translate": {
        "target": "zh-Hans",           # zh-Hans 简体 | zh-Hant 繁体
        "local": {                     # ① 默认首选：软件自带 llama.cpp + 按电脑配置推荐的本地模型（离线、不限次数）
            "enabled": True,
            "model": "auto",           # auto = 按配置推荐；或 hanhua.llm.CATALOG 的 id / custom:<文件名>
            "backend": "auto",         # auto | cuda | vulkan | cpu
            "gpu_layers": -1,          # -1 = 自动（显卡全放 / CPU 为 0）
            "ctx": 8192,
        },
        "api": {                       # ② 第二选择：填 Key 的大模型 API（OpenAI 兼容）
            "enabled": False,
            "provider": "deepseek",    # deepseek | siliconflow | openai | gemini | openrouter | custom
            "base_url": "",            # 空 = 用 provider 预设地址
            "api_key": "",
            "model": "",
        },
        "online_fallback": True,       # ③ 前两者不可用（模型没下载 / 没填 Key）时用免费在线引擎兜底
        "engines": ["google"],         # 免费在线引擎顺序（edge / tencent 2026-10 实测失效，保留实现不进默认）
        "llm": {                       # 旧版：客户自己装的 Ollama / LM Studio（界面里归入「API 服务 → 自定义」）
            "enabled": False,
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen3:8b",
            "api_key": "",
            "temperature": 0.3,
        },
        "timeout": 15,                 # 单次请求超时（秒）
        "max_retries": 3,
        "concurrency": 3,              # 同时进行的翻译请求数
        "glossary": [],                # [{"src": "原文", "dst": "译文", "note": ""}]
        "sfx_mode": "translate",       # translate 翻译替换 | keep 保留原样 | erase 只擦除
        "translate_free_text": True,   # 气泡外文字（旁白 / 招牌）也翻译
    },
    "typeset": {
        "font_dialog": "noto-sans-sc-bold",     # 对白
        "font_free": "noto-sans-sc-black",      # 气泡外旁白
        "font_sfx": "zcool-kuaile",             # 拟声词
        "direction": "v",              # auto | v 竖排 | h 横排
        "column_order": "ltr",         # 竖排列顺序：rtl | ltr
        "min_font_px": 0,              # 0 = 按页面分辨率自动
        "max_font_px": 0,
        "line_spacing": 1.12,
        "letter_spacing": 0.04,
        "stroke_free": 0.14,           # 气泡外文字描边宽度（相对字号）
        "punct_style": "cjk",          # cjk 「」 | western “”
        "keep_original_color": True,
        "bubble_margin": 0.12,         # 气泡内留白（相对气泡短边）
    },
    "inpaint": {
        "mode": "auto",                # auto 纯色气泡直接填色 + 复杂背景 LaMa | lama 全部 LaMa | fill 全部填色
        "max_side": 0,                 # 0 = 跟随画质档位
    },
    "output": {
        "format": "folder",            # same | folder | cbz | zip | pdf | epub
        "image_format": "same",        # same | jpg | png | webp
        "jpeg_quality": 92,
        "webp_quality": 90,
        "dir": "",                     # 空 = 和源文件同目录
        "suffix": "_汉化",
        "keep_workdir": True,          # 保留中间结果（可回到编辑器再改）
        "work_dir": "",                # 空 = 数据目录/jobs
    },
}

ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("general", "theme"): ("auto", "light", "dark"),
    ("hardware", "device"): ("auto", "cuda", "dml", "cpu"),
    ("hardware", "profile"): ("fast", "balanced", "quality"),
    ("source", "lang"): ("ja", "ko", "en", "zh", "auto"),
    ("source", "reading_order"): ("auto", "rtl", "ltr"),
    ("translate", "target"): ("zh-Hans", "zh-Hant"),
    ("translate", "sfx_mode"): ("translate", "keep", "erase"),
    ("typeset", "direction"): ("auto", "v", "h"),
    ("typeset", "column_order"): ("rtl", "ltr"),
    ("typeset", "punct_style"): ("cjk", "western"),
    ("inpaint", "mode"): ("auto", "lama", "fill"),
    ("output", "format"): ("same", "folder", "cbz", "zip", "pdf", "epub"),
    ("output", "image_format"): ("same", "jpg", "png", "webp"),
}


def deep_merge(base: dict, over: dict) -> dict:
    """over 覆盖 base（只接受 base 里已有的 key 和同类型值，防止脏数据写坏设置）。"""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            continue
        bv = out[k]
        if isinstance(bv, dict) and isinstance(v, dict):
            out[k] = deep_merge(bv, v)
        elif isinstance(bv, bool):
            out[k] = bool(v)
        elif isinstance(bv, (int, float)) and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = type(bv)(v) if isinstance(bv, int) and float(v).is_integer() else float(v)
        elif isinstance(bv, str) and isinstance(v, str):
            out[k] = v
        elif isinstance(bv, list) and isinstance(v, list):
            out[k] = copy.deepcopy(v)
    return out


def column_order(settings: dict) -> str:
    """用户的阅读方向同时决定竖排列顺序；旧任务 auto 时兼容原列设置。"""
    source = settings.get("source") if isinstance(settings.get("source"), dict) else {}
    selected = source.get("reading_order")
    if selected in ("ltr", "rtl"):
        return selected
    typeset = settings.get("typeset") if isinstance(settings.get("typeset"), dict) else {}
    return "rtl" if typeset.get("column_order") == "rtl" else "ltr"


def validate(s: dict) -> dict:
    for (sec, key), allowed in ENUMS.items():
        if s[sec][key] not in allowed:
            log.warning("设置 %s.%s=%r 无效，恢复默认", sec, key, s[sec][key])
            s[sec][key] = DEFAULTS[sec][key]
    s["translate"]["engines"] = [e for e in s["translate"]["engines"] if isinstance(e, str)] or list(DEFAULTS["translate"]["engines"])
    s["output"]["jpeg_quality"] = max(50, min(100, int(s["output"]["jpeg_quality"])))
    s["output"]["webp_quality"] = max(50, min(100, int(s["output"]["webp_quality"])))
    s["translate"]["concurrency"] = max(1, min(8, int(s["translate"]["concurrency"])))
    s["typeset"]["column_order"] = column_order(s)
    return s


def settings_path() -> str:
    return os.path.join(data_dir(), "settings.json")


class SettingsStore:
    """线程安全的设置存取。get() 返回深拷贝快照，任务开始时用快照，跑到一半改设置不会影响它。"""

    def __init__(self, path: str | None = None):
        self.path = path or settings_path()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        self._data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            return copy.deepcopy(DEFAULTS)
        except Exception as e:  # noqa: BLE001
            log.warning("设置文件损坏，已恢复默认（原文件备份为 .bad）: %s", e)
            try:
                os.replace(self.path, self.path + ".bad")
            except OSError:
                pass
            return copy.deepcopy(DEFAULTS)
        return validate(deep_merge(DEFAULTS, raw))

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def update(self, patch: dict) -> dict:
        """部分更新（深度合并）并立即保存。返回新的完整设置。"""
        patch = copy.deepcopy(patch) if isinstance(patch, dict) else {}
        source = patch.get("source") if isinstance(patch.get("source"), dict) else {}
        typeset = patch.get("typeset") if isinstance(patch.get("typeset"), dict) else {}
        selected = source.get("reading_order")
        legacy = typeset.get("column_order")
        if selected in ("ltr", "rtl"):
            patch["typeset"] = dict(typeset, column_order=selected)
        elif legacy in ("ltr", "rtl"):
            patch["source"] = dict(source, reading_order=legacy)
        with self._lock:
            self._data = validate(deep_merge(self._data, patch))
            self._save_locked()
            snap = copy.deepcopy(self._data)
        for cb in list(self._listeners):
            try:
                cb(snap)
            except Exception:  # noqa: BLE001
                log.exception("设置监听器出错")
        return snap

    def reset(self) -> dict:
        with self._lock:
            self._data = copy.deepcopy(DEFAULTS)
            self._save_locked()
            return copy.deepcopy(self._data)

    def on_change(self, cb: Callable[[dict], None]) -> None:
        self._listeners.append(cb)

    def _save_locked(self) -> None:
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

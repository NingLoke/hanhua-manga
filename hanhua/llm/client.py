"""用本地 llama-server（或任意 OpenAI 兼容 API）做整页漫画翻译。

整页台词 + 上文 + 术语表一次请求，用 JSON Schema 约束输出（llama.cpp 语法约束解码），
保证返回条数与编号和输入一一对应；对不上时自动拆半重试。
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

from ..types import HanhuaError

log = logging.getLogger("hanhua.llm")

LANG_NAMES = {"ja": "日语", "ko": "韩语", "en": "英语", "zh": "中文", "auto": "原文语言"}
TARGET_NAMES = {"zh-Hans": "简体中文", "zh-Hant": "繁体中文"}

SYSTEM_PROMPT = """你是资深漫画汉化组的翻译，负责把漫画台词从{src}翻译成{tgt}。
要求：
1. 先理解整句意思和说话场景再翻译，宁可意译也不要逐字硬翻；口语化，贴合角色语气，不要书面腔、翻译腔；
2. 保留情绪和语气词（啊、呢、吧、嘛、哦、呀……）；只有原文有结巴（如「マ…マネージャー」）才译成结巴（「经、经理」），原文没有的不要添加；
3. 拟声词、拟态词译成自然的中文拟声词；
4. 人名、称呼、专有名词严格按术语表；术语表里没有的人名音译，并与上文保持一致；
5. 标点用中文全角：省略号用「……」，感叹加疑问用「！？」；对白句末的句号可以省略；
6. 只输出译文本身，不要解释、注释或附带原文；输出条目的数量和 id 必须与输入完全一致。"""

# 少样本示范（与测试素材无关的通用台词），帮小模型抓住漫画口语和结巴的处理方式
FEW_SHOT_IN = {"上文": [], "术语表": {"先輩": "前辈"}, "台词": [
    {"id": 1, "原文": "べ、別にあんたのためじゃないんだからね！"},
    {"id": 2, "原文": "先輩、ちょっと待ってくださいよ～"},
    {"id": 3, "原文": "お前、今日は何曜日だと思ってるんだ！"},
    {"id": 4, "原文": "ゴゴゴゴ"},
]}
FEW_SHOT_OUT = {"translations": [
    {"id": 1, "text": "我、我才不是为了你呢！"},
    {"id": 2, "text": "前辈，等一下嘛～"},
    {"id": 3, "text": "你以为今天星期几啊！"},
    {"id": 4, "text": "轰轰轰轰"},
]}

_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["translations"],
}


class ChatTranslator:
    """OpenAI 兼容 /chat/completions 客户端。本地 llama-server 不需要 key；API 服务传 api_key。"""

    def __init__(self, base_url: str, model: str = "local", api_key: str = "", timeout: float = 180.0,
                 temperature: float = 0.3, json_schema: bool = True, extra: Optional[dict] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.json_schema = json_schema          # llama.cpp 支持语法约束；部分云端 API 只支持 json_object
        self.extra = extra or {}

    # ----------------------------------------------------------------------------------
    def translate(self, lines: list[str], src_lang: str = "ja", target: str = "zh-Hans",
                  glossary: Optional[list[dict]] = None, context: Optional[list[str]] = None) -> list[str]:
        if not lines:
            return []
        try:
            return self._call(lines, src_lang, target, glossary, context)
        except _Mismatch:
            if len(lines) == 1:
                raise HanhuaError("本地模型返回格式异常")
            mid = len(lines) // 2                 # 条数对不上：拆半再试（上下文保留）
            a = self.translate(lines[:mid], src_lang, target, glossary, context)
            b = self.translate(lines[mid:], src_lang, target, glossary, (context or []) + a)
            return a + b

    def _call(self, lines: list[str], src_lang: str, target: str, glossary: Optional[list[dict]],
              context: Optional[list[str]]) -> list[str]:
        payload_in = {
            "上文": (context or [])[-12:],          # 本批之前的几句原文，帮模型理解场景和人称
            "术语表": {g["src"]: g["dst"] for g in (glossary or []) if g.get("src") and g.get("dst")},
            "台词": [{"id": i + 1, "原文": t} for i, t in enumerate(lines)],
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(src=LANG_NAMES.get(src_lang, "原文语言"),
                                                               tgt=TARGET_NAMES.get(target, "简体中文"))},
            *([{"role": "user", "content": json.dumps(FEW_SHOT_IN, ensure_ascii=False)},
               {"role": "assistant", "content": json.dumps(FEW_SHOT_OUT, ensure_ascii=False)}]
              if src_lang in ("ja", "auto") and target == "zh-Hans" else []),
            {"role": "user", "content": json.dumps(payload_in, ensure_ascii=False)},
        ]
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": min(4096, 64 + 4 * sum(len(t) for t in lines) + 24 * len(lines)),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "translations", "schema": _SCHEMA}}
        else:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra)
        data = self._post("/chat/completions", body)
        try:
            content = data["choices"][0]["message"]["content"] or ""
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            items = json.loads(content)["translations"]
            by_id = {int(it["id"]): str(it["text"]).strip() for it in items}
        except Exception as e:  # noqa: BLE001
            log.warning("模型输出解析失败：%s", e)
            raise _Mismatch() from e
        if any(i + 1 not in by_id for i in range(len(lines))):
            raise _Mismatch()
        return [by_id[i + 1] for i in range(len(lines))]

    def _post(self, path: str, body: dict, retries: int = 2) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(self.base_url + path, data=raw, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                if e.code in (401, 403):
                    raise HanhuaError("API Key 无效或没有权限，请在设置里检查") from e
                if e.code == 400 and "response_format" in msg and self.json_schema:
                    self.json_schema = False      # 云端不支持 json_schema：降级为 json_object 再试
                    body["response_format"] = {"type": "json_object"}
                    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                    continue
                last = HanhuaError(f"翻译服务返回错误 {e.code}：{msg}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
            time.sleep(1.5 * (attempt + 1))
        raise HanhuaError(f"无法连接翻译模型：{last}")


class _Mismatch(Exception):
    pass

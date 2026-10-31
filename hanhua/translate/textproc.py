"""翻译前后的文本处理（纯函数，线程安全）。

- 原文规范化：manga-ocr 输出的全角 ASCII（．．．、！？、ＯＫ）等统一成翻译引擎更容易理解的形式；
- 口吃：「マ…マネージャー」先去掉口吃再翻译，译完按译文首字补回「经、经理」；
- 纯标点 / 符号块（…、！？）直接搬运，不浪费请求；
- 中文后处理：全角标点、成对省略号、破折号、去 CJK 间空格、引号风格、机翻多余句号；
- 术语表：翻译前把术语换成 {0} 占位符（2026-10 实测 google / edge 都原样保留；直接把中文译名
  塞进日文句子反而会被改写，如「经纪人が来た」→「一群人来了」），翻译后还原并校正。
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

log = logging.getLogger("hanhua.translate")

# ---- 字符类 -------------------------------------------------------------------------------------
_HIRA = "ぁ-ゖゝ-ゟ"
_KATA = "ァ-ヺヽ-ヿㇰ-ㇿｦ-ｯｱ-ﾝ"
_HANGUL = "가-힣ᄀ-ᇿ㄰-㆏"
_HAN = "㐀-䶿一-鿿豈-﫿"
_SMALL_KANA = "ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ"
# 中日文字 + 全角标点：它们之间的空格都是机翻 / OCR 带进来的，应当去掉
_CJKP = (f"{_HAN}{_HIRA}{_KATA}　-〿・ー！-／：-＠"
         "［-｀｛-･—…‘’“”")

_KANA_RE = re.compile(f"[{_HIRA}{_KATA}]")
_HANGUL_RE = re.compile(f"[{_HANGUL}]")
_HAN_RE = re.compile(f"[{_HAN}]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# 全角数字 / 字母 → 半角（manga-ocr 用 jaconv.h2z 把 ASCII 变成了全角）
_Z2H_ALNUM = {c: c - 0xFEE0 for c in [*range(0xFF10, 0xFF1A), *range(0xFF21, 0xFF3B), *range(0xFF41, 0xFF5B)]}

_ELLIPSIS_SRC_JA = re.compile(r"[.．。・･‥…]{2,}|[‥…]")
_ELLIPSIS_SRC_EN = re.compile(r"\.{3,}|…+")
_ELLIPSIS_OUT = re.compile(r"\.{2,}|…+|‥+|。{3,}|・{2,}|·{2,}|．{2,}")
_DASH_OUT = re.compile(rf"[—―─━]+|-{{2,}}|－{{2,}}|(?<=[{_HAN}！？…])ー+")
_SPACE_BETWEEN_CJK = re.compile(rf"(?<=[{_CJKP}])[ \t　]+(?=[{_CJKP}])")
_PUNCT_EXTRA = set("ー〜～・ｰ")
# 首尾的装饰符号（翻译引擎经常吞掉），翻译前摘下、翻译后补回
_AFFIX_CHARS = "♡♥❤💕💓💗♪♫♬☆★✨※💢💦"
_AFFIX_LEAD_RE = re.compile(f"^[{_AFFIX_CHARS}]+")
_AFFIX_TRAIL_RE = re.compile(f"[{_AFFIX_CHARS}]+$")
# 句首单字后跟逗号 + 汉字时，这些字更像感叹词（あ、本当だ）而不是口吃（ち、違う）
_INTERJECTIONS = set("あえおうんまさねよやはへふ")


# ---- 字符判断 -----------------------------------------------------------------------------------
def has_kana(s: str) -> bool:
    return bool(s) and _KANA_RE.search(s) is not None


def has_hangul(s: str) -> bool:
    return bool(s) and _HANGUL_RE.search(s) is not None


def has_han(s: str) -> bool:
    return bool(s) and _HAN_RE.search(s) is not None


def has_latin(s: str) -> bool:
    return bool(s) and _LATIN_RE.search(s) is not None


def kana_fold(s: str) -> str:
    """片假名 → 平假名（只处理全角片假名区）。"""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def effective_lang(text: str, lang: str) -> str:
    """把 auto 落实成具体语种（只用于选择规范化规则，不影响发给引擎的语种码）。"""
    if lang in ("ja", "ko", "en", "zh"):
        return lang
    if has_kana(text):
        return "ja"
    if has_hangul(text):
        return "ko"
    if has_han(text):
        return "zh"
    if has_latin(text):
        return "en"
    return "ja"


def _strip_punct(s: str) -> str:
    """去掉标点 / 符号 / 空白，只留文字（用于比较“译文是不是原样返回”）。"""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LN")


def same_text(a: str, b: str) -> bool:
    return _strip_punct(kana_fold(a or "")) == _strip_punct(kana_fold(b or ""))


# ---- 原文规范化 ---------------------------------------------------------------------------------
def _sentence_case(s: str) -> str:
    """英文漫画 OCR 多是全大写，转成句首大写，机翻质量明显更好。"""
    letters = [c for c in s if c.isalpha() and c.isascii()]
    if len(letters) < 3 or sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return s
    low = s.lower()
    low = re.sub(r"(^|[.!?…]\s*|[\"“(]\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), low)
    return re.sub(r"\bi\b", "I", low)


def normalize_source(text: str, lang: str = "ja") -> str:
    """OCR 原文 → 发给翻译引擎的规范文本。空 / 非字符串输入返回 ""。"""
    if not text or not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFC", text).replace("​", "").replace("﻿", "")
    eff = effective_lang(s, lang)
    if eff in ("ja", "zh"):
        # 日文 OCR 结果里的空白没有意义，但夹在其中的英文单词（招牌、外来语）之间的空格要留着
        s = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[A-Za-z0-9])", "\x00", s)
        s = re.sub(r"\s+", "", s).replace("\x00", " ")
        s = s.translate(_Z2H_ALNUM)
        s = s.replace("⁉", "！？").replace("⁈", "？！").replace("‼", "！！")
        s = s.replace("!", "！").replace("?", "？")
        if eff == "ja":
            # 日文几乎不用分号：竖排省略号「︙」常被 OCR 认成「；」（マ；マネージャー），还原后口吃才能识别
            s = s.replace("；", "…").replace(";", "…")
        s = _ELLIPSIS_SRC_JA.sub("…", s)
        s = re.sub(r"ー{2,}", "ー", s)
        s = re.sub(r"[〜～~]{2,}", "～", s)
    else:
        s = re.sub(r"\s+", " ", s).strip()
        s = _ELLIPSIS_SRC_EN.sub("…", s)
        if eff == "en":
            s = _sentence_case(s)
    return s.strip()


# ---- 口吃 ---------------------------------------------------------------------------------------
_STUTTER_RE = re.compile(rf"([{_HIRA}{_KATA}][{_SMALL_KANA}]?)(?:[…‥・.．]+|[、，,])")
_OPEN_QUOTES_RE = re.compile(r"^[「『（(“\"《【]+")


def split_stutter(text: str) -> tuple[int, str]:
    """「マ…マネージャー！？」→ (1, 「マネージャー！？」)；「な、な、なんで」→ (2, 「なんで」)。

    只认句首的口吃；单个感叹词后接汉字（あ、本当だ）不算口吃。"""
    if not text:
        return 0, ""
    m = _OPEN_QUOTES_RE.match(text)
    lead = m.group(0) if m else ""
    body = text[len(lead):]
    n = 0
    while n < 4:
        m = _STUTTER_RE.match(body)
        if not m:
            break
        unit, rest = m.group(1), body[m.end():]
        if len(rest) <= len(unit):
            break
        head = rest[:len(unit)]
        if kana_fold(head) == kana_fold(unit):
            pass
        elif len(unit) == 1 and _HAN_RE.match(rest[0]) and kana_fold(unit) not in _INTERJECTIONS:
            pass  # ち、違う / な、何で：口吃后面接汉字
        else:
            break
        n += 1
        body = rest
    return n, lead + body


def apply_stutter(n: int, text: str) -> str:
    """按译文首字补回口吃：(1, 「经理！？」) → 「经、经理！？」；英文 → 「W-wait」。"""
    if n <= 0 or not text:
        return text
    m = re.match(r"^([\s「『（(“\"'‘《【]*)(.)", text)
    if not m:
        return text
    lead, ch = m.group(1), m.group(2)
    rest = text[len(lead):]
    if _HAN_RE.match(ch) or _KANA_RE.match(ch) or _HANGUL_RE.match(ch):
        return lead + (ch + "、") * n + rest
    if ch.isalpha():
        return lead + (ch + "-") * n + rest
    return text


# ---- 首尾装饰符号 -------------------------------------------------------------------------------
def split_affixes(text: str) -> tuple[str, str, str]:
    """「ありがとう♡」→ ("", "ありがとう", "♡")。"""
    m1 = _AFFIX_LEAD_RE.search(text)
    lead = m1.group(0) if m1 else ""
    body = text[len(lead):]
    m2 = _AFFIX_TRAIL_RE.search(body)
    trail = m2.group(0) if m2 else ""
    if trail:
        body = body[: len(body) - len(trail)]
    return lead, body, trail


def attach_affixes(lead: str, text: str, trail: str) -> str:
    if lead and not text.startswith(lead):
        text = lead + text
    if trail and not text.endswith(trail):
        text = text + trail
    return text


# ---- 纯标点 -------------------------------------------------------------------------------------
def is_punct_only(text: str) -> bool:
    """只有标点 / 符号（…、！？、ー、♡），没有任何文字。"""
    t = (text or "").strip()
    if not t:
        return False
    for ch in t:
        if ch.isspace() or ch in _PUNCT_EXTRA:
            continue
        if unicodedata.category(ch)[0] in "PSZ":
            continue
        return False
    return True


def punct_passthrough(text: str, punct_style: str = "cjk") -> str:
    """纯标点块直接搬成中文排版习惯：… → ……，!? → ！？，ー → ——。"""
    t = re.sub(r"\s+", "", text or "")
    t = t.replace("⁉", "！？").replace("⁈", "？！").replace("‼", "！！")
    t = t.replace("!", "！").replace("?", "？")
    t = re.sub(r"[.．。・･‥…]{2,}|[‥…]", "……", t)
    t = re.sub(r"[ーｰ]+", "——", t)
    t = re.sub(r"[〜~]+", "～", t)
    return _quote_style(t, punct_style)


# ---- 中文后处理 ---------------------------------------------------------------------------------
def _to_fullwidth_punct(s: str) -> str:
    s = s.replace("!", "！").replace("?", "？")
    s = re.sub(rf"(?<=[{_CJKP}]),|,(?=\s*[{_CJKP}])", "，", s)
    s = re.sub(rf"(?<=[{_CJKP}]);|;(?=\s*[{_CJKP}])", "；", s)
    s = re.sub(rf"(?<=[{_HAN}]):|:(?=\s*[{_HAN}])", "：", s)
    s = re.sub(rf"\(([^()]*[{_HAN}][^()]*)\)", r"（\1）", s)
    s = re.sub(rf"(?<=[{_HAN}])\.(?=\s|$|[」』”])", "。", s)
    return s


def _quote_style(s: str, style: str) -> str:
    if style == "western":
        return s.translate(str.maketrans({"「": "“", "」": "”", "『": "‘", "』": "’"}))
    s = s.replace("“", "「").replace("”", "」").replace("＂", '"')
    s = re.sub(r"(?<![A-Za-z])‘", "『", s)
    s = re.sub(r"’(?![A-Za-z])", "』", s)
    s = re.sub(r'"([^"\n]*)"', r"「\1」", s)
    return s


def _strip_trailing_period(s: str, src: str) -> str:
    """机翻爱在句尾补「。」，而漫画气泡原文通常没有句号：原文不以句号结尾就去掉。"""
    if s.endswith("。") and len(s) > 1:
        tail = (src or "").rstrip()
        if not tail.endswith(("。", ".", "．")):
            s = s[:-1].rstrip()
    return s


_WRAP_PAIRS = {"「": "」", "『": "』", "（": "）", "(": ")", "“": "”", "《": "》", "【": "】", "〈": "〉"}


def _drop_stray_brackets(src: str, out: str) -> str:
    """大模型偶尔在句尾多带一个「」」、句首多一个「「」——原文里没有、译文里又不成对的，去掉。"""
    for o, c in _WRAP_PAIRS.items():
        if o in src or c in src:
            continue
        while out.count(c) > out.count(o) and out.endswith(c):
            out = out[:-1].rstrip()
        while out.count(o) > out.count(c) and out.startswith(o):
            out = out[1:].lstrip()
    # 同一对括号重复包了两层（「「…」」）且原文只有一层：剥掉一层
    for o, c in _WRAP_PAIRS.items():
        if out.startswith(o + o) and out.endswith(c + c) and not src.strip().startswith(o + o):
            out = out[1:-1]
    return out


def fix_wrapping(src: str, out: str) -> str:
    """原文整句被一对括号 / 引号包住时，保证译文也被同一对包住（机翻常把「」挪位或吃掉：「ありがとう」→ 译「」）。"""
    s = (src or "").strip()
    out = _drop_stray_brackets(s, out)
    if len(s) < 3 or s[0] not in _WRAP_PAIRS:
        return out
    o, c = s[0], _WRAP_PAIRS[s[0]]
    if s[-1] != c or s.count(o) != 1 or s.count(c) != 1:
        return out                        # 只处理「整句外面一层」的简单情况，嵌套 / 多段不动
    inner = out.replace(o, "").replace(c, "").strip()
    return o + inner + c if inner else out


def postprocess_zh(text: str, src: str = "", punct_style: str = "cjk") -> str:
    """机翻中文 → 漫画嵌字用的中文。"""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text)).replace("\r\n", "\n").replace("\r", "\n").strip()
    s = _ELLIPSIS_OUT.sub("……", s)
    s = _DASH_OUT.sub("——", s)
    s = _to_fullwidth_punct(s)
    s = s.replace("~", "～")
    s = re.sub(r"～{2,}", "～", s)
    s = _SPACE_BETWEEN_CJK.sub("", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"，{2,}", "，", s)
    s = re.sub(r"。{2}", "。", s)
    s = re.sub(r"……[，。]", "……", s)
    s = fix_wrapping(src, s)
    s = _quote_style(s, punct_style)
    s = _strip_trailing_period(s, src)
    return s.strip()


def looks_untranslated(src: str, out: str, src_lang: str) -> bool:
    """引擎是不是把原文原样（或基本原样）退回来了。

    英文 / 纯汉字原文原样返回可能是对的（OK、NASA、人名），只对含假名 / 谚文的原文判断。"""
    o = (out or "").strip()
    if not o:
        return True
    if src_lang == "zh" or not (has_kana(src) or has_hangul(src)):
        return False
    if same_text(src, o):
        return True
    letters = [c for c in o if unicodedata.category(c)[0] == "L"]
    if not letters:
        return False
    bad = sum(1 for c in letters if _KANA_RE.match(c) or _HANGUL_RE.match(c))
    return bad / len(letters) > 0.5


# ---- 术语表 -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GlossaryEntry:
    src: str
    dst: str


_PH_RE = re.compile(r"[{｛]\s*([0-9０-９]{1,2})\s*[}｝]")


class Glossary:
    """术语表：settings["translate"]["glossary"] = [{"src": "原文", "dst": "译文", "note": ""}]。"""

    def __init__(self, items: Any = None) -> None:
        entries: list[GlossaryEntry] = []
        seen: set[str] = set()
        for it in items if isinstance(items, (list, tuple)) else []:
            if not isinstance(it, dict):
                continue
            src = str(it.get("src") or "").strip()
            dst = str(it.get("dst") or "").strip()
            if not src or not dst or src in seen:
                continue
            seen.add(src)
            entries.append(GlossaryEntry(src, dst))
        entries.sort(key=lambda e: len(e.src), reverse=True)   # 长词优先匹配
        self.entries = entries
        self._map: dict[str, GlossaryEntry] = {}
        parts: list[str] = []
        for e in entries:
            if e.src.isascii():
                parts.append(rf"(?i:(?<![A-Za-z0-9]){re.escape(e.src)}(?![A-Za-z0-9]))")
                self._map.setdefault(e.src.lower(), e)
            else:
                parts.append(re.escape(e.src))
                self._map.setdefault(e.src, e)
        self._re = re.compile("|".join(parts)) if parts else None

    def __bool__(self) -> bool:
        return bool(self.entries)

    def _entry(self, matched: str) -> GlossaryEntry | None:
        return self._map.get(matched) or self._map.get(matched.lower())

    def protect(self, text: str) -> tuple[str, list[str]]:
        """把术语换成 {0} {1} 占位符；返回 (新文本, 占位符序号 → 译名)。"""
        if not self._re or not text or _PH_RE.search(text):
            return text, []
        dsts: list[str] = []

        def repl(m: re.Match[str]) -> str:
            e = self._entry(m.group(0))
            if e is None:
                return m.group(0)
            if e.dst not in dsts:
                dsts.append(e.dst)
            return "{%d}" % dsts.index(e.dst)

        out = self._re.sub(repl, text)
        return out, dsts

    def direct(self, text: str) -> str:
        """直接把术语替换成译名（占位符丢失时的退路）。"""
        if not self._re or not text:
            return text
        return self._re.sub(lambda m: (self._entry(m.group(0)) or GlossaryEntry(m.group(0), m.group(0))).dst, text)

    @staticmethod
    def restore(text: str, dsts: list[str]) -> tuple[str, bool]:
        """把译文里的占位符（含全角 ｛０｝ 变体）换回译名；返回 (文本, 是否全部找回)。"""
        if not dsts or not text:
            return text, not dsts
        found: set[int] = set()

        def repl(m: re.Match[str]) -> str:
            i = int(m.group(1).translate(_Z2H_ALNUM))
            if 0 <= i < len(dsts):
                found.add(i)
                return dsts[i]
            return m.group(0)

        out = _PH_RE.sub(repl, text)
        return out, len(found) == len(dsts)

    def correct(self, src: str, out: str) -> str:
        """翻译后校正：原文里出现的术语如果在译文里还残留原文（引擎没翻），替换成译名。"""
        if not self._re or not out:
            return out
        for e in self.entries:
            if e.src in out:
                out = out.replace(e.src, e.dst)
            elif e.src.isascii() and e.src.lower() in out.lower():
                out = re.sub(re.escape(e.src), e.dst, out, flags=re.I)
        return out

    def relevant(self, texts: Iterable[str]) -> list[GlossaryEntry]:
        """这批文本里出现过的术语（给本地大模型的提示词用，避免整张大表塞进上下文）。"""
        if not self._re:
            return []
        hit: dict[str, GlossaryEntry] = {}
        for t in texts:
            for m in self._re.finditer(t or ""):
                e = self._entry(m.group(0))
                if e is not None:
                    hit[e.src] = e
        return list(hit.values())

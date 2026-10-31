"""内置拟声词 / 拟态词词典（日文为主，附少量韩文 / 英文），全部是中性词汇。

机翻对拟声词很差（「ドキドキ」→「心跳加速」还算好，「ゴゴゴ」常被译成「哥哥哥」），所以
is_sfx 的块先查这里，查不到再机翻。

匹配规则（词典 key 和查询文本走同一套归一化，保证对称）：
- f0：NFKC（半角片假名 → 全角）、片假名 → 平假名、英文小写、去掉空白 / 标点 / 符号；
- f1：去长音「ー」、去拖音元音（ゴオオ → ご、ギィ → ぎ）、英文连续同字母合一（BOOOOM → bom）；
- f2：再去促音「っ」和小元音（ドキッ → どき、バシッ → ばし）；
- 依次查 f0 / f1 / f2；都查不到时找最小重复单元（ドドドド = ど×4、ざわざわざわ = ざわ×3），
  先查「单元×2」（どきどき），再查单元本身（ど → 咚，单字译文按次数重复，最多 4 次）。
- 原文句尾的 ！？… 会搬到译文后面（ドーン！！ → 轰——！！）。
"""
from __future__ import annotations

import logging
import re
import unicodedata

from .textproc import kana_fold, punct_passthrough

log = logging.getLogger("hanhua.translate")

# ---- 词典数据：每行「原文 译文」 -------------------------------------------------------------------
# 同一归一化形式下先出现的优先；单字单元（ド、ゴ…）放在前面，保证重复匹配时取到单字译文。
_JA = """
ド 咚
ゴ 轰
ダ 哒
タ 嗒
バ 啪
ガ 嘎
ピ 哔
ポ 啵
ヴ 嗡
ドン 咚
ドーン 轰——
ドドーン 轰轰——
バン 砰
バーン 砰——
ドカン 轰隆
ドカーン 轰隆——
ドッカーン 轰隆——
ドガッ 砰
ドゴッ 咚
ボカッ 砰
バキッ 咔嚓
ボキッ 咔嚓
ベキッ 咔
ポキッ 啪
パキッ 啪
メキッ 咔
ガン 哐
ガツン 咣
ゴン 咚
ゴツン 咚
ゴチン 咚
コツン 嗒
ドスッ 噗
ドスン 咚
ズシン 咚
ズドン 轰
ズドーン 轰隆——
ズガーン 轰隆——
チュドーン 轰隆——
ボン 嘭
ボーン 嘭——
ボカーン 轰——
バシッ 啪
パシッ 啪
ビシッ 唰
ピシッ 啪
パン 啪
パーン 啪——
パチン 啪
パチッ 啪
ペチ 啪
ペシッ 啪
ベシッ 啪
ゲシッ 咚
ボコッ 咚
ボコボコ 噼里啪啦
ガシッ 抓
ギュッ 紧
ギュー 紧紧——
ムギュ 挤
グイッ 拽
ポン 啵
ポンポン 拍拍
ナデナデ 摸摸
ツンツン 戳戳
プニプニ 软软
ムニッ 捏
ブチッ 啪
プチッ 啪
バシャ 哗
バシャッ 哗啦
バシャバシャ 哗啦哗啦
ザバーン 哗啦——
ザブン 扑通
ドボン 扑通
ポチャン 扑通
ピチャ 啪嗒
ピチャピチャ 啪嗒啪嗒
ポタ 滴答
ポタポタ 滴答滴答
ポツポツ 滴滴答答
ザー 哗——
ザーザー 哗哗
シトシト 淅淅沥沥
ジャー 哗——
ゴボゴボ 咕嘟咕嘟
ブクブク 咕嘟咕嘟
ヌルヌル 滑溜溜
ベタベタ 黏糊糊
ネバネバ 黏糊糊
ビショビショ 湿漉漉
ゴロゴロ 轰隆隆
ピカッ 闪
ピカピカ 闪闪发亮
キラキラ 亮晶晶
キラッ 闪
ギラギラ 明晃晃
ヒュー 呼——
ヒュッ 嗖
ビュン 嗖
ビューン 嗖——
ビュッ 嗖
ヒュルル 咻——
サワサワ 沙沙
ザワザワ 嘈杂
ザワッ 哗
メラメラ 熊熊
ボウッ 呼
ゴウッ 呼
ゴォォ 轰——
パチパチ 噼啪
ジュー 滋滋
グツグツ 咕嘟咕嘟
コトコト 咕嘟咕嘟
ポカポカ 暖洋洋
ヒンヤリ 凉飕飕
ジメジメ 潮乎乎
シーン 一片寂静
シン 寂静
バタン 砰
バタッ 扑通
バタバタ 噼里啪啦
ドタバタ 乒乒乓乓
ドタドタ 噔噔噔
ガチャ 咔嚓
ガチャン 哐当
ガシャン 哐啷
ガシャーン 哐啷——
ガラガラ 哗啦啦
ガラッ 哗啦
ガタッ 哐
ガタン 哐当
ガタガタ 咯噔咯噔
カタカタ 咔哒咔哒
カタッ 咔哒
カチ 咔
カチッ 咔嚓
カチャ 咔嚓
カチャカチャ 咔嚓咔嚓
カシャ 咔嚓
パシャ 咔嚓
カチカチ 咔哒咔哒
チクタク 嘀嗒嘀嗒
コンコン 咚咚
トントン 咚咚
ドンドン 咚咚咚
ピンポーン 叮咚
ピンポン 叮咚
ブブー 嘟嘟
ブー 嘟——
ピッ 哔
ピピッ 哔哔
ピー 哔——
プルルル 嘟噜噜
ジリリリ 叮铃铃
リンリン 铃铃
チリン 叮铃
カーン 铛——
ゴーン 当——
キーン 嗡——
カラン 当啷
カランコロン 叮当
チャリン 叮当
ジャラジャラ 哗啦哗啦
ゴトッ 咚
コトッ 嗒
ドサッ 扑通
ドサドサ 扑通扑通
バサッ 唰
バサバサ 扑棱扑棱
パタパタ 啪嗒啪嗒
パラパラ 哗啦啦
ペラペラ 哗啦哗啦
ペラッ 哗
ビリビリ 刺啦刺啦
ビリッ 刺啦
バリバリ 噼里啪啦
バリッ 咔嚓
メキメキ 咔咔
ミシミシ 咯吱咯吱
ギシギシ 嘎吱嘎吱
ギー 嘎吱——
キー 吱——
キキーッ 吱——
ブーン 嗡——
ブンブン 嗡嗡
ブロロロ 轰隆隆
ヴー 嗡——
ウィーン 嗡——
シュッ 唰
シュン 咻
サッ 唰
ササッ 唰唰
スッ 嗖
スーッ 嗖——
ズバッ 唰
ザシュッ 噗嗤
ザクッ 噗
ブスッ 噗
グサッ 噗嗤
シャキーン 锵——
チャキ 咔
カキーン 铿——
キン 铮
ガキン 锵
ダッ 嗖
スタスタ 噔噔
トコトコ 哒哒
テクテク 溜达溜达
コツコツ 嗒嗒
カツカツ 嗒嗒
ノシノシ 咚咚
ズルズル 吸溜吸溜
ズルッ 哧溜
ツルッ 哧溜
スルッ 嗖
クルッ 一转
クルクル 团团转
グルグル 转啊转
フラフラ 晃晃悠悠
ヨロヨロ 踉踉跄跄
フワフワ 轻飘飘
フワッ 飘
ユラユラ 摇摇晃晃
グラグラ 摇摇晃晃
ブラブラ 晃来晃去
ピョン 蹦
ピョンピョン 蹦蹦跳跳
ゴロン 骨碌
コロコロ 骨碌碌
ジタバタ 扑腾扑腾
ウロウロ 转来转去
ソワソワ 坐立不安
モジモジ 扭扭捏捏
ノロノロ 慢吞吞
ダラダラ 懒洋洋
ドキドキ 怦怦
ドキッ 怦
ドクン 扑通
ドクドク 扑通扑通
バクバク 怦怦怦
ゴクッ 咕嘟
ゴクリ 咕嘟
ゴクゴク 咕嘟咕嘟
ゴックン 咕咚
モグモグ 嚼嚼
ムシャムシャ 吧唧吧唧
パクッ 啊呜
パクパク 啊呜啊呜
ズズッ 吸溜
グー 咕——
グーグー 呼呼
スースー 呼呼
スヤスヤ 呼呼
ハァハァ 呼哧呼哧
ゼェゼェ 呼哧呼哧
フゥ 呼
ハァ 唉
ハッ 啊
ギクッ 咯噔
ギクリ 咯噔
ビクッ 一颤
ビクビク 战战兢兢
ブルブル 哆哆嗦嗦
ガクガク 哆哆嗦嗦
ゾクッ 一阵寒意
ゾクゾク 毛骨悚然
ゾワッ 发毛
ヒヤッ 心头一凉
ドヨーン 阴沉沉
ズーン 沉——
ガーン 晴天霹雳
ジーン 心头一热
ムカッ 火大
ムカムカ 火冒三丈
イライラ 烦躁
プンプン 气鼓鼓
プイッ 哼
ニコッ 微笑
ニコニコ 笑眯眯
ニヤッ 坏笑
ニヤニヤ 嘿嘿
ニッ 嘻
クスクス 嗤嗤
クスッ 噗嗤
プッ 噗
ゲラゲラ 哈哈大笑
ケラケラ 咯咯咯
アハハ 啊哈哈
ハハ 哈哈
ワハハ 哇哈哈
ウフフ 呵呵
エヘヘ 嘿嘿
シクシク 呜呜
メソメソ 哭哭啼啼
ポロポロ 扑簌簌
グスッ 抽噎
ジー 盯——
ジッ 盯
ジロッ 瞪
ジロジロ 上下打量
チラッ 瞥
チラチラ 偷瞄
キョロキョロ 东张西望
パチクリ 眨眨眼
ポッ 脸红
カァッ 脸红
ボーッ 发呆
ウトウト 昏昏欲睡
コクリ 点头
コクコク 连连点头
ペコリ 鞠躬
ヘトヘト 筋疲力尽
クタクタ 精疲力竭
ヘロヘロ 软绵绵
ワクワク 期待
ウキウキ 兴冲冲
ルンルン 乐呵呵
キュン 心动
キュンキュン 心动不已
ズキズキ 阵阵作痛
ズキッ 刺痛
チクッ 刺
ヒリヒリ 火辣辣
グガー 呼噜
ワンワン 汪汪
ワン 汪
ニャー 喵——
ニャン 喵
チュンチュン 啾啾
カーカー 嘎嘎
ミーンミーン 知了知了
ケロケロ 呱呱
モー 哞——
コケコッコー 喔喔喔——
ブヒ 哼哧
ヒヒーン 咴儿——
ピヨピヨ 叽叽
ガオー 嗷呜——
ジャーン 锵锵
ジャジャーン 锵锵锵——
チーン 叮——
ピキーン 灵光一闪
ピコーン 叮
ゴゴゴ 轰轰轰
ドドド 咚咚咚
ズズズ 隆隆隆
ジジジ 滋滋滋
ヒソヒソ 窃窃私语
ガヤガヤ 吵吵嚷嚷
ワイワイ 热热闹闹
ペチャクチャ 叽叽喳喳
ボソッ 嘀咕
ボソボソ 嘀嘀咕咕
ブツブツ 嘟嘟囔囔
チッ 啧
ケッ 切
フン 哼
ムッ 不爽
ギャー 哇啊——
キャー 呀啊——
ワー 哇——
ウワァ 呜哇
ヒィ 咿
ヒッ 咿
グッ 唔
ググッ 唔唔
ムクッ 一骨碌
ガバッ 猛地
バッ 唰
パッ 唰
ピタッ 定
ピタリ 戛然而止
ピクッ 一抽
ピクピク 一抽一抽
ドロン 嘭
ボワッ 呼
モクモク 滚滚
フッ 呼
シャッ 唰
シャー 嘶——
ザッ 唰
ザッザッ 沙沙
ジャリ 嚓
ガリガリ 咔哧咔哧
カリカリ 沙沙
サラサラ 沙沙
カサカサ 窸窸窣窣
ガサガサ 窸窸窣窣
ゴソゴソ 窸窸窣窣
ガサッ 窸窣
ゴシゴシ 使劲搓
フキフキ 擦擦
チョキチョキ 咔嚓咔嚓
"""

# 韩漫常见拟声词
_KO = """
쾅 轰
쿵 咚
꽝 哐
펑 嘭
탕 砰
탁 啪
퍽 砰
쿡 戳
휙 嗖
휘익 嗖——
쓱 唰
슥 唰
스윽 唰——
화악 唰
번쩍 闪
두근 怦
두근두근 怦怦
덜컥 咯噔
철컥 咔嚓
딸깍 咔哒
똑똑 咚咚
딩동 叮咚
짝짝 啪啪
부들부들 瑟瑟发抖
덜덜 哆哆嗦嗦
꿀꺽 咕嘟
냠냠 吧唧吧唧
털썩 扑通
우르르 轰隆隆
와장창 哐啷
쨍그랑 当啷
흠칫 一颤
움찔 一缩
헉 呃
후다닥 噔噔噔
끄덕 点头
꾸벅 鞠躬
"""

# 英文漫画常见拟声词
_EN = """
boom 轰
bang 砰
bam 砰
pow 砰
wham 砰
slam 砰
crash 哐啷
thud 咚
thump 咚
click 咔哒
knock 咚咚
snap 啪
crack 咔嚓
splash 哗啦
whoosh 呼——
zoom 嗖
buzz 嗡嗡
ring 铃铃
beep 哔
ding 叮
dingdong 叮咚
rumble 隆隆
creak 嘎吱
drip 滴答
tap 嗒
gasp 呃
sigh 唉
gulp 咕嘟
sniff 抽噎
haha 哈哈
hmph 哼
ugh 呃
wow 哇
"""

# ---- 归一化 -------------------------------------------------------------------------------------
_VOWEL_GROUPS = {
    "a": "あかがさざただなはばぱまやらわぁゃゎ",
    "i": "いきぎしじちぢにひびぴみりぃ",
    "u": "うくぐすずつづぬふぶぷむゆるぅゅゔ",
    "e": "えけげせぜてでねへべぺめれぇ",
    "o": "おこごそぞとどのほぼぽもよろをぉょ",
}
_VOWEL_OF = {ch: v for v, chars in _VOWEL_GROUPS.items() for ch in chars}
_PURE_VOWEL = {ch: _VOWEL_OF[ch] for ch in "あいうえおぁぃぅぇぉ"}
# 繁体目标：只覆盖本词典译文里简繁不同的字（拟声用字按台湾习惯：当→噹、里→哩〔噼哩啪啦〕、冲→沖〔兴冲冲〕）
_S2T = str.maketrans(
    "东乐儿兴冲凉动劲发叽呜哒哔哗啧噜团坏头尽张当懒战扑挤摇杂来棱气沥湿滚灵点烦热窃紧红绵缩脸腾语跄转软轰轻达连里"
    "铃铛铮铿锵闪闹阴阵雳静颤飕飘",
    "東樂兒興沖涼動勁發嘰嗚噠嗶嘩嘖嚕團壞頭盡張噹懶戰撲擠搖雜來稜氣瀝濕滾靈點煩熱竊緊紅綿縮臉騰語蹌轉軟轟輕達連哩"
    "鈴鐺錚鏗鏘閃鬧陰陣靂靜顫颼飄")
_LATIN_RUN = re.compile(r"([a-z])\1+")
_LATIN_RUN3 = re.compile(r"([a-z])\1{2,}")
_SMALL_VOWEL = re.compile("[ぁぃぅぇぉ]")
_TRAIL_PUNCT = re.compile(r"[!！?？…‥.．・･]+$")


def _f0(text: str) -> str:
    """基本形：拖音元音统一成单个「ー」（ドオオン / ドーーン / ドーン → どーん），英文 3 连以上压成 2 个。"""
    s = unicodedata.normalize("NFKC", text)
    s = kana_fold(s).lower()
    s = "".join(ch for ch in s if ch == "ー" or unicodedata.category(ch)[0] in "LN")
    out: list[str] = []
    prev: str | None = None
    for ch in s:
        v = _PURE_VOWEL.get(ch)
        if v is not None and prev is not None and v == prev:
            ch = "ー"                       # 拖音：ゴオオ / ギィ / じゅう
        if ch == "ー":
            if out and out[-1] == "ー":
                continue
        else:
            prev = _VOWEL_OF.get(ch)
        out.append(ch)
    return _LATIN_RUN3.sub(r"\1\1", "".join(out))


def _f1(s: str) -> str:
    """去掉长音，英文连续同字母合一（boom → bom）。"""
    return _LATIN_RUN.sub(r"\1", s.replace("ー", ""))


def _f2(s: str) -> str:
    return _SMALL_VOWEL.sub("", s.replace("っ", ""))


def _forms(text: str) -> tuple[str, str, str]:
    a = _f0(text)
    b = _f1(a)
    return a, b, _f2(b)


# ---- 建索引 -------------------------------------------------------------------------------------
def _parse(block: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in block.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
        elif parts:
            log.debug("拟声词词典行格式不对，已忽略：%r", line)
    return pairs


JA_SFX: list[tuple[str, str]] = _parse(_JA)
KO_SFX: list[tuple[str, str]] = _parse(_KO)
EN_SFX: list[tuple[str, str]] = _parse(_EN)

_LEVELS: tuple[dict[str, str], dict[str, str], dict[str, str]] = ({}, {}, {})
for _k, _v in (*JA_SFX, *KO_SFX, *EN_SFX):
    for _m, _f in zip(_LEVELS, _forms(_k)):
        if _f:
            _m.setdefault(_f, _v)


def size() -> int:
    """词典条数（按归一化后的不同原文计）。"""
    return len(_LEVELS[0])


def _get_any(key: str) -> str | None:
    for m in _LEVELS:
        v = m.get(key)
        if v is not None:
            return v
    return None


def _repeat_value(unit_v: str, k: int) -> str:
    """单元译文按次数重复：咚 ×3 → 咚咚咚；长词（一片寂静）/ 带破折号（轰——）不叠。"""
    if "—" in unit_v or len(unit_v) > 2:
        return unit_v
    return unit_v * max(1, min(k, 4, 6 // len(unit_v)))


def _repeat(q: str) -> str | None:
    """最小重复单元匹配：どどどど → ど×4 → 咚咚咚咚；ざわざわざわ → 嘈杂；とんとんとん → 咚咚咚。"""
    n = len(q)
    for ul in range(1, n // 2 + 1):
        if n % ul or q[:ul] * (n // ul) != q:
            continue
        unit, k = q[:ul], n // ul
        # 单字单元只认 f0 精确词条（ド / ゴ），不能用 f1 / f2 的缩减形式（は ← ハァ「唉」）
        single = _LEVELS[0].get(unit) if ul == 1 else _get_any(unit)
        double = _get_any(unit * 2)
        if ul == 1 and single is not None:
            return _repeat_value(single, k)
        if double is not None:
            half = len(double) // 2
            if len(double) % 2 == 0 and half and double[:half] == double[half:]:
                return _repeat_value(double[:half], k)     # 咚咚 → 咚 ×k
            return double                                   # 嘈杂：不可拆，原样
        if single is not None:
            return _repeat_value(single, k)
        return None
    return None


def to_traditional(text: str) -> str:
    """词典译文 → 繁体（只保证本词典用到的字）。"""
    return text.translate(_S2T)


def lookup(text: str, target: str = "zh-Hans") -> str | None:
    """查拟声词；查不到返回 None（交给机翻）。target 为 zh-Hant 时返回繁体。任何输入都不会抛异常。"""
    if not text or not isinstance(text, str):
        return None
    hit = _lookup(text)
    if hit is not None and "hant" in str(target or "").lower():
        hit = to_traditional(hit)
    return hit


def _lookup(text: str) -> str | None:
    try:
        core = text.strip()
        m = _TRAIL_PUNCT.search(core)
        tail = m.group(0) if m else ""
        if tail:
            core = core[: len(core) - len(tail)]
        forms = _forms(core)
        if not forms[0]:
            return None
        hit: str | None = None
        for mp, f in zip(_LEVELS, forms):
            if f and f in mp:
                hit = mp[f]
                break
        if hit is None:
            for f in dict.fromkeys(forms[1:]):
                if len(f) >= 2:
                    hit = _repeat(f)
                    if hit is not None:
                        break
        if hit is None:
            return None
        return hit + (punct_passthrough(tail) if tail else "")
    except Exception as e:  # noqa: BLE001 —— 词典查询绝不能让流程崩
        log.debug("拟声词查询出错 %r：%s", text, e)
        return None

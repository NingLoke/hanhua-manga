# 一键汉化漫画 4.0 —— 架构合同

> 这份文件是各模块之间的**接口合同**。实现模块时必须严格按这里的类名、方法签名、数据约定来写；
> 需要改合同（加字段 / 改签名 / 加设置项）时不要直接改底座文件，在交付说明里写清楚“需要底座改什么”。

## 0. 目标：逐条解决 3.0 的 7 个问题

| # | 3.0 的问题 | 4.0 的解法 | 负责模块 |
|---|---|---|---|
| 1 | 界面丑、功能少 | pywebview 网页风现代界面：任务队列、实时预览、前后对比滑块、校对编辑器、术语表、设置中心、深浅色 | `app` `server` `ui/` |
| 2 | 字放不进气泡 | 检测器直接给出**气泡框**；在去字后的图上求气泡内部区域（mask），按 mask 逐行 / 逐列可用宽度排版，二分字号，禁则处理，绝不越界 | `typeset/` |
| 3 | 背景被搞坏 | 只在**文字像素 mask** 内修改；纯色气泡直接填色；复杂背景用动态尺寸 LaMa（不变形）；最后强制 `mask==0` 处与原图逐位相同 | `inpaint` |
| 4 | 不能适配每本漫画 | RT-DETR（漫画 / 韩漫 / 国漫 / 美漫训练）；任意分辨率；韩漫长条自动切块；竖排 / 横排；日 / 韩 / 英 / 中源语言；阈值全部相对化（不写死像素） | `detect` `ocr` `typeset` |
| 5 | 太慢 | GPU（CUDA / DirectML）；白气泡不过 LaMa；OCR 批处理；整页批量翻译 + 缓存；检测 / 翻译 / 渲染流水线并行 | `runtime` `engine` `translate` |
| 6 | 格式太少 | 文件夹 / 散图 / zip / cbz / 7z / cb7 / rar / cbr / tar / pdf / epub / 套娃压缩包；输出 同格式 / 文件夹 / cbz / zip / pdf / epub，图片 jpg / png / webp | `io/` |
| 7 | 中途断了要重来 | 每页每个阶段落盘（原子写），重启自动续跑；暂停 / 继续 / 取消；翻译全挂时自动暂停并定时重试 | `engine/` |

## 1. 硬规则（所有模块）

1. **中文路径安全**：禁止 `cv2.imread` / `cv2.imwrite` / 任何把路径交给 C 库 ANSI 接口的写法。图片读写只用 `hanhua.imageio`（`read` / `decode` / `encode` / `write` / `write_bytes_atomic`）。客户的路径里经常有中文。
2. **推理只走 `Runtime`**：`rt.model(key).run(feeds)`；不准自己 new `InferenceSession`，不准依赖 torch / transformers。
3. **依赖白名单**：标准库 + `numpy` `opencv-python-headless` `pillow` `onnxruntime(-gpu/-directml)` `requests` `jaconv` `pypdfium2` `pywebview`（测试可用 `pytest`）。要加别的依赖先在交付说明里提出。
4. **授权**：只能用 Apache-2.0 / MIT / BSD / OFL 的模型、字体、代码。不准复制 GPL 项目（manga-image-translator、BallonsTranslator、comic-text-detector 源码等）的代码。
5. **图片约定**：内部一律 BGR `uint8 (H, W, 3)`；颜色 tuple 一律 RGB；Box = `(x1, y1, x2, y2)` 开区间。
6. **阈值相对化**：所有尺寸阈值按页面尺寸 / 文字框尺寸的比例来算，不写死像素（3.0 写死 `40×70` kernel、`>600px` 是适配差的根源）。
7. **日志**：`log = logging.getLogger("hanhua.<模块>")`；库代码里不准 `print`。给客户看的错误信息用中文、说人话（抛 `HanhuaError` 子类）。
8. **线程**：Engine 会在多个线程里调用各模块。模块对象构造一次、反复调用；内部可变缓存要加锁。GPU 调用已由 `Model.run` 加锁。
9. **代码风格**：Python 3.12，类型注解，中文注释（技术词保留英文），函数短小。跑命令用 `-X utf8`（控制台是 cp1252）。

## 2. 目录

```
4.0/
  hanhua/
    __init__.py  types.py  config.py  runtime.py  imageio.py      ← 底座（已完成，勿改）
    detect.py                 TextDetector                         ← 视觉
    ocr.py                    OCREngine（manga-ocr + PP-OCR）       ← 视觉
    inpaint.py                Inpainter                            ← 去字
    typeset/  __init__.py fonts.py layout.py render.py  Typesetter / FontRegistry
    translate/ __init__.py service.py engines.py cache.py textproc.py sfx_dict.py
    bookio/    __init__.py formats.py pdf.py epub.py archive.py export.py   （不叫 io，避免和标准库混淆）
    engine/    __init__.py job.py engine.py
    server.py  app.py  ui/ (index.html app.css app.js demo.py)
  models/  fonts/  tests/  build/  envs/(cuda|dml|cpu)
```

## 3. 底座（已实现）

- `hanhua.types`：`TextBlock`、`PageData`、`STAGES`、`stage_at_least`、Box 工具函数（`clamp_box` `expand_box` `box_iou` `box_contain_ratio` `box_area`）、异常 `HanhuaError` / `TranslationUnavailable` / `UnsupportedFormat` / `Cancelled`。
- `hanhua.runtime`：`Runtime(device, cpu_threads, gpu_id)`（`.device` ∈ cuda/dml/cpu，`.label`，`.model(key)` → `Model.run(feeds) -> list[np.ndarray]`），`resolve_profile(name, device) -> Profile`，`detect_hardware()`，路径函数 `app_root()` `resource_path()` `models_dir()` `fonts_dir()` `data_dir()`，`build_info()`，`MODEL_FILES`。
- `hanhua.config`：`DEFAULTS`（完整设置结构）、`SettingsStore(path).get() / update(patch) / reset() / on_change(cb)`。
- `hanhua.imageio`：见硬规则 1。

`Profile` 字段：`name device detector detector_size ocr_encoder ocr_decoder ocr_batch inpaint_max_side inpaint_bucket cpu_workers`。

## 4. 模型事实（已在 RTX 5060 Ti 上实测）

| key | 输入 | 输出 | 说明 |
|---|---|---|---|
| `detector` / `detector_int8` | `images` f32 `[N,3,H,W]`（RGB，÷255，**不做** mean/std，直接 resize 到 640×640，不 letterbox）；`orig_target_sizes` int64 `[[W, H]]` | `labels` `[N,300]`、`boxes` `[N,300,4]`（xyxy，已是原图像素坐标）、`scores` `[N,300]` | 类别 0 bubble 气泡、1 text_bubble 气泡内文字、2 text_free 气泡外文字。韩漫长条训练时是竖向切块的 |
| `detector_s_int8` | 同上但固定 `[1,3,640,640]` | 同上 | CPU 极速档 |
| `ocr_encoder(_int8)` | `pixel_values` f32 `[B,3,224,224]` | `last_hidden_state` `[B,197,768]` | 预处理见 `models/manga-ocr/preprocessor_config.json` |
| `ocr_decoder(_int8)` | `input_ids` int64 `[B,T]`、`encoder_hidden_states` `[B,197,768]` | `logits` `[B,T,6144]` | 无 KV cache，greedy 每步全序列；特殊 token 见 `config.json` / `generation_config.json`；词表 `vocab.txt` |
| `lama` | `image` f32 `[1,3,h,w]`（RGB 0..1）、`mask` f32 `[1,1,h,w]`（0/1） | `inpainted` `[1,3,h,w]`（**0..1**） | 动态尺寸；**每种新尺寸 GPU 首次要预热 0.5~5s**，必须按 `profile.inpaint_bucket` 分桶 |
| `ppocr_det` | `x` f32 `[N,3,H,W]` | 概率图 `[N,1,H,W]` | 前后处理参数见 `models/ppocr/det.yml` |
| `ppocr_rec_multi/korean/en` | `x` f32 `[N,3,48,W]` | `[N,T,C]` 概率 | CTC，字典在对应 `.yml` 的 `PostProcess.character_dict` |

实测速度（CUDA / DML / CPU）：LaMa 512×768 86 / 121 / 1256 ms；检测 19 / 23 / 209 ms；OCR 编码 batch8 19 / 31 / 387 ms。
int8 动态量化模型在 GPU 后端会退回 CPU 执行，所以 GPU 档位一律用 fp32（`resolve_profile` 已处理）。

## 5. 模块接口

### 5.1 `hanhua/detect.py`
```python
class TextDetector:
    def __init__(self, rt: Runtime, profile: Profile) -> None: ...
    def detect(self, img: np.ndarray, reading: str = "rtl") -> tuple[list[TextBlock], list[Box]]:
        """返回 (按阅读顺序排好、id=0..n-1 的文字块, 全部气泡框)。"""
```
- 任意尺寸：长宽比 > 2.2 的长条（韩漫）按宽度切成约 `1.6×宽` 高、25% 重叠的块分别检测再合并；超宽跨页同理横切。
- 按类别分开阈值 + NMS；去掉极小框（相对页面）；同一气泡内相邻的多个文字框合并为一块。
- 每个文字框找包含它最多的气泡（containment ≥ 0.6）→ `kind="bubble"`、`bubble_box`；类别是 text_bubble 但没找到气泡 → 仍是 bubble，`bubble_box=None`；text_free 且不在气泡里 → `kind="free"`。
- `direction`：按框长宽比 + 列 / 行投影判断竖排 `v` / 横排 `h`。
- 阅读顺序：`rtl`（日漫：行从上到下，行内从右到左）/ `ltr`（其它）。按纵向重叠聚成行再排序。

### 5.2 `hanhua/ocr.py`
```python
class OCREngine:
    def __init__(self, rt: Runtime, profile: Profile, lang: str = "ja") -> None: ...
    def recognize(self, img: np.ndarray, blocks: list[TextBlock]) -> None:
        """原地填写 src_text、ocr_conf、text_color、stroke_color、is_sfx（必要时修正 direction）。"""
```
- `ja`：manga-ocr，批量编码（`profile.ocr_batch`），批量 greedy 解码；置信度 = 生成 token 的平均最大概率；后处理与 manga_ocr 一致（去空白、`…`→`...`、连续 `・.` 归一、`jaconv.h2z(ascii=True, digit=True)`）。
- `ko` / `en` / `zh`：PP-OCR（DB 检测出行 → 逐行 CTC 识别 → 按阅读顺序拼接；ko/en 用空格连接）。`auto`：竖排或含假名走 manga-ocr，否则 PP-OCR multi，识别出谚文再用 korean 模型。
- `is_sfx`：气泡外 + 文本很短 / 大量重复假名、长音、促音 / 置信度低 / 字形明显大于本页对白。
- `text_color`：文字笔画主色（框内 Otsu 后少数类）；`stroke_color`：气泡外文字的外描边色（有的话）。

### 5.3 `hanhua/inpaint.py`
```python
class Inpainter:
    def __init__(self, rt: Runtime, profile: Profile, settings: dict) -> None: ...
    def text_mask(self, img: np.ndarray, blocks: list[TextBlock]) -> np.ndarray:
        """uint8 HxW（0/255），所有 skip=False 的块的文字像素 mask 的并集。"""
    def clean(self, img: np.ndarray, blocks: list[TextBlock]) -> tuple[np.ndarray, np.ndarray]:
        """返回 (去字后的图, 实际使用的 mask)。不变量：mask==0 的像素与输入逐位相同。"""
```
- 气泡内文字：在气泡内部估计底色，和底色差异大的像素 = 笔画；膨胀覆盖抗锯齿边缘；**不能碰到气泡描边**。
- 气泡外文字：框内笔画 + 描边；可以用 `ppocr_det` 概率图细化文字区域；异常时退回圆角框 mask。
- `settings["inpaint"]["mode"]`：`auto`（文字周围一圈是纯色 → 直接填中位色；否则 LaMa）/ `fill` / `lama`。
- LaMa：按连通区域取上下文裁剪（不缩放变形，必要时按比例缩小到 `profile.inpaint_max_side`），pad 到 `profile.inpaint_bucket` 的倍数，跑完缩回原尺寸，只在 mask 内合成（羽化只在 mask 内侧）。

### 5.4 `hanhua/typeset/`
```python
class FontRegistry:
    def __init__(self, fonts_dir: str, user_dirs: list[str] | None = None) -> None: ...
    def list(self) -> list[dict]:          # [{"id","name","path","tags":[...]}]
    def get(self, font_id: str, size: int) -> PIL.ImageFont.FreeTypeFont: ...   # 带缓存；未知 id 退回默认
    def has_glyph(self, font_id: str, ch: str) -> bool: ...

class Typesetter:
    def __init__(self, settings: dict, fonts: FontRegistry) -> None: ...
    def render(self, clean: np.ndarray, blocks: list[TextBlock]) -> np.ndarray:
        """把每个 skip=False 且 translation 非空的块排进去；回写 render_box / render_font_size / overflow。"""
```
- 内置字体 id：`noto-sans-sc-medium/bold/black`、`noto-sans-tc-medium/bold/black`（可变字体 wght 500/700/900）、`lxgw-wenkai`、`zcool-kuaile`、`ma-shan-zheng`。目标是繁体时 sc 字体自动换成 tc。用户字体目录：`data_dir()/fonts`。
- 区域：气泡块 → 在**去字后的图**上、`bubble_box` 内从文字区域 flood fill 出气泡内部 mask（漏出去就退回内切椭圆），再按 `typeset.bubble_margin` 内缩；气泡外块 → 原文字框略放大的矩形。
- 方向：`out_direction` > 设置 `typeset.direction` > 自动（原文竖排且区域偏高 → 竖排，否则横排）。
- 适配：在 [min_px, max_px] 内二分字号；每行（横排）/ 每列（竖排）的可用长度按 mask 在该行 / 列带上的实际宽度计算；中文禁则（行首不放 `，。！？」』）…` 等，行尾不放 `「『（` 等，`……` `——` 不拆开）；竖排标点用竖排形式（`︙ ︱ ﹁﹂ ︵︶`，`，。` 靠右上）。字号范围按原文字号估计 + 页面高度相对化，可被设置覆盖。放不下 → 降到下限 0.75 倍仍不行则 `overflow=True`，但**永远不画出气泡**。
- 渲染：气泡内默认黑字（或原文颜色）；气泡外文字带描边（宽度 `typeset.stroke_free`×字号）；拟声词用 `font_sfx`。

### 5.5 `hanhua/translate/`
```python
class TranslationService:
    def __init__(self, settings: dict, cache_path: str) -> None: ...
    def translate_blocks(self, pages: list[list[TextBlock]], src_lang: str,
                         cancel: threading.Event | None = None) -> None:
        """为所有 (not skip) 且 src_text 非空且 translation=="" 的块填写译文（不覆盖人工修改）。
        所有引擎都失败时抛 TranslationUnavailable。"""
    def translate_texts(self, texts: list[str], src_lang: str, target: str | None = None) -> list[str]: ...
    def test_engine(self, engine_id: str) -> tuple[bool, str]: ...
    @staticmethod
    def engines() -> list[dict]:   # [{"id","name","kind":"free"|"local","batch":bool}]
    def close(self) -> None: ...
```
- **只用免费引擎**（用户决定）：google、edge（微软 Edge 翻译）、tencent（腾讯翻译君），实现时逐个实测，坏了的不要放进默认列表；另有免费的本地大模型（OpenAI 兼容接口，Ollama / LM Studio），整页上下文 + 术语表。
- 批量：一页（或几页）的所有块一次请求；数量对不上就拆分重试。按 `translate.engines` 顺序自动切换，429 / 5xx / 超时指数退避，引擎熔断冷却。
- 缓存：sqlite（WAL），key = (engine, src, tgt, 规范化原文)。
- 文本处理：日文 OCR 规范化（`...`→`…`、`!?`→`！？` 等）；中文后处理（全角标点、`……`、`——`、去 CJK 间空格、引号风格按 `typeset.punct_style`）；纯标点（`…` `！？`）直接搬运；术语表替换；拟声词先查内置词典（`sfx_dict.py`）再机翻。

### 5.5.1 本地翻译 `hanhua/llm/`（2026-10-31 用户要求：本地优先）
- 优先级：**内置本地大模型**（llama.cpp + 按配置推荐的 Qwen GGUF，离线不限次数）→ **填 Key 的 API 服务**（OpenAI 兼容预设）→ **免费在线引擎**兜底（2026-10 只有 Google 批量接口还能用；Edge 404、腾讯证书过期）。设置见 `config.DEFAULTS["translate"]["local" | "api" | "online_fallback"]`。
- `catalog.py`：模型清单（Qwen3.5 2B/4B/9B 的 Q4/Q6/Q8 + Qwen3.8 27B，均 Apache-2.0，带 SHA-256）和 `recommend(hw, backends, max_params)`：GPU 预算 = 显存 − 2.5GB（识别 / 去字模型）− 0.6GB；放不下用 CPU 小模型；「极速」档限制 ≤4B。
- `manager.py`：模型下载（HF / ModelScope 测速择优、断点续传、校验）、导入 GGUF、`LlamaServer`（单实例 llama-server，换模型自动重启，CUDA 后端与 onnxruntime 共用 `<app>/cuda` 的 cuBLAS）。模型目录 `data_dir()/models/llm`，可用 `HANHUA_LLM_DIR` 覆盖。
- `client.py`：`ChatTranslator`，整页台词 + 上文 + 术语表一次请求，JSON Schema 约束输出，条数不对自动二分；少样本示范。
- `engine.py`：`LocalLLMEngine` / `ApiLLMEngine` / `API_PRESETS`；在 `translate/engines.py` 注册为 `local` / `api` 两个引擎。
- 运行库：`runtimes/llama/<cuda|vulkan|cpu>/`（`build/fetch_assets.py --only runtimes` 下载，版本固定 b12001，已核对 GitHub 官方 SHA-256）；三个发行版分别打包 CUDA / Vulkan / CPU 后端。

### 5.6 `hanhua/bookio/`（`from hanhua.bookio import scan_inputs, open_book, Book, default_output_path, export_book`）
```python
def scan_inputs(paths: list[str], tmp_root: str) -> list[str]:
    """把拖进来的路径展开成“书”的列表：压缩包 / pdf / epub / 含图片的文件夹 / 散图（同目录散图算一本）。
    套娃压缩包（zip 里全是 cbz）展开到 tmp_root 下，每卷一本。"""

class Book:
    path: str; kind: str; title: str
    pages: list[str]        # 页面名（相对路径），自然排序 = 阅读顺序
    extras: list[str]       # 需要原样带走的非图片条目（ComicInfo.xml 等）
    def read_page(self, name: str) -> np.ndarray: ...       # BGR
    def read_page_bytes(self, name: str) -> bytes: ...
    def read_extra(self, name: str) -> bytes: ...
    def close(self) -> None: ...

def open_book(path: str, tmp_dir: str) -> Book: ...           # 失败抛 UnsupportedFormat（中文说明）
def default_output_path(book: Book, settings: dict) -> str: ...
def export_book(book: Book, pages: list[tuple[str, str]], out_path: str, settings: dict,
                progress: Callable[[int, int], None] | None = None) -> str:
    """pages = [(页面名, 渲染好的图片文件路径)]；按 settings["output"] 写出；先写 .part 再改名；返回最终路径。"""
```
- 输入：文件夹（递归）、散图、zip/cbz（修复 Shift-JIS / GBK 文件名乱码）、7z/cb7、rar/cbr、tar/cbt（用 Windows 自带 `tar.exe`=bsdtar 解压到 tmp）、pdf（pypdfium2：单图页取原始分辨率，否则按长边约 2000px 渲染）、epub（按 OPF spine 顺序取图）。跳过 `__MACOSX`、`Thumbs.db`、隐藏文件。
- 输出：`same`（zip→zip、cbz→cbz、7z/rar→cbz、pdf→pdf、epub→epub、文件夹 / 散图→文件夹）/ `folder` / `cbz` / `zip` / `pdf`（直接嵌 JPEG 流的轻量 PDF writer，不全量载入内存）/ `epub`（复制原 epub，只替换图片条目）。ComicInfo.xml 带走并标注中文。图片格式 `same` 时保留原扩展名。

### 5.7 `hanhua/engine/`
```python
class Job:      # 一本书
    id: str; source: str; title: str; kind: str; workdir: str; out_path: str
    status: str   # queued | running | paused | done | error | canceled
    message: str  # 给客户看的中文状态
    settings: dict           # 创建时的设置快照
    pages: list[PageData]
    def summary(self) -> dict: ...

class Engine:
    def __init__(self, store: SettingsStore, on_event: Callable[[dict], None], work_root: str | None = None,
                 components: "ComponentFactory | None" = None) -> None: ...
        # components：测试时注入假组件；None = 真实组件的懒加载工厂（首次开始任务时在后台线程加载模型）
    def add_paths(self, paths: list[str]) -> list[dict]: ...      # 同一源文件已有任务 → 复用并续跑
    def list_jobs(self) -> list[dict]: ...
    def start(self) -> None: ...
    def pause(self, job_id: str | None = None) -> None: ...
    def resume(self, job_id: str | None = None) -> None: ...
    def cancel(self, job_id: str) -> None: ...
    def remove(self, job_id: str, delete_files: bool = False) -> None: ...
    def retry_failed(self, job_id: str) -> None: ...
    def get_page(self, job_id: str, index: int) -> dict: ...
    def update_page(self, job_id: str, index: int, blocks: list[dict]) -> dict: ...   # 保存人工修改并重新排版
    def retranslate_page(self, job_id: str, index: int) -> dict: ...
    def redetect_page(self, job_id: str, index: int) -> dict: ...
    def rerender_all(self, job_id: str) -> None: ...
    def export(self, job_id: str) -> str: ...
    def page_image_path(self, job_id: str, index: int, kind: str) -> str | None: ...  # kind: src|clean|out|thumb
    def shutdown(self) -> None: ...
```
工作目录 `<work_root>/<job_id>/`（`work_root` 默认 `data_dir()/jobs`，可在设置里改）：
```
job.json            任务元数据 + 设置快照（原子写）
pages/0001.json     PageData（每完成一个阶段原子写一次）
clean/0001.png      去字补丁：RGBA，alpha=mask，未改动处全透明（比整页 PNG 小一个数量级）
out/0001.<ext>      成品页
thumbs/0001.jpg     360px 缩略图
tmp/                rar/7z/pdf 解压缓存
```
- job_id = sha1(源文件绝对路径 + 大小 + 修改时间)[:12]：同一本书再拖进来就是续跑。
- 流水线：视觉线程（检测 + OCR，GPU）→ 翻译（按页攒批，网络并发）→ 去字（GPU）→ 渲染 / 编码线程池（`profile.cpu_workers`）。阶段间有界队列控制内存。
- 断点：启动时扫描 work_root，未完成的任务恢复为 paused（`general.auto_resume` 为真则自动继续）；页面从已完成阶段的下一阶段继续。
- 翻译全挂（`TranslationUnavailable`）→ 任务暂停并显示中文原因，1 / 2 / 5 分钟后自动重试。单页出错 → 该页标 error 并继续，导出时用原图并在任务里列出。
- 拟声词策略在这里落实：`sfx_mode=keep` → `skip=True`；`erase` → 擦除但不嵌字。`translate_free_text=False` → 气泡外非拟声词块 `skip=True`。
- 事件（节流 ≤ 10 次/秒）：
  - `{"type":"job","job":summary}`，summary = `{id,title,source,kind,status,message,total,done,failed,stage_counts,speed_ppm,eta_s,out_path,cover,updated}`
  - `{"type":"page","job_id","index","stage","version"}`
  - `{"type":"log","level","message"}`
  - `{"type":"engine","running":bool,"device":label}`

### 5.8 界面 `hanhua/server.py` + `hanhua/app.py` + `hanhua/ui/`
通信统一走**本地 HTTP**（界面在任何浏览器里都能打开调试，pywebview 只负责窗口、拖拽完整路径、系统文件对话框）：
- `server.py`：`AppServer(api, ui_dir)`，`ThreadingHTTPServer` 绑 `127.0.0.1:0`，每次启动随机 token。
  - `GET /` → 重定向 `/ui/index.html?t=<token>`；`GET /ui/...` 静态文件（防路径穿越）。
  - `POST /api/<method>`（请求头 `X-Token`，body = JSON 参数对象）→ 调 `Api.<method>(**body)` → `{"ok": true, "data": ...}` 或 `{"ok": false, "error": "中文说明"}`。
  - `GET /events?t=<token>` → SSE 事件流（Engine 事件 100ms 合并成一批推送，带心跳）。
  - `GET /img/<job>/<index>/<kind>?t=<token>&v=<version>` → 页面图片（kind = src|clean|out|thumb）；带 `v` 的可长缓存。
  - `.base_url`、`.token`、`.start()`、`.stop()`。
- `app.py`：
  - `class Api`：纯 Python，不依赖 pywebview（浏览器 / 测试都能用）。方法（长操作放后台线程，接口立即返回）：`bootstrap()`、`pick_files()`、`pick_folder()`、`pick_output_dir()`、`add_paths(paths)`、`start()`、`pause(job_id)`、`resume(job_id)`、`cancel(job_id)`、`remove(job_id, delete_files)`、`retry(job_id)`、`get_page(job_id, index)`、`update_page(job_id, index, blocks)`、`retranslate_page(job_id, index)`、`redetect_page(job_id, index)`、`rerender_all(job_id)`、`export(job_id)`、`open_path(path)`、`reveal(path)`、`get_settings()`、`save_settings(patch)`、`reset_settings()`、`test_engine(engine_id)`、`list_fonts()`、`add_font(path)`、`hardware()`。文件对话框通过 app 注入的 `dialogs` 钩子实现，没有窗口时返回中文错误。
  - `main()`：日志（`data_dir()/logs/app.log` 轮转）、`SettingsStore`、`Engine`、`Api`、`AppServer`、pywebview 窗口（标题“一键汉化漫画 4.0”，1280×820，最小 1080×700）加载 `base_url`。关闭窗口 → `engine.shutdown()`（暂停并落盘）。
  - 拖拽：pywebview DOM 事件 `window.dom.document.events.drop += DOMEventHandler(handler, prevent_default=True)`，从 `event["dataTransfer"]["files"][i]["pywebviewFullPath"]` 取完整路径（6.2.1 已确认支持），调用 `api.add_paths`。
  - `python -m hanhua.app --demo` 用 `ui/demo.py` 的假引擎（同样的 Engine 方法签名）演示；`--demo --no-window` 只起服务器并打印 URL，供浏览器调试。
- 前端：纯 HTML/CSS/JS，无 CDN、无构建步骤（必须离线可用）；界面字体用系统 `"Microsoft YaHei UI", "Segoe UI"`。

## 6. 测试

- 只用 `tests/fixtures/real/`（非成人页面，见其 README）和合成页面；**不要打开原始 zip**。
- `tests/synth.py`：合成页面生成器（多种气泡形状 / 背景 / 竖横排 / 描边字 / 拟声词 / 长条），带真值（气泡 mask、文字 mask、文字内容）。
- 每个模块写自己的 `tests/test_<模块>.py`（pytest），调试图输出到 `tests/out/<模块>/`。
- 运行：在 4.0 目录下 `envs/cuda/Scripts/python.exe -X utf8 -m pytest tests/test_xxx.py -q`。

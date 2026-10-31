"""打包三个硬件版本。

用法（在 4.0 目录下，用任意 Python 3.10+ 运行；会自动调用对应 envs/<env> 里的 PyInstaller）:
    python build/build.py --variant gpu          # 通用显卡版（DirectML，默认推荐）
    python build/build.py --variant nvidia       # NVIDIA 加速版（CUDA）
    python build/build.py --variant cpu          # CPU 轻量版
    python build/build.py --variant all          # 三个都打
选项: --no-zip 不压缩；--skip-selftest 不跑自检；--skip-pyinstaller 只重新拷贝资源

产物: build/dist/<variant>/一键汉化漫画/ 和 build/release/一键汉化漫画4.0-<版本名>-v<版本>.zip
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
APP_NAME = "一键汉化漫画"

sys.path.insert(0, ROOT)
from hanhua.runtime import VERSION  # noqa: E402

VARIANTS = {
    "nvidia": {
        "env": "cuda",
        "llama": "cuda",               # 本地翻译引擎 llama.cpp 的后端（与 onnxruntime 共用 <app>/cuda 里的 cuBLAS）
        "title": "NVIDIA加速版",
        "models": [
            "detector/detector.onnx", "detector/config.json",
            "manga-ocr/encoder_model.onnx", "manga-ocr/decoder_model.onnx",
            "lama/lama-manga-dynamic.onnx",
        ],
    },
    "gpu": {
        "env": "dml",
        "llama": "vulkan",             # Vulkan：AMD / Intel / NVIDIA 都能跑
        "title": "通用显卡版",
        "models": [
            "detector/detector.onnx", "detector/config.json",
            "manga-ocr/encoder_model.onnx", "manga-ocr/decoder_model.onnx",
            "lama/lama-manga-dynamic.onnx",
        ],
    },
    "cpu": {
        "env": "cpu",
        "llama": "cpu",
        "title": "CPU轻量版",
        "models": [
            "detector/detector-v4-s_int8.onnx", "detector/detector_int8.onnx", "detector/config.json",
            "manga-ocr/encoder_model_int8.onnx", "manga-ocr/decoder_model_int8.onnx",
            "lama/lama-manga-dynamic.onnx",
        ],
    },
}
# 所有版本都带的模型配套文件
COMMON_MODEL_FILES = [
    "manga-ocr/config.json", "manga-ocr/generation_config.json", "manga-ocr/preprocessor_config.json",
    "manga-ocr/vocab.txt", "manga-ocr/tokenizer_config.json", "manga-ocr/special_tokens_map.json",
    "ppocr/det.onnx", "ppocr/det.yml", "ppocr/rec_multi.onnx", "ppocr/rec_multi.yml",
    "ppocr/rec_korean.onnx", "ppocr/rec_korean.yml", "ppocr/rec_en.onnx", "ppocr/rec_en.yml",
]
# NVIDIA 版实测需要的 CUDA / cuDNN DLL（curand、nvrtc、nvJitLink、cufftw、nvblas 用不到，已裁掉）
CUDA_DLLS = [
    "cu13/bin/x86_64/cudart64_13.dll", "cu13/bin/x86_64/cublas64_13.dll", "cu13/bin/x86_64/cublasLt64_13.dll",
    "cu13/bin/x86_64/cufft64_12.dll",
    "cudnn/bin/cudnn64_9.dll", "cudnn/bin/cudnn_graph64_9.dll", "cudnn/bin/cudnn_ops64_9.dll",
    "cudnn/bin/cudnn_cnn64_9.dll", "cudnn/bin/cudnn_adv64_9.dll", "cudnn/bin/cudnn_heuristic64_9.dll",
    "cudnn/bin/cudnn_engines_precompiled64_9.dll", "cudnn/bin/cudnn_engines_runtime_compiled64_9.dll",
    "cudnn/bin/cudnn_engines_tensor_ir64_9.dll", "cudnn/bin/cudnn_ext64_9.dll",
]
EXCLUDES = [
    "torch", "torchvision", "transformers", "tensorflow", "matplotlib", "scipy", "pandas",
    "IPython", "jupyter", "notebook", "pytest", "_pytest", "tkinter", "sympy", "PyQt5", "PyQt6", "PySide6",
]

THIRD_PARTY = """一键汉化漫画 {version} 第三方组件授权说明
=====================================================

本软件使用了以下第三方组件，均允许商业使用与再分发。各组件版权归其原作者所有。

【AI 模型】
- comic-text-and-bubble-detector（ogkalu，RT-DETR-v2）          Apache License 2.0
- manga-ocr-base（kha-white；ONNX 版 onnx-community）             Apache License 2.0
- AnimeMangaInpainting / lama_large_512px（dreMaz，基于 LaMa）    MIT License
  lama-manga-onnx-dynamic（ogkalu，ONNX 转换）                    Apache License 2.0
  LaMa（advimman）                                                Apache License 2.0
- PP-OCRv5 mobile det / rec（PaddlePaddle）                       Apache License 2.0
- 本地翻译模型 Qwen3.5 / Qwen3.8 系列（阿里通义千问，GGUF 量化版由 unsloth / ggml-org 提供）
  Apache License 2.0（首次使用时按电脑配置下载，不随安装包分发）
- llama.cpp 本地推理引擎（ggml-org）                             MIT License

【字体】（均为 SIL Open Font License 1.1，授权全文见 fonts/licenses/）
- Noto Sans SC / Noto Sans TC（Google / Adobe，思源黑体同源）
- LXGW WenKai 霞鹜文楷（LXGW）
- ZCOOL KuaiLe 站酷快乐体（站酷）
- Ma Shan Zheng 马善政毛笔楷书（马善政）

【运行库】
- Python（PSF License）、NumPy（BSD）、OpenCV（Apache 2.0）、Pillow（MIT-CMU）
- ONNX Runtime（MIT）、DirectML（MIT，仅通用显卡版）
- NVIDIA CUDA Runtime / cuBLAS / cuFFT / cuDNN（NVIDIA 软件许可，按其再分发条款随附，仅 NVIDIA 加速版）
- pywebview（BSD）、pythonnet（MIT）、pypdfium2 / PDFium（Apache 2.0 / BSD）
- requests（Apache 2.0）、jaconv（MIT）、PyInstaller 引导程序（GPL 附带例外条款，允许商业分发）
"""


def run(cmd: list[str], **kw) -> None:
    print(">", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def env_python(env: str) -> str:
    p = os.path.join(ROOT, "envs", env, "Scripts", "python.exe")
    if not os.path.exists(p):
        raise SystemExit(f"缺少环境 envs/{env}，请先运行 powershell -File build/setup_envs.ps1 -Variant {env}")
    return p


def make_icon() -> str:
    """生成软件图标（渐变圆角方块 + 「汉」字，用自带的 OFL 字体绘制）。"""
    ico = os.path.join(BUILD, "assets", "app.ico")
    if os.path.exists(ico):
        return ico
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(os.path.dirname(ico), exist_ok=True)
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    grad = Image.new("RGBA", (size, size))
    gd = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / (size - 1)
        gd.line([(0, y), (size, y)], fill=(int(99 + (236 - 99) * t), int(102 + (72 - 102) * t), int(241 + (153 - 241) * t), 255))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([8, 8, size - 8, size - 8], radius=56, fill=255)
    img.paste(grad, (0, 0), mask)
    font = ImageFont.truetype(os.path.join(ROOT, "fonts", "NotoSansSC-VF.ttf"), 168)
    try:
        font.set_variation_by_axes([900])
    except Exception:  # noqa: BLE001
        pass
    d = ImageDraw.Draw(img)
    d.text((size / 2, size / 2 + 4), "汉", font=font, fill=(255, 255, 255, 255), anchor="mm")
    img.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return ico


def make_version_file(variant: str) -> str:
    v = [int(x) for x in VERSION.split(".")] + [0]
    title = VARIANTS[variant]["title"]
    text = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers=({v[0]}, {v[1]}, {v[2]}, 0), prodvers=({v[0]}, {v[1]}, {v[2]}, 0),
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('080404B0', [
      StringStruct('CompanyName', '一键汉化漫画'),
      StringStruct('FileDescription', '一键汉化漫画 {VERSION} {title}'),
      StringStruct('FileVersion', '{VERSION}'),
      StringStruct('InternalName', '{APP_NAME}'),
      StringStruct('OriginalFilename', '{APP_NAME}.exe'),
      StringStruct('ProductName', '一键汉化漫画'),
      StringStruct('ProductVersion', '{VERSION}')])]),
    VarFileInfo([VarStruct('Translation', [2052, 1200])])
  ]
)
"""
    p = os.path.join(BUILD, "work", variant, "version_info.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def pyinstaller(variant: str) -> str:
    cfg = VARIANTS[variant]
    py = env_python(cfg["env"])
    work = os.path.join(BUILD, "work", variant)
    dist = os.path.join(BUILD, "dist", variant)
    cmd = [
        py, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--onedir",
        "--name", APP_NAME, "--icon", make_icon(), "--version-file", make_version_file(variant),
        "--distpath", dist, "--workpath", work, "--specpath", work,
        "--paths", ROOT,
        "--add-data", f"{os.path.join(ROOT, 'hanhua', 'ui')}{os.pathsep}hanhua/ui",
        "--collect-submodules", "hanhua",
        "--hidden-import", "clr_loader", "--hidden-import", "pythonnet",
    ]
    for m in EXCLUDES:
        cmd += ["--exclude-module", m]
    cmd.append(os.path.join(BUILD, "launcher.py"))
    run(cmd, cwd=ROOT)
    return os.path.join(dist, APP_NAME)


def copy_resources(variant: str, app_dir: str) -> None:
    cfg = VARIANTS[variant]
    # 模型（放在 exe 旁边，runtime.app_root() 会优先找这里）
    mdst = os.path.join(app_dir, "models")
    if os.path.isdir(mdst):
        shutil.rmtree(mdst)
    for rel in cfg["models"] + COMMON_MODEL_FILES:
        src = os.path.join(ROOT, "models", rel)
        if not os.path.exists(src):
            raise SystemExit(f"缺少模型 {rel}，请先运行 build/fetch_assets.py")
        dst = os.path.join(mdst, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    for readme in ("detector/README.md", "manga-ocr/README.md", "lama/README.md"):
        src = os.path.join(ROOT, "models", readme)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(mdst, readme))
    # 字体 + 授权
    fdst = os.path.join(app_dir, "fonts")
    if os.path.isdir(fdst):
        shutil.rmtree(fdst)
    shutil.copytree(os.path.join(ROOT, "fonts"), fdst, ignore=shutil.ignore_patterns("*.part", ".tmp_*"))
    # CUDA DLL（只 NVIDIA 版）
    if variant == "nvidia":
        sp = os.path.join(ROOT, "envs", "cuda", "Lib", "site-packages", "nvidia")
        cdst = os.path.join(app_dir, "cuda")
        os.makedirs(cdst, exist_ok=True)
        for rel in CUDA_DLLS:
            shutil.copy2(os.path.join(sp, rel), os.path.join(cdst, os.path.basename(rel)))
        # TensorRT provider 依赖 TensorRT 运行库，我们不用，删掉省空间也避免加载报错
        for root, _, files in os.walk(app_dir):
            for f in files:
                if f.lower() == "onnxruntime_providers_tensorrt.dll":
                    os.remove(os.path.join(root, f))
    # 本地翻译引擎 llama.cpp：只带 llama-server 及其依赖（bench / cli / quantize 等工具不带）
    lsrc = os.path.join(ROOT, "runtimes", "llama", cfg["llama"])
    if not os.path.exists(os.path.join(lsrc, "llama-server.exe")):
        raise SystemExit("缺少 llama.cpp 运行库，请先运行 build/fetch_assets.py --only runtimes")
    ldst = os.path.join(app_dir, "runtimes", "llama", cfg["llama"])
    if os.path.isdir(os.path.join(app_dir, "runtimes")):
        shutil.rmtree(os.path.join(app_dir, "runtimes"))
    os.makedirs(ldst)
    for fn in os.listdir(lsrc):
        low = fn.lower()
        wanted = (low == "llama-server.exe" or low.startswith("license") or low == "llama_build.txt"
                  or (low.endswith(".dll") and (not low.endswith("-impl.dll") or low == "llama-server-impl.dll")))
        if wanted:
            shutil.copy2(os.path.join(lsrc, fn), os.path.join(ldst, fn))
    # 版本信息
    info = {"variant": variant, "title": cfg["title"], "version": VERSION,
            "built": _dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(app_dir, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    with open(os.path.join(app_dir, "第三方授权说明.txt"), "w", encoding="utf-8-sig") as f:
        f.write(THIRD_PARTY.format(version=VERSION))
    guide = os.path.join(BUILD, "使用说明.txt")
    if os.path.exists(guide):
        shutil.copy2(guide, os.path.join(app_dir, "使用说明.txt"))


def selftest(app_dir: str) -> dict:
    exe = os.path.join(app_dir, f"{APP_NAME}.exe")
    report = os.path.join(BUILD, "work", "selftest_report.json")
    if os.path.exists(report):
        os.remove(report)
    env = dict(os.environ, HANHUA_DATA=os.path.join(BUILD, "work", "selftest_data"))
    proc = subprocess.run([exe, "--selftest", report], env=env, timeout=900)
    try:
        with open(report, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"ok": False, "error": f"自检没有生成报告（退出码 {proc.returncode}）"}
    print(json.dumps(data, ensure_ascii=False, indent=2))
    if not data.get("ok"):
        raise SystemExit("自检失败")
    return data


def make_zip(variant: str, app_dir: str) -> str:
    rel_dir = os.path.join(BUILD, "release")
    os.makedirs(rel_dir, exist_ok=True)
    name = f"一键汉化漫画{VERSION.rsplit('.', 1)[0]}-{VARIANTS[variant]['title']}-v{VERSION}.zip"
    out = os.path.join(rel_dir, name)
    tmp = out + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, _, files in os.walk(app_dir):
            for f in files:
                p = os.path.join(root, f)
                arc = os.path.join(APP_NAME, os.path.relpath(p, app_dir))
                z.write(p, arc)
    os.replace(tmp, out)
    print(f"发行包：{out}（{os.path.getsize(out) / 1024**3:.2f} GB）")
    return out


def dir_size(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1024**3


def build_one(variant: str, args: argparse.Namespace) -> None:
    print(f"\n========== {variant}（{VARIANTS[variant]['title']}）==========", flush=True)
    app_dir = os.path.join(BUILD, "dist", variant, APP_NAME)
    if not args.skip_pyinstaller:
        app_dir = pyinstaller(variant)
    copy_resources(variant, app_dir)
    print(f"程序目录：{app_dir}（{dir_size(app_dir):.2f} GB）")
    if not args.skip_selftest:
        selftest(app_dir)
    if not args.no_zip:
        make_zip(variant, app_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=[*VARIANTS, "all"])
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--skip-pyinstaller", action="store_true")
    args = ap.parse_args()
    for v in (VARIANTS if args.variant == "all" else [args.variant]):
        build_one(v, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

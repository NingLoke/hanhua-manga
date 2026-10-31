# 第三方组件与资源

仓库根目录的 MIT 许可证仅适用于本项目自有源码。以下组件归各自作者，下载、使用与再分发须遵守对应版本附带的许可证。模型、运行库、Python 环境及 GGUF 文件不随源码仓库提交。

## 视觉与翻译模型

- [comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)：模型卡标注 Apache-2.0。
- [manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base) 与 [ONNX 转换版](https://huggingface.co/onnx-community/manga-ocr-base-ONNX)：模型卡标注 Apache-2.0。
- [AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting)：模型卡标注 MIT；[LaMa ONNX 转换版](https://huggingface.co/ogkalu/lama-manga-onnx-dynamic) 标注 Apache-2.0。
- [PaddlePaddle PP-OCRv5](https://huggingface.co/PaddlePaddle)：对应 ONNX 模型卡标注 Apache-2.0。
- 本地翻译模型来源与 SHA-256 见 [`hanhua/llm/catalog.py`](hanhua/llm/catalog.py)；权重应分别遵守原模型和量化发布者提供的许可，不由本项目重新授权。
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：MIT。下载脚本使用固定发行版本，运行文件不入库。

## 字体

Noto Sans SC / TC、LXGW WenKai、ZCOOL KuaiLe、Ma Shan Zheng 使用 SIL Open Font License。许可全文保留于 [`fonts/licenses/`](fonts/licenses/)，字体文件通过下载脚本取得。

## Python 与系统组件

使用 Python、NumPy、OpenCV、Pillow、requests、jaconv、ONNX Runtime、pywebview、pythonnet、pypdfium2 / PDFium 等组件。其许可证文件随各软件包发布。PyInstaller 及其引导程序适用其 GPL 与引导程序例外条款；不改变本项目自有源码的 MIT 许可。

若制作包含 NVIDIA CUDA / cuDNN、DirectML 或其它二进制依赖的发行包，请保留其实际版本的许可与再分发说明；这些二进制文件不适用本项目的 MIT 许可。

## 展示图片

[`docs/examples/`](docs/examples/) 的漫画对比图用于展示作者已有汉化结果，漫画版权归原作者，不以 MIT 重新授权。源码测试不依赖这些展示图片，也不包含整本漫画或用户的输入文件。

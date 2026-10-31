"""pytest 公共配置：路径、共享 fixture、markers，以及把 HANHUA_DATA 隔离到临时目录。

- 项目根目录（4.0/）和 tests/ 都加入 sys.path：测试里 ``import hanhua``、``import synth``、
  ``from tests import synth`` 三种写法都能用。
- HANHUA_DATA（``hanhua.runtime.data_dir()`` 的覆盖变量）：
  * conftest 一导入就先指向本次会话专用的临时目录 —— 连测试模块顶层代码、session / module
    级 fixture 也写不进用户真实的 ``%APPDATA%/HanhuaManga``；
  * 每个测试函数再由 autouse fixture 指向自己独立的空目录（设置 / 缓存 / 任务互不污染），
    需要路径时请求 ``hanhua_data`` fixture。
- markers：``network``（需要联网）、``integration``（端到端 / 较慢）、``gpu``（需要 CUDA / DirectML）。
  当前环境没有 GPU 后端时 gpu 用例自动 skip；设环境变量 ``HANHUA_OFFLINE=1`` 时 network 用例自动 skip；
  也可以用 ``-m "not network"`` 之类的标准写法筛选。
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

log = logging.getLogger("hanhua.tests")

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
REAL_DIR = FIXTURES_DIR / "real"
OUT_ROOT = TESTS_DIR / "out"

# tests/ 先插、项目根后插 → 项目根排在最前（避免 tests 里的同名文件遮住正式模块）
for _p in (str(TESTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- 会话级 HANHUA_DATA：导入即生效（早于任何测试模块 / fixture） -----------------------------
_ORIG_HANHUA_DATA = os.environ.get("HANHUA_DATA")
_SESSION_DATA = Path(tempfile.mkdtemp(prefix="hanhua_test_data_"))
os.environ["HANHUA_DATA"] = str(_SESSION_DATA)


class FixturePath(str):
    """str 子类的路径：既能直接当字符串用（``hanhua.imageio.read(p)``、``p.endswith(".jpg")``），
    也带常用的 pathlib 属性（``p.name`` / ``p.stem`` / ``p.suffix`` / ``p.parent`` / ``p / "x"``），
    这样不同模块的测试不管按 str 还是按 Path 的习惯写都不会出错。"""

    __slots__ = ()

    @property
    def path(self) -> Path:
        return Path(self)

    @property
    def name(self) -> str:
        return os.path.basename(self)

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self))[0]

    @property
    def suffix(self) -> str:
        return os.path.splitext(self)[1]

    @property
    def parent(self) -> Path:
        return Path(self).parent

    def exists(self) -> bool:
        return os.path.exists(self)

    def is_file(self) -> bool:
        return os.path.isfile(self)

    def is_dir(self) -> bool:
        return os.path.isdir(self)

    def read_bytes(self) -> bytes:
        with open(self, "rb") as f:
            return f.read()

    def __truediv__(self, other: str | os.PathLike[str]) -> Path:
        return Path(self) / other


# ---- pytest 钩子 ---------------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: 需要联网（翻译引擎实测等）；离线时用 -m \"not network\" 跳过")
    config.addinivalue_line("markers", "integration: 端到端 / 较慢的集成测试")
    config.addinivalue_line("markers", "gpu: 需要 CUDA 或 DirectML 推理后端；没有 GPU 后端时自动 skip")


def pytest_unconfigure(config: pytest.Config) -> None:
    if _ORIG_HANHUA_DATA is None:
        os.environ.pop("HANHUA_DATA", None)
    else:
        os.environ["HANHUA_DATA"] = _ORIG_HANHUA_DATA
    shutil.rmtree(_SESSION_DATA, ignore_errors=True)


def _gpu_providers() -> list[str]:
    try:
        import onnxruntime as ort  # 只在确实有 gpu 用例时才导入

        return [p for p in ort.get_available_providers() if p in ("CUDAExecutionProvider", "DmlExecutionProvider")]
    except Exception as e:  # noqa: BLE001 - 没装 / 装坏了都当作没有 GPU
        log.debug("onnxruntime 不可用：%s", e)
        return []


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if any(item.get_closest_marker("gpu") for item in items) and not _gpu_providers():
        skip_gpu = pytest.mark.skip(reason="当前环境没有 CUDA / DirectML 推理后端")
        for item in items:
            if item.get_closest_marker("gpu"):
                item.add_marker(skip_gpu)
    if os.environ.get("HANHUA_OFFLINE", "").strip() not in ("", "0"):
        skip_net = pytest.mark.skip(reason="HANHUA_OFFLINE=1：跳过需要联网的用例")
        for item in items:
            if item.get_closest_marker("network"):
                item.add_marker(skip_net)


# ---- fixtures ------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_hanhua_data(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """每个测试一个独立的空数据目录；测试结束 monkeypatch 自动恢复成会话目录。"""
    d = tmp_path_factory.mktemp("hanhua_data", numbered=True)
    monkeypatch.setenv("HANHUA_DATA", str(d))
    yield d


@pytest.fixture
def hanhua_data(_isolate_hanhua_data: Path) -> Path:
    """当前测试的 HANHUA_DATA 目录（与 ``hanhua.runtime.data_dir()`` 一致）。"""
    return _isolate_hanhua_data


@pytest.fixture(scope="session")
def project_root() -> Path:
    """4.0 项目根目录（含 hanhua/ models/ fonts/ tests/）。"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def real_pages() -> list[FixturePath]:
    """tests/fixtures/real/ 下的真实页面路径（按文件名排序）。素材缺失时（如新克隆的仓库）整体 skip。"""
    pages = sorted(
        (p for p in REAL_DIR.glob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS),
        key=lambda p: p.name,
    ) if REAL_DIR.is_dir() else []
    if not pages:
        pytest.skip("tests/fixtures/real/ 里没有测试页面（版权素材不入库，需本地放置）")
    return [FixturePath(str(p)) for p in pages]


@pytest.fixture(scope="session")
def real_images(real_pages: list[FixturePath]) -> dict[str, "object"]:
    """真实页面解码后的 BGR 图（key = 文件名去扩展名，如 "p002"），整个会话只解码一次。
    注意：是共享对象，测试里要修改请先 .copy()。"""
    from hanhua import imageio as hio

    return {p.stem: hio.read(p) for p in real_pages}


@pytest.fixture
def out_dir(request: pytest.FixtureRequest) -> Path:
    """调试输出目录 tests/out/<模块名>/（test_detect.py → tests/out/detect/），自动创建。"""
    mod = request.module.__name__.rsplit(".", 1)[-1]
    name = mod[5:] if mod.startswith("test_") else mod
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d

"""任务引擎测试：视觉 / 翻译 / 去字 / 排版全部用假组件（快、不联网、不需要 GPU），书本读写用真实 bookio。"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from hanhua import imageio
from hanhua.config import SettingsStore
from hanhua.engine import ComponentFactory, Engine
from hanhua.types import HanhuaError, TextBlock, TranslationUnavailable


class Calls:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.n: dict[str, int] = {}

    def hit(self, k: str) -> None:
        with self.lock:
            self.n[k] = self.n.get(k, 0) + 1


class FakeFactory(ComponentFactory):
    """假组件：每页一个文字块；全黑的页 OCR 报错；translator 可按需先失败 fail_times 次。"""

    def __init__(self, calls: Calls, fail_times: int = 0, slow: float = 0.0) -> None:
        super().__init__()
        self.calls, self.fail_times, self.slow = calls, fail_times, slow
        self._label = "假设备"

    def runtime(self, settings: dict):  # noqa: D401
        return SimpleNamespace(device="cpu", label="假设备", hw=None, release=lambda: None)

    def profile(self, settings: dict):
        return SimpleNamespace(name="balanced", cpu_workers=2)

    def detector(self, settings: dict):
        calls, slow = self.calls, self.slow

        def detect(img, reading="rtl"):
            calls.hit("detect")
            time.sleep(slow)
            return [TextBlock(id=0, box=(4, 4, 40, 40), kind="bubble", bubble_box=(0, 0, 48, 48))], [(0, 0, 48, 48)]

        return SimpleNamespace(detect=detect)

    def ocr(self, settings: dict):
        calls = self.calls

        def recognize(img, blocks):
            calls.hit("ocr")
            if float(img.mean()) < 1.0:
                raise RuntimeError("坏页")
            for b in blocks:
                b.src_text, b.ocr_conf = "テスト", 1.0

        return SimpleNamespace(recognize=recognize)

    def translator(self, settings: dict):
        factory = self

        def translate_blocks(pages, src_lang, cancel=None):
            factory.calls.hit("translate")
            if factory.fail_times > 0:
                factory.fail_times -= 1
                raise TranslationUnavailable("翻译引擎暂时都不可用（测试）")
            for page in pages:
                for b in page:
                    if not b.skip and b.src_text and not b.translation:
                        b.translation = "测试"

        return SimpleNamespace(translate_blocks=translate_blocks, close=lambda: None)

    def inpainter(self, settings: dict):
        calls = self.calls

        def clean(img, blocks):
            calls.hit("clean")
            return img.copy(), np.zeros(img.shape[:2], np.uint8)

        return SimpleNamespace(clean=clean)

    def typesetter(self, settings: dict):
        calls = self.calls

        def render(clean, blocks, *a, **k):
            calls.hit("render")
            return clean.copy()

        return SimpleNamespace(render=render)

    def cpu_workers(self, settings: dict) -> int:
        return 2


def make_book(root: Path, n: int, black: tuple[int, ...] = ()) -> str:
    d = root / "书"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((64, 48, 3), 0 if i in black else 200, np.uint8)
        imageio.write(str(d / f"{i + 1:03d}.png"), img)
    return str(d)


def wait_status(eng: Engine, jid: str, statuses: tuple[str, ...], timeout: float = 20.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = next(j for j in eng.list_jobs() if j["id"] == jid)
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"等待状态 {statuses} 超时，当前 {job['status']}：{job.get('message')}")


def new_engine(tmp: Path, calls: Calls, **kw) -> Engine:
    store = SettingsStore(str(tmp / "settings.json"))
    store.update({"output": {"dir": str(tmp / "out"), "format": "folder"}, "general": {"auto_resume": False}})
    fail_times = kw.pop("fail_times", 0)
    slow = kw.pop("slow", 0.0)
    return Engine(store, None, work_root=str(tmp / "jobs"), components=FakeFactory(calls, fail_times, slow), **kw)


def test_full_run_and_export(tmp_path: Path) -> None:
    calls = Calls()
    eng = new_engine(tmp_path, calls)
    try:
        jid = eng.add_paths([make_book(tmp_path, 4)])[0]["id"]
        eng.start()
        job = wait_status(eng, jid, ("done", "error"))
        assert job["status"] == "done" and job["done"] == 4 and job["failed"] == 0
        assert os.path.isdir(job["out_path"]) and len(os.listdir(job["out_path"])) == 4
        assert calls.n["detect"] == 4 and calls.n["render"] == 4
    finally:
        eng.shutdown()


def test_same_file_imports_create_independent_direction_versions(tmp_path: Path) -> None:
    calls = Calls()
    eng = new_engine(tmp_path, calls)
    try:
        source = make_book(tmp_path, 2)
        jid = eng.add_paths([source])[0]["id"]
        eng.start()
        first = wait_status(eng, jid, ("done", "error"))
        assert first["status"] == "done"
        first_bytes = {p.name: p.read_bytes() for p in Path(first["out_path"]).iterdir()}
        before = dict(calls.n)
        eng.store.update({"source": {"reading_order": "rtl"}})
        assert eng._jobs[jid].settings["typeset"]["column_order"] == "ltr"
        assert calls.n == before
        second = eng.add_paths([source])[0]
        assert second["id"] != jid and second["done"] == 0
        second = wait_status(eng, second["id"], ("done", "error"))
        assert second["status"] == "done"
        assert "竖排左到右" in first["out_path"]
        assert "竖排右到左" in second["out_path"]
        assert second["out_path"] != first["out_path"]
        assert {p.name: p.read_bytes() for p in Path(first["out_path"]).iterdir()} == first_bytes
        assert calls.n["detect"] == before["detect"] + 2
        restored = SettingsStore(eng.store.path)
        assert restored.get()["typeset"]["column_order"] == "rtl"
    finally:
        eng.shutdown()


def test_direction_change_discards_inflight_old_layout(tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()
    orders = []

    class BlockingFactory(FakeFactory):
        def typesetter(self, settings):
            order = settings["typeset"]["column_order"]
            def render(clean, blocks):
                orders.append(order)
                if len(orders) == 1:
                    entered.set()
                    assert release.wait(10)
                return clean.copy()
            return SimpleNamespace(render=render)

    store = SettingsStore(str(tmp_path / "settings.json"))
    store.update({"typeset": {"column_order": "rtl"}, "output": {"dir": str(tmp_path / "out")}})
    eng = Engine(store, components=BlockingFactory(Calls()), work_root=str(tmp_path / "jobs"))
    try:
        jid = eng.add_paths([make_book(tmp_path, 1)])[0]["id"]
        eng.start()
        assert entered.wait(10)
        store.update({"typeset": {"column_order": "ltr"}})
        eng.rerender_all(jid)
        release.set()
        assert wait_status(eng, jid, ("done", "error"))["status"] == "done"
        assert orders == ["rtl", "ltr"]
    finally:
        release.set()
        eng.shutdown()


def test_new_output_setting_does_not_change_previous_version(tmp_path: Path) -> None:
    calls = Calls()
    eng = new_engine(tmp_path, calls)
    try:
        eng.store.update({"output": {"format": "zip"}})
        source = make_book(tmp_path, 2)
        jid = eng.add_paths([source])[0]["id"]
        eng.start()
        old = wait_status(eng, jid, ("done", "error"))
        assert old["status"] == "done" and os.path.isfile(old["out_path"])
        before = dict(calls.n)
        eng.store.update({"output": {"format": "folder"}})
        assert eng._jobs[jid].settings["output"]["format"] == "zip"
        assert calls.n == before
        new_id = eng.add_paths([source])[0]["id"]
        new = wait_status(eng, new_id, ("done", "error"))
        assert new["status"] == "done" and os.path.isdir(new["out_path"])
        assert len(os.listdir(new["out_path"])) == 2
        assert os.path.isfile(old["out_path"])
        assert new_id != jid and calls.n["detect"] == before["detect"] + 2
    finally:
        eng.shutdown()


def test_restart_preserves_each_tasks_own_settings(tmp_path: Path) -> None:
    calls = Calls()
    eng = new_engine(tmp_path, calls)
    eng.store.update({"typeset": {"column_order": "rtl"}, "output": {"format": "zip"}})
    jid = eng.add_paths([make_book(tmp_path, 2)])[0]["id"]
    eng.start()
    assert wait_status(eng, jid, ("done", "error"))["status"] == "done"
    settings_path = eng.store.path
    eng.shutdown()
    before = dict(calls.n)
    store = SettingsStore(settings_path)
    store.update({"typeset": {"column_order": "ltr"}, "output": {"format": "folder"},
                  "general": {"auto_resume": True}})
    restored = Engine(store, components=FakeFactory(calls), work_root=str(tmp_path / "jobs"))
    try:
        state = wait_status(restored, jid, ("done", "error"))
        assert state["status"] == "done" and os.path.isfile(state["out_path"])
        assert restored._jobs[jid].settings["typeset"]["column_order"] == "rtl"
        assert restored._jobs[jid].settings["output"]["format"] == "zip"
        for stage in ("detect", "ocr", "translate", "clean", "render"):
            assert calls.n[stage] == before[stage]
    finally:
        restored.shutdown()


def test_remove_emits_only_removal_and_reimport_starts_fresh(tmp_path: Path) -> None:
    calls, events = Calls(), []
    eng = new_engine(tmp_path, calls)
    eng._on_event = events.append
    try:
        source = make_book(tmp_path, 2)
        jid = eng.add_paths([source])[0]["id"]
        eng.start()
        old = wait_status(eng, jid, ("done",))
        eng._tick_once()
        events.clear()
        workdir = eng._jobs[jid].workdir
        eng.remove(jid)
        eng._tick_once()
        time.sleep(0.2)
        assert not eng.list_jobs()
        assert not os.path.exists(workdir)
        assert os.path.isdir(source) and os.path.isdir(old["out_path"])
        assert any(e.get("type") == "removed" and e.get("job_id") == jid for e in events)
        assert not any(e.get("type") == "job" and e.get("job", {}).get("id") == jid for e in events)
        eng.remove(jid)  # 重复删除应当安全，不把幽灵卡片变成报错。
        before = calls.n["detect"]
        added = eng.add_paths([source])[0]
        assert added["id"] != jid and added["done"] == 0
        again = wait_status(eng, added["id"], ("done",))
        assert again["out_path"] != old["out_path"]
        assert calls.n["detect"] == before + 2
    finally:
        eng.shutdown()


def test_retained_old_workdir_never_reuses_finished_status(tmp_path: Path) -> None:
    calls = Calls()
    eng = new_engine(tmp_path, calls)
    try:
        source = make_book(tmp_path, 1)
        jid = eng.add_paths([source])[0]["id"]
        eng.start()
        wait_status(eng, jid, ("done",))
        workdir = eng._jobs[jid].workdir
        eng.remove(jid, delete_files=False)
        assert os.path.isdir(workdir)
        before = calls.n["detect"]
        new = eng.add_paths([source])[0]
        assert new["id"] != jid and new["done"] == 0
        wait_status(eng, new["id"], ("done",))
        assert calls.n["detect"] == before + 1
    finally:
        eng.shutdown()


def test_same_name_sources_and_same_batch_dedup(tmp_path: Path) -> None:
    eng = new_engine(tmp_path, Calls())
    try:
        first = make_book(tmp_path / "A", 1)
        second = make_book(tmp_path / "B", 1)
        jobs = eng.add_paths([first, first, second])
        assert len(jobs) == 2 and jobs[0]["id"] != jobs[1]["id"]
        assert {j["source"] for j in jobs} == {first, second}
    finally:
        eng.shutdown()


def test_parallel_exports_do_not_overwrite_same_name(tmp_path: Path) -> None:
    eng = new_engine(tmp_path, Calls())
    try:
        source = make_book(tmp_path, 1)
        ids = [eng.add_paths([source])[0]["id"] for _ in range(2)]
        for jid in ids:
            job = eng._jobs[jid]
            job.pages[0].stage = "rendered"
            Path(job.out_file(0)).parent.mkdir(parents=True, exist_ok=True)
            imageio.write(job.out_file(0), np.full((64, 48, 3), 200, np.uint8))
        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(eng.export, ids))
        assert paths[0] != paths[1]
        assert all(os.path.isdir(p) and len(os.listdir(p)) == 1 for p in paths)
        assert all("竖排左到右" in p for p in paths)
    finally:
        eng.shutdown()


def test_export_refuses_to_silently_include_unfinished_pages(tmp_path: Path) -> None:
    eng = new_engine(tmp_path, Calls())
    try:
        jid = eng.add_paths([make_book(tmp_path, 3)])[0]['id']
        with pytest.raises(HanhuaError, match='3 页尚未完成'):
            eng.export(jid)
        assert not list((tmp_path / 'out').glob('*'))
    finally:
        eng.shutdown()


def test_translation_unavailable_pauses_then_auto_retries(tmp_path: Path) -> None:
    calls = Calls()
    eng = new_engine(tmp_path, calls, fail_times=1, retry_delays=(0.5,))
    try:
        jid = eng.add_paths([make_book(tmp_path, 3)])[0]["id"]
        eng.start()
        job = wait_status(eng, jid, ("paused",))
        assert "翻译" in job["message"]                       # 暂停原因是中文、说清楚
        job = wait_status(eng, jid, ("done",))                 # 0.5 秒后自动重试成功
        assert job["done"] == 3
    finally:
        eng.shutdown()


def test_single_bad_page_does_not_stop_others(tmp_path: Path) -> None:
    calls = Calls()
    eng = new_engine(tmp_path, calls)
    try:
        jid = eng.add_paths([make_book(tmp_path, 4, black=(1,))])[0]["id"]
        eng.start()
        job = wait_status(eng, jid, ("done", "error"))
        assert job["failed"] == 1 and job["done"] + job["failed"] == 4
        assert job["status"] == "done" and os.path.isdir(job["out_path"])   # 坏页用原图导出
    finally:
        eng.shutdown()


def test_pause_and_resume(tmp_path: Path) -> None:
    calls = Calls()
    eng = new_engine(tmp_path, calls, slow=0.15)
    try:
        jid = eng.add_paths([make_book(tmp_path, 8)])[0]["id"]
        eng.start()
        time.sleep(0.3)
        eng.pause(jid)
        job = wait_status(eng, jid, ("paused",))
        n1 = calls.n.get("detect", 0)
        time.sleep(0.6)
        assert calls.n.get("detect", 0) <= n1 + 1              # 暂停后最多做完手上那一页
        eng.resume(jid)
        job = wait_status(eng, jid, ("done",), timeout=30)
        assert job["done"] == 8 and calls.n["detect"] == 8     # 不重复检测
    finally:
        eng.shutdown()


def test_update_page_only_rerenders(tmp_path: Path) -> None:
    calls = Calls()
    eng = new_engine(tmp_path, calls)
    try:
        jid = eng.add_paths([make_book(tmp_path, 2)])[0]["id"]
        eng.start()
        wait_status(eng, jid, ("done",))
        before = dict(calls.n)
        page = eng.get_page(jid, 0)
        blocks = page["blocks"]
        blocks[0]["translation"] = "人工修改的译文"
        newp = eng.update_page(jid, 0, blocks)               # 立即返回，渲染线程异步重排（界面不卡）
        assert newp["blocks"][0]["translation"] == "人工修改的译文"
        t0 = time.time()
        while calls.n["render"] < before["render"] + 1 and time.time() - t0 < 10:
            time.sleep(0.05)
        assert calls.n["render"] == before["render"] + 1
        assert eng.get_page(jid, 0)["stage"] == "rendered"
        for k in ("detect", "ocr", "translate"):
            assert calls.n.get(k, 0) == before.get(k, 0)        # 不重检测、不重识别、不重翻译
    finally:
        eng.shutdown()

import ctypes
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows 子进程生命周期回归')
def test_model_child_stops_when_launcher_exits(tmp_path):
    pid_file = tmp_path / 'child.pid'
    project = str(Path(__file__).resolve().parents[1])
    code = (
        'import sys, subprocess; '
        f'sys.path.insert(0, {project!r}); '
        'from pathlib import Path; from hanhua.llm.manager import LlamaServer; '
        'server=LlamaServer(); '
        "server._proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'], "
        'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
        f'Path({str(pid_file)!r}).write_text(str(server._proc.pid))'
    )
    subprocess.run([sys.executable, '-X', 'utf8', '-c', code], check=True, timeout=20)
    child = int(pid_file.read_text())
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x00100000, False, child)
    if handle:
        try:
            assert kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))

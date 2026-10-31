"""打包入口。

  一键汉化漫画.exe                     启动界面
  一键汉化漫画.exe 某本漫画.zip ...     启动界面并直接加入这些文件（拖到 exe 图标上）
  一键汉化漫画.exe --selftest 报告.json  无界面自检（打包脚本用）
"""
import multiprocessing
import os
import sys


def _ensure_streams() -> None:
    # --windowed 打包时 stdout/stderr 是 None，任何 print / logging 兜底输出都会崩
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def main() -> None:
    multiprocessing.freeze_support()
    _ensure_streams()
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        from hanhua.selftest import main as selftest_main

        sys.exit(selftest_main(sys.argv[2:]))
    from hanhua.app import main as app_main

    sys.exit(app_main() or 0)


if __name__ == "__main__":
    main()

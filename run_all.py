#!/usr/bin/env python3
"""命令入口；实现位于 hicache 包。"""
from hicache.runner import main
import sys

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"任务配置错误：{exc}", file=sys.stderr)
        raise SystemExit(1)

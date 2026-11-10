#!/usr/bin/env python3
"""命令入口；实现位于 hicache 包。"""
from hicache.benchmark import main
import sys

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

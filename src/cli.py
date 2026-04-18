"""ai_suite CLI entrypoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .e6ai.fetch import cache_path, fetch


def _out_dir(site: str, tag: str) -> Path:
    return Path("out") / site / tag.replace("/", "_").replace("\\", "_")


def cmd_e6ai_fetch(args: argparse.Namespace) -> int:
    fetch(args.tag, mode=args.mode, proxy=args.proxy)
    return 0


def cmd_e6ai_analyze(args: argparse.Namespace) -> int:
    from .analysis.report import render

    cp = cache_path(args.tag)
    if not cp.exists():
        print(f"no cache at {cp} — run fetch first", file=sys.stderr)
        return 2
    out = _out_dir("e6ai", args.tag)
    report = render(args.tag, cp, out)
    print(f"report: {report}")
    return 0


def cmd_e6ai_run(args: argparse.Namespace) -> int:
    rc = cmd_e6ai_fetch(args)
    if rc != 0:
        return rc
    return cmd_e6ai_analyze(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai_suite")
    sub = p.add_subparsers(dest="site", required=True)

    e6 = sub.add_parser("e6ai", help="e6ai.net tools")
    e6sub = e6.add_subparsers(dest="action", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--tag", required=True, help="artist tag to search")
        sp.add_argument("--mode", choices=("full", "incremental"), default=None,
                        help="fetch mode (default: incremental if cache exists, else full)")
        sp.add_argument("--proxy", default=None,
                        help="proxy URL; overrides E6AI_PROXY and AI_SUITE_PROXY")

    pf = e6sub.add_parser("fetch", help="fetch posts into data/e6ai/<tag>.json")
    add_common(pf)
    pf.set_defaults(func=cmd_e6ai_fetch)

    pa = e6sub.add_parser("analyze", help="analyze cached posts into out/e6ai/<tag>/")
    pa.add_argument("--tag", required=True)
    pa.set_defaults(func=cmd_e6ai_analyze)

    pr = e6sub.add_parser("run", help="fetch + analyze in one go")
    add_common(pr)
    pr.set_defaults(func=cmd_e6ai_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

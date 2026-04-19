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


def cmd_e6ai_breakers(args: argparse.Namespace) -> int:
    from .analysis.breakers import render_breakers

    cp = cache_path(args.tag)
    if not cp.exists():
        print(f"no cache at {cp} — run fetch first", file=sys.stderr)
        return 2
    out = _out_dir("e6ai", args.tag)
    path = render_breakers(args.tag, cp, out)
    print(f"breakers: {path}")
    return 0


def cmd_e6ai_peers(args: argparse.Namespace) -> int:
    from .analysis.correlations import per_tag_stats
    from .analysis.load import load_posts
    from .analysis.peers import render_peers_report
    from .e6ai.peers_fetch import fetch_tag_benchmarks, fetch_top_peers

    cp = cache_path(args.tag)
    if not cp.exists():
        print(f"no cache at {cp} — run fetch first", file=sys.stderr)
        return 2

    user_posts_df, user_tags_df = load_posts(cp)
    if user_posts_df.empty:
        print("user cache is empty", file=sys.stderr)
        return 2
    sample_size = min(len(user_posts_df), args.peer_cap)

    # User's top tags by lift, restricted to general+species (most actionable)
    user_tag_stats = per_tag_stats(user_posts_df, user_tags_df)
    if user_tag_stats.empty:
        print("no per-tag stats yet for user", file=sys.stderr)
        return 2
    bench_pool = user_tag_stats[user_tag_stats["category"].isin(["general", "species"])]
    bench_tags = bench_pool.sort_values("lift_favs", ascending=False).head(args.bench_tags)["tag"].tolist()

    exclude_raw = args.exclude_self if args.exclude_self else args.tag
    exclude_list = [n.strip() for n in exclude_raw.split(",") if n.strip()]
    print(f"fetching top {args.top} peers (sample {sample_size}/peer; excluding {exclude_list})...", file=sys.stderr)
    peers = fetch_top_peers(
        sample_size=sample_size,
        top_n=args.top,
        exclude_uploaders=exclude_list,
        proxy=args.proxy,
        refresh=args.refresh,
    )

    print(f"fetching site benchmarks for {len(bench_tags)} tags...", file=sys.stderr)
    tag_samples = fetch_tag_benchmarks(
        tags=bench_tags,
        sample_per_tag=args.bench_sample,
        proxy=args.proxy,
        refresh=args.refresh,
    )

    out = _out_dir("e6ai", args.tag)
    path = render_peers_report(
        user_tag=args.tag,
        user_cache_path=cp,
        peers=peers,
        tag_samples=tag_samples,
        out_dir=out,
        bench_tag_count=len(bench_tags),
    )
    print(f"peers: {path}")
    return 0


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

    pb = e6sub.add_parser("breakers", help="surface posts that beat or busted their tag mix")
    pb.add_argument("--tag", required=True)
    pb.set_defaults(func=cmd_e6ai_breakers)

    pp = e6sub.add_parser("peers", help="compare your stats to top e6ai uploaders + benchmark your tags vs site")
    pp.add_argument("--tag", required=True, help="your artist/director tag (used for output dir + cache lookup)")
    pp.add_argument("--top", type=int, default=10, help="number of top uploaders to compare against")
    pp.add_argument("--peer-cap", type=int, default=2000,
                    help="hard cap on per-peer sample size (default 2000; effective sample is min of this and your post count)")
    pp.add_argument("--bench-tags", type=int, default=20,
                    help="number of your top-lift general/species tags to benchmark vs site")
    pp.add_argument("--bench-sample", type=int, default=320,
                    help="site posts to fetch per benchmarked tag (max 320 = one API page)")
    pp.add_argument("--exclude-self", default=None,
                    help="uploader name(s) to drop from the leaderboard. Defaults to --tag with case/underscore-insensitive match (e.g. --tag ayo_keito automatically excludes uploader AyoKeito). Pass comma-separated names for multiple.")
    pp.add_argument("--refresh", action="store_true",
                    help="ignore cached peer/tag samples; re-fetch from API")
    pp.add_argument("--proxy", default=None, help="proxy URL")
    pp.set_defaults(func=cmd_e6ai_peers)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

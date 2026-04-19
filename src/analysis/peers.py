"""Peer comparison: leaderboard, site-wide tag benchmark, peer overlap matrix."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tabulate import tabulate

from .correlations import per_tag_stats
from .load import build_dataframes, load_posts


# ---------- aggregation primitives ----------

def aggregate_corpus(posts: list[dict], owner: str) -> dict:
    """Compute headline stats for a corpus of posts."""
    if not posts:
        return {
            "owner": owner, "n": 0, "span_days": 0, "posts_per_day": 0.0,
            "mean_score": 0.0, "median_score": 0.0,
            "mean_favs": 0.0, "median_favs": 0.0, "explicit_share": 0.0, "solo_share": 0.0,
        }
    posts_df, _ = build_dataframes(posts)
    n = len(posts_df)
    explicit_share = (posts_df["rating"] == "e").mean() if "rating" in posts_df else 0.0
    solo_share = posts_df["tags_general"].fillna("").str.split().apply(lambda xs: "solo" in xs).mean()
    if posts_df["created_at"].notna().any():
        span_days = max((posts_df["created_at"].max() - posts_df["created_at"].min()).days, 1)
    else:
        span_days = 1
    return {
        "owner": owner,
        "n": n,
        "span_days": int(span_days),
        "posts_per_day": float(n) / span_days,
        "mean_score": float(posts_df["score_total"].mean()),
        "median_score": float(posts_df["score_total"].median()),
        "mean_favs": float(posts_df["fav_count"].mean()),
        "median_favs": float(posts_df["fav_count"].median()),
        "explicit_share": float(explicit_share),
        "solo_share": float(solo_share),
    }


def build_leaderboard(user_posts: list[dict], user_name: str, peers: list[tuple[dict, list[dict]]]) -> pd.DataFrame:
    rows = [aggregate_corpus(user_posts, user_name + "  (you)")]
    for u, posts in peers:
        rows.append(aggregate_corpus(posts, u.get("name", "?")))
    df = pd.DataFrame(rows)
    df.sort_values("mean_favs", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "rank", df.index + 1)
    return df


# ---------- tag benchmark ----------

def _tag_mean(posts: list[dict]) -> tuple[int, float, float]:
    if not posts:
        return 0, 0.0, 0.0
    df, _ = build_dataframes(posts)
    return len(df), float(df["score_total"].mean()), float(df["fav_count"].mean())


def build_tag_benchmark(
    user_tag_stats: pd.DataFrame,
    user_posts_df: pd.DataFrame,
    site_samples: dict[str, list[dict]],
) -> pd.DataFrame:
    """For each benchmarked tag: user mean vs site mean, plus user's lift relative to site."""
    rows = []
    for tag, posts in site_samples.items():
        site_n, site_mean_score, site_mean_favs = _tag_mean(posts)
        user_row = user_tag_stats[user_tag_stats["tag"] == tag]
        if user_row.empty or site_n == 0:
            continue
        user_n = int(user_row.iloc[0]["n_posts"])
        user_mean_score = float(user_row.iloc[0]["mean_score"])
        user_mean_favs = float(user_row.iloc[0]["mean_favs"])
        rows.append({
            "tag": tag,
            "category": user_row.iloc[0].get("category", ""),
            "your_n": user_n,
            "site_n": site_n,
            "your_mean_favs": user_mean_favs,
            "site_mean_favs": site_mean_favs,
            "delta_favs": user_mean_favs - site_mean_favs,
            "your_vs_site_pct": (user_mean_favs / site_mean_favs - 1.0) * 100 if site_mean_favs > 0 else 0.0,
            "your_mean_score": user_mean_score,
            "site_mean_score": site_mean_score,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values("delta_favs", ascending=False, inplace=True)
    return df


# ---------- peer overlap matrix ----------

def build_peer_overlap(
    user_top_tags: list[str],
    peers: list[tuple[dict, list[dict]]],
) -> pd.DataFrame:
    """Rows = your top tags. Columns = peer names. Cells = share of peer's posts using that tag (%)."""
    rows = []
    for tag in user_top_tags:
        row = {"tag": tag}
        for u, posts in peers:
            if not posts:
                row[u["name"]] = 0.0
                continue
            df, _ = build_dataframes(posts)
            mask = df["tags_general"].fillna("").str.split().apply(lambda xs, t=tag: t in xs)
            mask |= df["tags_species"].fillna("").str.split().apply(lambda xs, t=tag: t in xs)
            row[u["name"]] = float(mask.mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- rendering ----------

def _md(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    if df.empty:
        return "_(no data)_\n"
    return tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=floatfmt) + "\n"


def render_peers_report(
    user_tag: str,
    user_cache_path: Path,
    peers: list[tuple[dict, list[dict]]],
    tag_samples: dict[str, list[dict]],
    out_dir: Path,
    bench_tag_count: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load user data
    user_posts_df, user_tags_df = load_posts(user_cache_path)
    user_posts_raw = []
    import json as _json
    user_posts_raw = _json.loads(user_cache_path.read_text(encoding="utf-8"))
    user_tag_stats = per_tag_stats(user_posts_df, user_tags_df)

    # Leaderboard
    leaderboard = build_leaderboard(user_posts_raw, user_tag, peers)

    # Tag benchmark
    tag_bench = build_tag_benchmark(user_tag_stats, user_posts_df, tag_samples)

    # Peer overlap (for the same tags we benchmarked)
    overlap = build_peer_overlap(list(tag_samples.keys()), peers)

    leaderboard.to_csv(out_dir / "peers_leaderboard.csv", index=False)
    if not tag_bench.empty:
        tag_bench.to_csv(out_dir / "peers_tag_benchmark.csv", index=False)
    if not overlap.empty:
        overlap.to_csv(out_dir / "peers_overlap.csv", index=False)

    you_row = leaderboard[leaderboard["owner"].str.startswith(user_tag)]
    your_rank = int(you_row.iloc[0]["rank"]) if not you_row.empty else None
    your_mean_favs = float(you_row.iloc[0]["mean_favs"]) if not you_row.empty else None

    lines: list[str] = []
    lines.append(f"# Peer comparison — `{user_tag}` vs top e6ai uploaders\n")

    # Issues section computed first so it can lead the report
    issues = _compute_issues(user_tag, leaderboard, tag_bench, overlap)
    if issues:
        lines.append("## Issues found\n")
        lines.append("_Most-critical findings up front. These are the things to actually fix._\n")
        for i in issues:
            lines.append(f"- {i}")
        lines.append("")

    lines.append("## Leaderboard (mean fav_count, descending)\n")
    lines.append(
        "_Each peer sampled at the same N as you. **`span_days` is critical context** — peers' samples cover wildly different time windows. "
        "A peer whose 415-post sample spans 30 days is publishing ~14×/day; yours spans years. Compare `posts_per_day`, not just `mean_favs`._\n"
    )
    pretty = leaderboard.copy()
    pretty["mean_score"] = pretty["mean_score"].round(0)
    pretty["median_score"] = pretty["median_score"].round(0)
    pretty["mean_favs"] = pretty["mean_favs"].round(0)
    pretty["median_favs"] = pretty["median_favs"].round(0)
    pretty["posts_per_day"] = pretty["posts_per_day"].round(2)
    pretty["explicit_share"] = (pretty["explicit_share"] * 100).round(0).astype(int).astype(str) + "%"
    pretty["solo_share"] = (pretty["solo_share"] * 100).round(0).astype(int).astype(str) + "%"
    lines.append(_md(pretty, floatfmt=".0f"))

    if your_rank is not None:
        you_row_data = leaderboard.iloc[your_rank - 1]
        median_peer_pace = leaderboard[~leaderboard["owner"].str.contains(r"\(you\)", regex=True)]["posts_per_day"].median()
        bits = [f"You rank **#{your_rank}** of {len(leaderboard)} on mean favs ({your_mean_favs:.0f})"]
        if your_rank > 1:
            bits.append(f"ahead: `{leaderboard.iloc[your_rank - 2]['owner']}` ({leaderboard.iloc[your_rank - 2]['mean_favs']:.0f})")
        if your_rank < len(leaderboard):
            bits.append(f"behind: `{leaderboard.iloc[your_rank]['owner']}` ({leaderboard.iloc[your_rank]['mean_favs']:.0f})")
        lines.append("- " + ". ".join(bits) + ".")
        lines.append(
            f"- Your pace: **{you_row_data['posts_per_day']:.2f} posts/day** vs peer median **{median_peer_pace:.2f}/day** "
            f"(over their 415-post window). The leaderboard rewards fav-rate but the audience hears from peers far more often.\n"
        )

    lines.append("## Tag benchmark — your mean favs vs site mean favs per tag\n")
    lines.append(
        f"_For each of your top {bench_tag_count} tags by lift, sampled the latest 320 site-wide posts using that tag. "
        "Positive `delta_favs` = your posts with that tag fav above the typical site post with that tag (you've found a niche where your execution wins). "
        "Negative = the tag works site-wide but your specific posts under-deliver on it._\n"
    )
    if tag_bench.empty:
        lines.append("_(no benchmarked tags overlapped with your tag stats)_\n")
    else:
        bench_pretty = tag_bench.copy()
        bench_pretty["your_vs_site_pct"] = bench_pretty["your_vs_site_pct"].round(0).astype(int).astype(str) + "%"
        lines.append(_md(bench_pretty, floatfmt=".0f"))

    lines.append("## Peer overlap — % of each peer's posts using your top-lift tags\n")
    lines.append(
        "_How much each top uploader uses the same tags that drive your engagement. "
        "High % = direct competitor for that niche; low % = you're alone in that lane._\n"
    )
    if overlap.empty:
        lines.append("_(no overlap data)_\n")
    else:
        ov_pretty = overlap.copy()
        for col in ov_pretty.columns:
            if col != "tag":
                ov_pretty[col] = ov_pretty[col].round(0).astype(int).astype(str) + "%"
        lines.append(_md(ov_pretty))

    lines.append("## How to read this\n")
    lines.append(
        "- **Per-post fav rate ≠ total reach.** `mean_favs` rewards efficiency; `posts_per_day × mean_favs` approximates total favs/day delivered to your audience. The latter is the number the leaderboard hides.\n"
        "- **'Uncontested niche' usually means small sample.** A tag at 0% peer overlap with your n=3 isn't a moat; it's three posts. Check `your_n` before claiming a niche.\n"
        "- **Negative tag delta with high `your_n` is a real problem.** The tag works site-wide; yours specifically don't. Composition or execution issue.\n"
        "- **Watch for selection bias.** Your sample = your full catalog. Peers' samples = their most recent posts (often their best, since they've improved over time). Treat any peer who beats you as beating you with their *recent* work, not their average.\n"
    )

    path = out_dir / "peers.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _compute_issues(
    user_tag: str,
    leaderboard: pd.DataFrame,
    tag_bench: pd.DataFrame,
    overlap: pd.DataFrame,
) -> list[str]:
    issues: list[str] = []
    you = leaderboard[leaderboard["owner"].str.contains(r"\(you\)", regex=True)]
    if you.empty:
        return issues
    you = you.iloc[0]
    peers = leaderboard[~leaderboard["owner"].str.contains(r"\(you\)", regex=True)]
    if peers.empty:
        return issues

    # 1. Volume disadvantage
    median_pace = float(peers["posts_per_day"].median())
    your_pace = float(you["posts_per_day"])
    if median_pace > 0 and your_pace < median_pace * 0.5:
        ratio = median_pace / your_pace if your_pace > 0 else float("inf")
        issues.append(
            f"**Volume disadvantage:** you publish {your_pace:.2f} posts/day; peer median is {median_pace:.2f}/day "
            f"(**{ratio:.1f}× yours**). Even if your fav-rate were higher, peers reach the audience far more often. "
            "Per-post quality cannot compensate for being absent from the feed."
        )

    # 2. Rating share gap
    your_explicit = float(you["explicit_share"]) * 100
    peer_med_explicit = float(peers["explicit_share"].median()) * 100
    if peer_med_explicit - your_explicit > 15:
        issues.append(
            f"**Under-shipping the highest-fav rating:** your explicit share is {your_explicit:.0f}% vs peer median {peer_med_explicit:.0f}%. "
            "Explicit posts on this site mean ~2× the favs of safe (per your earlier insight). Either commit to the rating that wins or accept the cap."
        )

    # 3. Solo over-share
    your_solo = float(you["solo_share"]) * 100
    peer_med_solo = float(peers["solo_share"].median()) * 100
    if your_solo - peer_med_solo > 10:
        issues.append(
            f"**Over-shipping solo:** {your_solo:.0f}% of your work is solo vs peer median {peer_med_solo:.0f}%. "
            "Cast size correlates with favs site-wide; you're skewing toward the lower-engagement framing."
        )

    # 4. Negative tag deltas (your version of a tag underperforms site)
    if not tag_bench.empty:
        bad = tag_bench[(tag_bench["delta_favs"] < -20) & (tag_bench["your_n"] >= 5)]
        if not bad.empty:
            tags = ", ".join(f"`{r['tag']}` ({r['delta_favs']:+.0f})" for _, r in bad.head(5).iterrows())
            issues.append(
                f"**Tags where you under-deliver vs site:** {tags}. You keep using these tags but your posts using them "
                "trail the typical site post — composition or quality issue, not a tag-selection issue."
            )

    # 5. Tag-bench winners with small n (statistical fragility)
    if not tag_bench.empty:
        fragile = tag_bench[(tag_bench["delta_favs"] > 50) & (tag_bench["your_n"] < 5)]
        if len(fragile) >= 3:
            issues.append(
                f"**'Niches' propped up by tiny samples:** {len(fragile)} of your top tag wins have n<5 posts. "
                "These are not niches — they're a handful of lucky posts. Re-examine after you have n≥10 in each."
            )

    # 6. Owned-niches alarm: many of your top tags have ~0% peer overlap (could be moat OR could be no audience)
    if not overlap.empty and len(overlap) >= 5:
        ncols = [c for c in overlap.columns if c != "tag"]
        if ncols:
            row_max = overlap[ncols].max(axis=1)
            zero_competition = int((row_max < 2).sum())
            if zero_competition >= len(overlap) * 0.5:
                issues.append(
                    f"**Most of your top-lift tags have <2% peer usage** ({zero_competition}/{len(overlap)}). "
                    "This is either a moat *or* a sign you're optimizing for tags nobody else bothers with because the audience is small. "
                    "Cross-check: do these tags have high site-wide fav means in the benchmark, or just high lift on your few posts?"
                )

    # 7. You sit mid-pack but your sample is unusually long
    your_span = int(you["span_days"])
    peer_med_span = float(peers["span_days"].median())
    if peer_med_span > 0 and your_span > peer_med_span * 3:
        issues.append(
            f"**Sample-window asymmetry:** your 415 posts span {your_span} days; peer median {peer_med_span:.0f} days. "
            "Your average is dragged down by older work; peers are being judged on their recent (likely improved) output. "
            "Re-run with `--peer-cap` set lower, or compare against your own last-N posts to remove the asymmetry."
        )

    return issues

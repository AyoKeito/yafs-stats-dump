"""Auto-generate strategic insights from posts dataframe.

Each insight is a small function that returns either a Markdown bullet string
or None (if not enough data / not interesting). The renderer concatenates the
non-None bullets into a `## Strategic insights` block.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

# fav_count slice considered "elite" for hit/miss analyses
ELITE_PCTILE = 0.10
WEAK_PCTILE = 0.10
MIN_GROUP_N = 5


def _has_tag(series: pd.Series, tag: str) -> pd.Series:
    return series.fillna("").str.split().apply(lambda xs, t=tag: t in xs)


def insight_score_fav_correlation(posts_df: pd.DataFrame) -> str | None:
    if len(posts_df) < 20:
        return None
    corr = posts_df["score_total"].corr(posts_df["fav_count"])
    if pd.isna(corr):
        return None
    if abs(corr) >= 0.9:
        return (
            f"Score and favs correlate r={corr:.2f}. Tracking both is redundant — pick one and stop double-counting."
        )
    return f"Score vs fav_count correlation: r={corr:.2f}."


def insight_distribution_top_heavy(posts_df: pd.DataFrame) -> str | None:
    if len(posts_df) < 30:
        return None
    favs = posts_df["fav_count"].sort_values(ascending=False)
    n_top = max(int(len(favs) * 0.1), 5)
    top_share = favs.head(n_top).sum() / favs.sum() * 100
    median = favs.median()
    mean = favs.mean()
    skew_pct = (mean - median) / mean * 100
    note = ""
    if top_share > 25:
        note = " Your distribution is **top-heavy** — averages mask that most of your posts perform meaningfully worse than the headline number."
    elif top_share < 18:
        note = " Distribution is unusually flat — your floor and ceiling are close, which is a stability signal."
    return (
        f"**Distribution shape:** top 10% of posts hold {top_share:.0f}% of total favs; "
        f"median {median:.0f} sits {skew_pct:.0f}% below mean {mean:.0f}.{note}"
    )


def insight_failure_rate(posts_df: pd.DataFrame) -> str | None:
    if len(posts_df) < 30:
        return None
    n = len(posts_df)
    p25 = float(posts_df["fav_count"].quantile(0.25))
    p10 = float(posts_df["fav_count"].quantile(0.10))
    n_under_p25 = int((posts_df["fav_count"] < p25).sum())
    bottom_share_of_favs = posts_df["fav_count"].nsmallest(n_under_p25).sum() / posts_df["fav_count"].sum() * 100
    return (
        f"**Failure floor:** bottom 25% of posts (≤{p25:.0f} favs) = {bottom_share_of_favs:.0f}% of total favs delivered. "
        f"Bottom 10% (≤{p10:.0f} favs) is essentially wasted ship-effort. "
        "If you can identify these before posting, skipping them raises your average without producing anything new."
    )


def insight_posting_pace(posts_df: pd.DataFrame) -> str | None:
    if posts_df["created_at"].isna().all():
        return None
    by_year = posts_df.assign(year=posts_df["created_at"].dt.year).groupby("year").size()
    if len(by_year) < 2:
        return None
    # Approximate months per year present in data
    pace = {}
    for y, n in by_year.items():
        ys = posts_df[posts_df["created_at"].dt.year == y]["created_at"]
        span_days = (ys.max() - ys.min()).days
        months = max(span_days / 30.44, 1.0)
        pace[int(y)] = n / months
    pieces = ", ".join(f"{y}={p:.1f}/mo" for y, p in pace.items())
    years = sorted(pace.keys())
    first, last = pace[years[0]], pace[years[-1]]
    if first == 0:
        return None
    delta_pct = (last - first) / first * 100
    note = ""
    if delta_pct < -15:
        note = (
            f" **Posting pace dropped {abs(delta_pct):.0f}%** from {years[0]} to {years[-1]}. "
            "Per-post engagement may be up, but if total ship volume is down, your audience hears from you less. "
            "Per-post lift can't compensate for fewer posts indefinitely."
        )
    elif delta_pct > 15:
        note = f" Pace up {delta_pct:.0f}%."
    return f"**Posting pace:** {pieces}.{note}"


def insight_recent_decay(posts_df: pd.DataFrame) -> str | None:
    if posts_df["created_at"].isna().all():
        return None
    now = pd.Timestamp.now(tz="UTC")
    last30 = posts_df[posts_df["created_at"] > now - pd.Timedelta(days=30)]
    last90 = posts_df[posts_df["created_at"] > now - pd.Timedelta(days=90)]
    lifetime_mean = posts_df["fav_count"].mean()
    if len(last30) < 5:
        return None
    m30 = last30["fav_count"].mean()
    m90 = last90["fav_count"].mean() if len(last90) >= 5 else None
    bits = [f"last 30d n={len(last30)} mean={m30:.0f}"]
    if m90 is not None:
        bits.append(f"last 90d n={len(last90)} mean={m90:.0f}")
    bits.append(f"lifetime mean={lifetime_mean:.0f}")
    summary = ", ".join(bits)
    if m30 < lifetime_mean * 0.92:
        diff = (lifetime_mean - m30) / lifetime_mean * 100
        return (
            f"**Recent decay:** {summary}. Last 30d is **{diff:.0f}% below lifetime** — "
            "recent work is dragging the average down, not pushing it forward. Investigate before shipping more in the same vein."
        )
    if m30 > lifetime_mean * 1.08:
        diff = (m30 - lifetime_mean) / lifetime_mean * 100
        return f"**Recent momentum:** {summary}. Last 30d is +{diff:.0f}% above lifetime — keep doing what you've been doing."
    return f"**Recent vs lifetime:** {summary}. Stable."


def insight_character_dependency(posts_df: pd.DataFrame) -> str | None:
    if "tags_character" not in posts_df.columns or len(posts_df) < 30:
        return None
    char_lists = posts_df["tags_character"].fillna("").str.split()
    flat = [t for sub in char_lists for t in sub]
    if not flat:
        return None
    counts = pd.Series(flat).value_counts()
    top_char = counts.index[0]
    top_n = int(counts.iloc[0])
    if top_n < 10:
        return None
    has = char_lists.apply(lambda xs, t=top_char: t in xs)
    with_mean = posts_df.loc[has, "fav_count"].mean()
    without_mean = posts_df.loc[~has, "fav_count"].mean()
    overall = posts_df["fav_count"].mean()
    if without_mean <= 0:
        return None
    drop_pct = (overall - without_mean) / overall * 100
    if drop_pct < 5:
        return None
    return (
        f"**Single-character dependency:** `{top_char}` appears on {top_n}/{len(posts_df)} posts "
        f"({top_n/len(posts_df)*100:.0f}% of output) and means {with_mean:.0f} favs vs {without_mean:.0f} without. "
        f"Removing this one character drops your overall mean by **{drop_pct:.0f}%** — your headline number "
        "is propped up by one franchise. That's concentration risk, not strength."
    )


def insight_small_sample_winners(tag_stats: pd.DataFrame) -> str | None:
    if tag_stats.empty:
        return None
    winners = tag_stats[(tag_stats["lift_favs"] > 50) & (tag_stats["category"].isin(["general", "species"]))]
    if winners.empty:
        return None
    fragile = winners[winners["n_posts"] < 10]
    solid = winners[winners["n_posts"] >= 20]
    return (
        f"**Statistical honesty:** of your high-lift tags (>+50 favs), **{len(fragile)} have n<10 posts** "
        f"and only {len(solid)} have n≥20. With your fav std around the high end, n<10 lift values can swing "
        "±50 favs from a single outlier. Treat the small-sample 'wins' as hypotheses, not proven niches — "
        "the n=3 row at the top of the lift table is likely one viral post, not a discovered formula."
    )


def insight_rating_lift(posts_df: pd.DataFrame) -> str | None:
    if "rating" not in posts_df.columns or posts_df["rating"].nunique() < 2:
        return None
    g = posts_df.groupby("rating")["fav_count"].agg(["count", "mean"])
    if g["count"].min() < MIN_GROUP_N:
        g = g[g["count"] >= MIN_GROUP_N]
    if g.empty or len(g) < 2:
        return None
    best = g["mean"].idxmax()
    worst = g["mean"].idxmin()
    if best == worst:
        return None
    ratio = g.loc[best, "mean"] / max(g.loc[worst, "mean"], 1e-9)
    breakdown = ", ".join(f"`{r}`={int(v)}" for r, v in g["mean"].items())
    return (
        f"**Rating effect:** mean favs by rating — {breakdown}. "
        f"`{best}` is **{ratio:.1f}×** `{worst}`."
    )


def insight_year_trend(posts_df: pd.DataFrame) -> str | None:
    if posts_df["created_at"].isna().all():
        return None
    yearly = posts_df.assign(year=posts_df["created_at"].dt.year).groupby("year")["fav_count"].agg(["count", "mean"])
    yearly = yearly[yearly["count"] >= MIN_GROUP_N]
    if len(yearly) < 2:
        return None
    first_year, last_year = yearly.index.min(), yearly.index.max()
    first_mean = yearly.loc[first_year, "mean"]
    last_mean = yearly.loc[last_year, "mean"]
    if first_mean <= 0:
        return None
    pct = (last_mean - first_mean) / first_mean * 100
    direction = "growing" if pct > 5 else ("declining" if pct < -5 else "flat")
    pieces = ", ".join(f"{int(y)}={int(v)}" for y, v in yearly["mean"].items())
    note = ""
    if direction == "growing":
        note = " Newer posts fav higher → age confound favors recent posts; lift rankings are trustworthy (not inflated by accumulation)."
    elif direction == "declining":
        note = " Newer posts fav lower → some lift in older tags may be exposure-time, not the tag itself."
    return f"**Year trend (mean favs):** {pieces}. Audience is **{direction}** ({pct:+.0f}% {first_year}→{last_year}).{note}"


def insight_cast_size(posts_df: pd.DataFrame, tags_df: pd.DataFrame) -> str | None:
    if posts_df.empty:
        return None
    rows = []
    for tag in ("solo", "duo", "group"):
        mask = _has_tag(posts_df["tags_general"], tag)
        n = int(mask.sum())
        if n >= MIN_GROUP_N:
            rows.append((tag, n, posts_df.loc[mask, "fav_count"].mean()))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[2], reverse=True)
    best = rows[0]
    worst = rows[-1]
    if worst[2] <= 0:
        return None
    pct_solo = None
    solo_row = next((r for r in rows if r[0] == "solo"), None)
    if solo_row:
        pct_solo = solo_row[1] / len(posts_df) * 100
    summary = ", ".join(f"{t}={int(m)} (n={n})" for t, n, m in rows)
    extra = ""
    if pct_solo is not None and pct_solo > 60 and solo_row[2] < best[2]:
        gap = best[2] - solo_row[2]
        cost = gap * solo_row[1]
        extra = (
            f" You spent **{pct_solo:.0f}% of your output** on the framing that fav-rates worst — "
            f"approximately {cost:,.0f} favs left on the table vs. if those solo posts had been `{best[0]}`."
        )
    return f"**Cast size (mean favs):** {summary}.{extra}"


def insight_pairing(posts_df: pd.DataFrame) -> str | None:
    candidates = ("male/female", "female/female", "male/male")
    rows = []
    for tag in candidates:
        mask = _has_tag(posts_df["tags_general"], tag)
        n = int(mask.sum())
        if n >= MIN_GROUP_N:
            rows.append((tag, n, posts_df.loc[mask, "fav_count"].mean()))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[2], reverse=True)
    best = rows[0]
    summary = ", ".join(f"`{t}`={int(m)} (n={n})" for t, n, m in rows)
    return f"**Pairing types (mean favs):** {summary}. Strongest: **{best[0]}**."


def insight_animation(posts_df: pd.DataFrame) -> str | None:
    if "tags_meta" not in posts_df.columns:
        return None
    mask = _has_tag(posts_df["tags_meta"], "animated") | _has_tag(posts_df["tags_meta"], "webm")
    n = int(mask.sum())
    if n < MIN_GROUP_N:
        return None
    mean_anim = posts_df.loc[mask, "fav_count"].mean()
    mean_corpus = posts_df["fav_count"].mean()
    delta = mean_anim - mean_corpus
    sign = "above" if delta > 0 else "below"
    return (
        f"**Animation ROI:** {n} animated/webm posts at {mean_anim:.0f} mean favs "
        f"({delta:+.0f} {sign} corpus mean of {mean_corpus:.0f})."
    )


def insight_tag_richness(posts_df: pd.DataFrame) -> str | None:
    if posts_df.empty:
        return None
    counts = posts_df["tags_general"].fillna("").str.split().apply(len)
    if counts.sum() == 0:
        return None
    corr = counts.corr(posts_df["fav_count"])
    if pd.isna(corr):
        return None
    n_top = max(int(len(posts_df) * ELITE_PCTILE), 5)
    top_idx = posts_df["fav_count"].nlargest(n_top).index
    bot_idx = posts_df["fav_count"].nsmallest(n_top).index
    top_tags = counts.loc[top_idx].mean()
    bot_tags = counts.loc[bot_idx].mean()
    return (
        f"**Tag richness:** corr(general_tag_count, favs) = {corr:.2f}. "
        f"Top {int(ELITE_PCTILE*100)}% posts avg {top_tags:.0f} general tags, "
        f"bottom {int(WEAK_PCTILE*100)}% avg {bot_tags:.0f}. "
        "Sparse scenes are a negative indicator; don't pad artificially."
    )


def insight_top_species(tag_stats: pd.DataFrame) -> str | None:
    if tag_stats.empty or "category" not in tag_stats.columns:
        return None
    sp = tag_stats[tag_stats["category"] == "species"].copy()
    if sp.empty:
        return None
    sp_top = sp.sort_values("lift_favs", ascending=False).head(5)
    sp_bot = sp.sort_values("lift_favs", ascending=True).head(5)
    top_str = ", ".join(f"`{t}` (+{l:.0f}, n={int(n)})"
                        for t, l, n in zip(sp_top["tag"], sp_top["lift_favs"], sp_top["n_posts"])
                        if l > 0)
    bot_str = ", ".join(f"`{t}` ({l:.0f}, n={int(n)})"
                        for t, l, n in zip(sp_bot["tag"], sp_bot["lift_favs"], sp_bot["n_posts"])
                        if l < 0)
    if not top_str and not bot_str:
        return None
    parts = []
    if top_str:
        parts.append(f"top species: {top_str}")
    if bot_str:
        parts.append(f"bottom species: {bot_str}")
    return "**Species leverage:** " + " — ".join(parts) + "."


def insight_oc_underperformance(tag_stats: pd.DataFrame, artist_tag: str) -> str | None:
    if tag_stats.empty or "category" not in tag_stats.columns:
        return None
    chars = tag_stats[tag_stats["category"] == "character"]
    suffix = f"_({artist_tag.lower()})"
    ocs = chars[chars["tag"].str.lower().str.endswith(suffix)]
    if ocs.empty:
        return None
    rows = []
    total_n = 0
    total_loss = 0.0
    for _, r in ocs.iterrows():
        rows.append(f"`{r['tag']}` ({r['lift_favs']:+.0f} favs, n={int(r['n_posts'])})")
        total_n += int(r["n_posts"])
        total_loss += float(r["lift_favs"]) * int(r["n_posts"])
    note = ""
    if total_loss < -200:
        note = (
            f" — across {total_n} OC posts, that's **{abs(total_loss):,.0f} favs you didn't earn** "
            "vs. if those slots had been a corpus-average post. OCs are costing you, not building you."
        )
    return f"**OCs underperform:** {', '.join(rows)}.{note}"


def insight_watermark_confound(posts_df: pd.DataFrame, tag_stats: pd.DataFrame) -> str | None:
    if tag_stats.empty:
        return None
    wm = tag_stats[(tag_stats["category"] == "meta") & (tag_stats["tag"] == "watermark")]
    if wm.empty:
        return None
    lift = float(wm.iloc[0]["lift_favs"])
    n = int(wm.iloc[0]["n_posts"])
    if lift > 5:
        return (
            f"**Confound to ignore:** `watermark` shows +{lift:.0f} lift (n={n}) — likely because you only "
            "watermark your more polished pieces, not because the watermark draws favs."
        )
    return None


INSIGHT_FNS: list[Callable] = [
    insight_score_fav_correlation,
    insight_distribution_top_heavy,
    insight_failure_rate,
    insight_rating_lift,
    insight_posting_pace,
    insight_recent_decay,
    insight_year_trend,
    insight_cast_size,
    insight_pairing,
    insight_animation,
    insight_character_dependency,
    insight_top_species,
    insight_oc_underperformance,
    insight_tag_richness,
    insight_small_sample_winners,
    insight_watermark_confound,
]


def build_insights_section(
    posts_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    tag_stats: pd.DataFrame,
    artist_tag: str,
) -> str:
    """Run every insight function with the args it needs; collect non-None bullets."""
    if posts_df.empty:
        return "## Strategic insights\n\n_(no data)_\n"

    bullets: list[str] = []
    # dispatch by inspecting expected args (keep it dumb-simple: pass what each one wants)
    def call(fn):
        name = fn.__name__
        if "tag_stats" in fn.__code__.co_varnames and "artist_tag" in fn.__code__.co_varnames:
            return fn(tag_stats, artist_tag)
        if "tag_stats" in fn.__code__.co_varnames and "posts_df" in fn.__code__.co_varnames:
            return fn(posts_df, tag_stats)
        if "tags_df" in fn.__code__.co_varnames and "posts_df" in fn.__code__.co_varnames:
            return fn(posts_df, tags_df)
        if "tag_stats" in fn.__code__.co_varnames:
            return fn(tag_stats)
        return fn(posts_df)

    for fn in INSIGHT_FNS:
        try:
            out = call(fn)
        except Exception as e:  # noqa: BLE001
            out = f"_(insight `{fn.__name__}` failed: {e})_"
        if out:
            bullets.append(f"- {out}")

    body = "\n".join(bullets) if bullets else "_(no insights triggered)_"
    return (
        "## Strategic insights\n\n"
        "_Auto-generated from this run's data. Bias is toward critical findings — when something is genuinely positive it's stated, but the goal is to surface problems, not flatter the data._\n\n"
        f"{body}\n"
    )

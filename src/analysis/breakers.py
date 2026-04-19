"""Pattern-breakers analyzer.

Two questions:

1. **Hits despite losers** — which posts succeeded even though they carry
   tags that normally underperform? These are the most interesting outliers:
   they tell you what *else* you can do besides the proven recipe.

2. **Misses despite winners** — which posts underperformed despite carrying
   tags from the proven-winner set? These are diagnostic: same recipe,
   why didn't it land?

For each post we score:
- `expected_favs` = corpus mean + sum of tag lifts the post carries
  (general + species categories only — character/meta are too noisy/confounded)
- `surprise` = actual_favs - expected_favs

Then we surface the most positively-surprising posts among the bottom-tagged
half, and the most negatively-surprising among the top-tagged half.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tabulate import tabulate

from .correlations import per_tag_stats
from .load import load_posts

POST_URL = "https://e6ai.net/posts/{id}"
ANALYSIS_CATS = ("general", "species")
TOP_N = 15


def _tag_lift_lookup(tag_stats: pd.DataFrame) -> dict[tuple[str, str], float]:
    if tag_stats.empty:
        return {}
    sub = tag_stats[tag_stats["category"].isin(ANALYSIS_CATS)]
    return {(r["category"], r["tag"]): float(r["lift_favs"]) for _, r in sub.iterrows()}


def _post_tag_pairs(row: pd.Series) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for cat in ANALYSIS_CATS:
        col = f"tags_{cat}"
        raw = row.get(col)
        if isinstance(raw, str) and raw:
            for t in raw.split():
                pairs.append((cat, t))
    return pairs


def compute_surprise(posts_df: pd.DataFrame, tag_stats: pd.DataFrame) -> pd.DataFrame:
    """Return posts_df + expected_favs, surprise, n_winner_tags, n_loser_tags, mean_tag_lift."""
    if posts_df.empty:
        return posts_df

    lifts = _tag_lift_lookup(tag_stats)
    corpus_mean = posts_df["fav_count"].mean()

    expected = []
    n_winners = []
    n_losers = []
    mean_lifts = []
    for _, row in posts_df.iterrows():
        pairs = _post_tag_pairs(row)
        per_tag = [lifts.get(p) for p in pairs]
        per_tag = [v for v in per_tag if v is not None]
        if not per_tag:
            expected.append(corpus_mean)
            n_winners.append(0)
            n_losers.append(0)
            mean_lifts.append(0.0)
            continue
        # average lift, not sum — sum overcounts because lifts aren't independent
        avg_lift = sum(per_tag) / len(per_tag)
        expected.append(corpus_mean + avg_lift)
        n_winners.append(sum(1 for v in per_tag if v > 20))
        n_losers.append(sum(1 for v in per_tag if v < -20))
        mean_lifts.append(avg_lift)

    out = posts_df.copy()
    out["expected_favs"] = expected
    out["surprise"] = out["fav_count"] - out["expected_favs"]
    out["n_winner_tags"] = n_winners
    out["n_loser_tags"] = n_losers
    out["mean_tag_lift"] = mean_lifts
    return out


def hits_despite_losers(scored: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Posts whose tag mix predicts a low score, but actually did well."""
    if scored.empty:
        return scored
    candidates = scored[scored["mean_tag_lift"] < 0].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values("surprise", ascending=False).head(top_n)
    return candidates


def misses_despite_winners(scored: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Posts whose tag mix predicts a high score, but actually flopped."""
    if scored.empty:
        return scored
    candidates = scored[scored["mean_tag_lift"] > 0].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values("surprise", ascending=True).head(top_n)
    return candidates


def _exemplar_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    show = df.copy()
    show["url"] = show["id"].map(lambda i: POST_URL.format(id=i))
    show = show[[
        "id", "url", "rating", "fav_count", "expected_favs", "surprise",
        "n_winner_tags", "n_loser_tags", "tags_species", "tags_general",
    ]]
    return tabulate(show, headers="keys", tablefmt="github", showindex=False, floatfmt=".0f") + "\n"


def render_breakers(tag: str, cache_path: Path, out_dir: Path) -> Path:
    posts_df, tags_df = load_posts(cache_path)
    if posts_df.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "breakers.md"
        path.write_text(f"# Pattern breakers — `{tag}`\n\nNo posts.\n", encoding="utf-8")
        return path

    tag_stats = per_tag_stats(posts_df, tags_df)
    scored = compute_surprise(posts_df, tag_stats)

    hits = hits_despite_losers(scored)
    misses = misses_despite_winners(scored)

    out_dir.mkdir(parents=True, exist_ok=True)
    scored_csv = out_dir / "scored.csv"
    scored.sort_values("surprise", ascending=False).to_csv(scored_csv, index=False, encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Pattern breakers — `{tag}`\n")
    lines.append(
        "_Each post's `expected_favs` = corpus mean + average lift of its general+species tags. "
        "`surprise` = actual − expected. Positive surprise = punched above its tag mix; negative = flopped despite a winning tag mix._\n"
    )
    lines.append(f"- Corpus mean fav_count: **{posts_df['fav_count'].mean():.0f}** (n={len(posts_df)})")
    lines.append(f"- Posts with negative-lift tag mix: **{int((scored['mean_tag_lift'] < 0).sum())}**")
    lines.append(f"- Posts with positive-lift tag mix: **{int((scored['mean_tag_lift'] > 0).sum())}**\n")

    lines.append("## Hits despite losers — what *else* works for you\n")
    lines.append(
        "These posts carry tags that on average underperform, yet they overperformed anyway. "
        "Look for the unmodeled signal: a strong composition, a fresh subject, an unusual color story, a hook in the description. "
        "Patterns you spot here are candidates for new bets that aren't covered by the obvious 'recipe'.\n"
    )
    lines.append(_exemplar_table(hits))

    lines.append("## Misses despite winners — diagnostic for execution\n")
    lines.append(
        "These posts carry your proven winning tags but flopped. Common culprits: bad upload timing, weak focal subject, "
        "muddy lighting, posted to the wrong audience, or a 'good ingredients, bad recipe' problem. "
        "Compare them to the top-10 in the main report — what's *missing* from these that the hits have?\n"
    )
    lines.append(_exemplar_table(misses))

    lines.append("## How to use this\n")
    lines.append(
        "- A *hit despite losers* with a unique sub-theme = candidate for a deliberate series.\n"
        "- A *miss despite winners* that looks technically fine = treat as a posting/timing/discoverability issue, not an art issue.\n"
        "- A *miss despite winners* that looks technically off = your usual quality bar matters more than tag selection.\n"
        f"- Full ranking saved to `{scored_csv.name}` if you want to slice further.\n"
    )

    path = out_dir / "breakers.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

"""Render Markdown report + PNG charts from analysis outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from .correlations import (
    co_occurrence_matrix,
    per_tag_stats,
    tag_pairs_stats,
    top_bottom_posts,
)
from .load import TAG_CATEGORIES, load_posts

POST_URL = "https://e6ai.net/posts/{id}"


def _md_table(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    if df.empty:
        return "_(no data)_\n"
    return tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=floatfmt) + "\n"


def _save_hist(series: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(series.dropna(), bins=30, color="steelblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("posts")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_scatter(x: pd.Series, y: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.5, s=20)
    ax.set_title(title)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix.values, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_top_tag_bar(tag_stats: pd.DataFrame, category: str, path: Path) -> None:
    sub = tag_stats[tag_stats["category"] == category].head(20)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(sub["tag"][::-1], sub["lift_favs"][::-1], color="darkorange")
    ax.set_title(f"Top {category} tags by fav_count lift (n>={3})")
    ax.set_xlabel("mean fav_count minus corpus mean")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render(tag: str, cache_path: Path, out_dir: Path) -> Path:
    posts_df, tags_df = load_posts(cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "posts.csv"
    posts_df.to_csv(csv_path, index=False, encoding="utf-8")

    tag_stats = per_tag_stats(posts_df, tags_df)
    pairs = tag_pairs_stats(posts_df, tags_df)
    cooc = co_occurrence_matrix(tags_df, "general", top_n=20)
    top_posts, bottom_posts = top_bottom_posts(posts_df, by="fav_count", n=10)

    if not posts_df.empty:
        _save_hist(posts_df["fav_count"].rename("fav_count"), "fav_count distribution", charts_dir / "hist_favs.png")
        _save_hist(posts_df["score_total"].rename("score_total"), "score_total distribution", charts_dir / "hist_score.png")
        _save_scatter(
            posts_df["score_total"].rename("score_total"),
            posts_df["fav_count"].rename("fav_count"),
            "score vs fav_count",
            charts_dir / "scatter_score_vs_favs.png",
        )
        if posts_df["created_at"].notna().any():
            age_days = (pd.Timestamp.now(tz="UTC") - posts_df["created_at"]).dt.total_seconds() / 86400
            _save_scatter(
                age_days.rename("age_days"),
                posts_df["fav_count"].rename("fav_count"),
                "fav_count vs post age (days)",
                charts_dir / "scatter_age_vs_favs.png",
            )
    _save_heatmap(cooc, "Top 20 general tags — co-occurrence", charts_dir / "cooccurrence_general.png")
    if not tag_stats.empty:
        for cat in ("general", "species", "character", "meta"):
            _save_top_tag_bar(tag_stats, cat, charts_dir / f"top_tags_{cat}.png")
        cooc.to_csv(out_dir / "cooccurrence_general.csv")

    md = _build_markdown(
        tag=tag,
        posts_df=posts_df,
        tag_stats=tag_stats,
        pairs=pairs,
        top_posts=top_posts,
        bottom_posts=bottom_posts,
    )
    report_path = out_dir / "report.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path


def _exemplar_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[["id", "score_total", "fav_count", "rating", "tags_general", "tags_species"]].copy()
    out["url"] = out["id"].map(lambda i: POST_URL.format(id=i))
    return out[["id", "url", "score_total", "fav_count", "rating", "tags_species", "tags_general"]]


def _build_markdown(
    tag: str,
    posts_df: pd.DataFrame,
    tag_stats: pd.DataFrame,
    pairs: pd.DataFrame,
    top_posts: pd.DataFrame,
    bottom_posts: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# e6ai engagement report — `{tag}`\n")

    if posts_df.empty:
        lines.append("No posts found.\n")
        return "\n".join(lines)

    date_min = posts_df["created_at"].min()
    date_max = posts_df["created_at"].max()
    lines.append("## Summary\n")
    summary = pd.DataFrame([{
        "n_posts": len(posts_df),
        "date_min": date_min.date() if pd.notna(date_min) else "",
        "date_max": date_max.date() if pd.notna(date_max) else "",
        "mean_score": posts_df["score_total"].mean(),
        "median_score": posts_df["score_total"].median(),
        "mean_favs": posts_df["fav_count"].mean(),
        "median_favs": posts_df["fav_count"].median(),
    }])
    lines.append(_md_table(summary, floatfmt=".2f"))

    lines.append("## Per-category tag engagement\n")
    lines.append(f"_Tags with n_posts ≥ 3. `lift_favs` = mean fav_count with tag − corpus mean ({posts_df['fav_count'].mean():.1f})._\n")
    for cat in TAG_CATEGORIES:
        sub = tag_stats[tag_stats["category"] == cat].drop(columns=["category"]) if not tag_stats.empty else tag_stats
        if sub.empty:
            continue
        lines.append(f"### {cat} (top 25 by fav lift)\n")
        lines.append(_md_table(sub.head(25)))
        lines.append(f"### {cat} (bottom 10 by fav lift)\n")
        lines.append(_md_table(sub.tail(10).iloc[::-1]))

    lines.append("## Tag pair winners (general + species)\n")
    lines.append(_md_table(pairs))

    lines.append("## Top 10 posts by fav_count\n")
    lines.append(_md_table(_exemplar_rows(top_posts), floatfmt=".0f"))
    lines.append("## Bottom 10 posts by fav_count\n")
    lines.append(_md_table(_exemplar_rows(bottom_posts), floatfmt=".0f"))

    lines.append("## Charts\n")
    for name, caption in [
        ("hist_favs.png", "fav_count distribution"),
        ("hist_score.png", "score distribution"),
        ("scatter_score_vs_favs.png", "score vs fav_count"),
        ("scatter_age_vs_favs.png", "fav_count vs post age (days)"),
        ("cooccurrence_general.png", "top general tags co-occurrence"),
        ("top_tags_general.png", "top general tags by fav lift"),
        ("top_tags_species.png", "top species tags by fav lift"),
        ("top_tags_character.png", "top character tags by fav lift"),
        ("top_tags_meta.png", "top meta tags by fav lift"),
    ]:
        lines.append(f"![{caption}](charts/{name})\n")

    lines.append("## Caveats\n")
    lines.append(
        "- **Exposure confound:** older posts have had more time to accumulate favs. "
        "Compare the age-vs-favs scatter before attributing a tag's lift to the tag itself.\n"
        "- **Small samples:** tags with n_posts < 3 are filtered out, but n=3–5 still carries high variance.\n"
        "- **Correlation ≠ causation:** a tag that co-occurs with your best work may be a marker, not a driver. "
        "Look at which tags move together in the co-occurrence heatmap.\n"
        "- **Character tags** will often top the lift tables because a popular character pulls favs independent "
        "of your execution. Focus on the `general` and `species` tables for style/content levers you actually control.\n"
    )
    return "\n".join(lines)

"""Tag correlation analysis — what drives score and fav_count."""
from __future__ import annotations

from itertools import combinations

import pandas as pd

MIN_N = 3  # ignore tags present on fewer than this many posts


def per_tag_stats(posts_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (category, tag) with engagement stats and lift vs corpus mean."""
    if tags_df.empty or posts_df.empty:
        return pd.DataFrame()

    corpus_mean_score = posts_df["score_total"].mean()
    corpus_mean_favs = posts_df["fav_count"].mean()

    grouped = tags_df.groupby(["category", "tag"]).agg(
        n_posts=("post_id", "nunique"),
        mean_score=("score_total", "mean"),
        median_score=("score_total", "median"),
        mean_favs=("fav_count", "mean"),
        median_favs=("fav_count", "median"),
    ).reset_index()

    grouped["lift_score"] = grouped["mean_score"] - corpus_mean_score
    grouped["lift_favs"] = grouped["mean_favs"] - corpus_mean_favs
    grouped = grouped[grouped["n_posts"] >= MIN_N].copy()
    grouped.sort_values(["category", "lift_favs"], ascending=[True, False], inplace=True)
    return grouped


def tag_pairs_stats(
    posts_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    categories: tuple[str, ...] = ("general", "species"),
    top_n: int = 30,
) -> pd.DataFrame:
    """Top tag pairs by fav lift. Restricted to the given categories."""
    if posts_df.empty or tags_df.empty:
        return pd.DataFrame()

    filt = tags_df[tags_df["category"].isin(categories)]
    per_post = filt.groupby("post_id")["tag"].apply(list).to_dict()
    fav_by_post = posts_df.set_index("id")["fav_count"].to_dict()
    score_by_post = posts_df.set_index("id")["score_total"].to_dict()
    corpus_mean_favs = posts_df["fav_count"].mean()
    corpus_mean_score = posts_df["score_total"].mean()

    pair_posts: dict[tuple[str, str], list[int]] = {}
    for pid, taglist in per_post.items():
        unique_sorted = sorted(set(taglist))
        for a, b in combinations(unique_sorted, 2):
            pair_posts.setdefault((a, b), []).append(pid)

    rows = []
    for (a, b), pids in pair_posts.items():
        if len(pids) < MIN_N:
            continue
        favs = [fav_by_post[p] for p in pids if p in fav_by_post]
        scores = [score_by_post[p] for p in pids if p in score_by_post]
        if not favs:
            continue
        mean_favs = sum(favs) / len(favs)
        mean_score = sum(scores) / len(scores)
        rows.append({
            "tag_a": a,
            "tag_b": b,
            "n_posts": len(pids),
            "mean_favs": mean_favs,
            "mean_score": mean_score,
            "lift_favs": mean_favs - corpus_mean_favs,
            "lift_score": mean_score - corpus_mean_score,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values("lift_favs", ascending=False, inplace=True)
    return df.head(top_n)


def co_occurrence_matrix(
    tags_df: pd.DataFrame,
    category: str = "general",
    top_n: int = 20,
) -> pd.DataFrame:
    """Symmetric count matrix for the top_n most frequent tags in a category."""
    if tags_df.empty:
        return pd.DataFrame()

    filt = tags_df[tags_df["category"] == category]
    if filt.empty:
        return pd.DataFrame()

    top_tags = filt["tag"].value_counts().head(top_n).index.tolist()
    sub = filt[filt["tag"].isin(top_tags)]
    matrix = pd.DataFrame(0, index=top_tags, columns=top_tags, dtype=int)
    per_post = sub.groupby("post_id")["tag"].apply(list)
    for taglist in per_post:
        unique = [t for t in set(taglist) if t in top_tags]
        for a in unique:
            for b in unique:
                matrix.at[a, b] += 1
    return matrix


def top_bottom_posts(posts_df: pd.DataFrame, by: str = "fav_count", n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    if posts_df.empty:
        return posts_df, posts_df
    sorted_df = posts_df.sort_values(by, ascending=False)
    return sorted_df.head(n), sorted_df.tail(n).iloc[::-1]

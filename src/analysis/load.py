"""Load cached e6ai JSON into pandas DataFrames."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TAG_CATEGORIES = ("general", "species", "character", "meta", "copyright", "artist")


def load_posts(json_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return build_dataframes(raw)


def build_dataframes(raw: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    post_rows = []
    tag_rows = []
    for p in raw:
        score = p.get("score", {}) or {}
        tags = p.get("tags", {}) or {}
        row = {
            "id": p["id"],
            "created_at": p.get("created_at"),
            "rating": p.get("rating"),
            "score_up": score.get("up", 0),
            "score_down": score.get("down", 0),
            "score_total": score.get("total", 0),
            "fav_count": p.get("fav_count", 0),
            "comment_count": p.get("comment_count", 0),
        }
        for cat in TAG_CATEGORIES:
            vals = tags.get(cat, []) or []
            row[f"tags_{cat}"] = " ".join(vals)
            for t in vals:
                tag_rows.append({
                    "post_id": p["id"],
                    "category": cat,
                    "tag": t,
                    "score_total": row["score_total"],
                    "fav_count": row["fav_count"],
                })
        post_rows.append(row)

    posts_df = pd.DataFrame(post_rows)
    if not posts_df.empty:
        posts_df["created_at"] = pd.to_datetime(posts_df["created_at"], errors="coerce", utc=True)
    tags_df = pd.DataFrame(tag_rows)
    return posts_df, tags_df

"""Full and incremental fetch strategies for e6ai."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .client import E6aiClient, resolve_proxy

DATA_DIR = Path("data") / "e6ai"


def cache_path(tag: str) -> Path:
    safe = tag.replace("/", "_").replace("\\", "_")
    return DATA_DIR / f"{safe}.json"


def load_cache(tag: str) -> list[dict]:
    p = cache_path(tag)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(tag: str, posts: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path(tag).open("w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)


def _make_client(proxy: str | None) -> E6aiClient:
    load_dotenv()
    login = os.environ.get("E6AI_LOGIN", "").strip()
    api_key = os.environ.get("E6AI_API_KEY", "").strip()
    return E6aiClient(login=login, api_key=api_key, proxy=resolve_proxy(proxy))


def fetch_full(tag: str, proxy: str | None = None) -> list[dict]:
    posts: list[dict] = []
    with _make_client(proxy) as client:
        for i, post in enumerate(client.iter_posts_descending(tag), 1):
            posts.append(post)
            if i % 100 == 0:
                print(f"  fetched {i} posts...", file=sys.stderr)
    save_cache(tag, posts)
    print(f"  full fetch complete: {len(posts)} posts -> {cache_path(tag)}", file=sys.stderr)
    return posts


def fetch_incremental(tag: str, proxy: str | None = None) -> list[dict]:
    existing = load_cache(tag)
    if not existing:
        print("  no cache, falling back to full fetch", file=sys.stderr)
        return fetch_full(tag, proxy=proxy)

    by_id: dict[int, dict] = {p["id"]: p for p in existing}
    max_id = max(by_id.keys())
    new_count = 0
    with _make_client(proxy) as client:
        for post in client.iter_posts_ascending_after(tag, max_id):
            by_id[post["id"]] = post
            new_count += 1
    merged = sorted(by_id.values(), key=lambda p: p["id"], reverse=True)
    save_cache(tag, merged)
    print(f"  incremental fetch: +{new_count} new, total {len(merged)}", file=sys.stderr)
    return merged


def fetch(tag: str, mode: str | None = None, proxy: str | None = None) -> list[dict]:
    if mode is None:
        mode = "incremental" if cache_path(tag).exists() else "full"
    if mode == "full":
        return fetch_full(tag, proxy=proxy)
    if mode == "incremental":
        return fetch_incremental(tag, proxy=proxy)
    raise ValueError(f"unknown mode: {mode!r}")

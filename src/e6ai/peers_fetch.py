"""Fetch peer data: top uploaders + per-uploader post samples + per-tag site samples.

Cached to data/e6ai/_peers/<name>.json so re-runs don't re-hammer the API.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .client import E6aiClient, resolve_proxy
from .fetch import DATA_DIR

PEERS_DIR = DATA_DIR / "_peers"
TAGBENCH_DIR = DATA_DIR / "_tagbench"


def _make_client(proxy: str | None) -> E6aiClient:
    load_dotenv()
    login = os.environ.get("E6AI_LOGIN", "").strip()
    api_key = os.environ.get("E6AI_API_KEY", "").strip()
    return E6aiClient(login=login, api_key=api_key, proxy=resolve_proxy(proxy))


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def _peer_cache(name: str) -> Path:
    return PEERS_DIR / f"{_safe(name)}.json"


def _tagbench_cache(tag: str) -> Path:
    return TAGBENCH_DIR / f"{_safe(tag)}.json"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, posts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_top_peers(
    sample_size: int,
    top_n: int,
    exclude_uploaders: list[str] | None,
    proxy: str | None,
    refresh: bool,
) -> list[tuple[dict, list[dict]]]:
    """Returns [(user_dict, posts), ...] for the top_n uploaders."""
    PEERS_DIR.mkdir(parents=True, exist_ok=True)
    with _make_client(proxy) as client:
        excludes = tuple(exclude_uploaders or ())
        users = client.top_uploaders(limit=top_n, exclude_names=excludes)
        results: list[tuple[dict, list[dict]]] = []
        for u in users:
            name = u["name"]
            cache = _peer_cache(name)
            if cache.exists() and not refresh:
                posts = _load(cache)
                if len(posts) >= sample_size:
                    print(f"  cached: {name} ({len(posts)} posts)", file=sys.stderr)
                    results.append((u, posts[:sample_size]))
                    continue
            print(f"  fetching: {name} (target {sample_size} posts)", file=sys.stderr)
            try:
                posts = client.fetch_uploader_posts(name, max_posts=sample_size)
            except Exception as e:  # noqa: BLE001
                print(f"  ! skipped {name}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            _save(cache, posts)
            results.append((u, posts))
        return results


def fetch_tag_benchmarks(
    tags: list[str],
    sample_per_tag: int,
    proxy: str | None,
    refresh: bool,
) -> dict[str, list[dict]]:
    """Returns {tag: posts}. Sample is the latest `sample_per_tag` site posts for that tag."""
    TAGBENCH_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {}
    with _make_client(proxy) as client:
        for tag in tags:
            cache = _tagbench_cache(tag)
            if cache.exists() and not refresh:
                posts = _load(cache)
                if len(posts) >= sample_per_tag:
                    print(f"  cached tag: {tag} ({len(posts)} posts)", file=sys.stderr)
                    out[tag] = posts[:sample_per_tag]
                    continue
            print(f"  fetching tag: {tag} (target {sample_per_tag})", file=sys.stderr)
            posts: list[dict] = []
            try:
                for p in client.iter_posts_descending(tag):
                    posts.append(p)
                    if len(posts) >= sample_per_tag:
                        break
            except Exception as e:  # noqa: BLE001
                print(f"  ! skipped tag {tag}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            _save(cache, posts)
            out[tag] = posts
    return out

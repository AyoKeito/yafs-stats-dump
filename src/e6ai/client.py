"""Rate-limited, paginated e6ai.net API client."""
from __future__ import annotations

import os
import time
from typing import Iterator

import httpx

BASE_URL = "https://e6ai.net"
USER_AGENT = "ai_suite/0.1 (engagement-stats; by AyoKeito on e6ai)"
MAX_LIMIT = 320
MIN_INTERVAL_S = 1.05  # stay below the 1 req/s sustained guideline


def resolve_proxy(cli_proxy: str | None) -> str | None:
    """Priority: CLI flag -> E6AI_PROXY -> AI_SUITE_PROXY -> None."""
    if cli_proxy:
        return cli_proxy
    return os.environ.get("E6AI_PROXY") or os.environ.get("AI_SUITE_PROXY") or None


class E6aiClient:
    def __init__(
        self,
        login: str,
        api_key: str,
        proxy: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not login or not api_key:
            raise ValueError("E6AI_LOGIN and E6AI_API_KEY must be set")
        self._login = login
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            proxy=proxy,
        )
        self._last_request_ts = 0.0

    def __enter__(self) -> "E6aiClient":
        return self

    def __exit__(self, *exc) -> None:
        self._client.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - elapsed)

    def _get(self, path: str, params: dict) -> httpx.Response:
        merged = {**params, "login": self._login, "api_key": self._api_key}
        backoff = 1.0
        for attempt in range(4):
            self._throttle()
            self._last_request_ts = time.monotonic()
            r = self._client.get(path, params=merged)
            if r.status_code == 503 and attempt < 3:
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.is_error:
                # Strip credentials before bubbling up so the URL never lands in logs/tracebacks
                safe_params = {k: v for k, v in params.items()}
                raise httpx.HTTPStatusError(
                    f"{r.status_code} {r.reason_phrase} on GET {path} params={safe_params}",
                    request=r.request,
                    response=r,
                )
            return r
        raise RuntimeError("unreachable")

    def search_posts_page(self, tags: str, page: str | None = None) -> list[dict]:
        params: dict = {"tags": tags, "limit": MAX_LIMIT}
        if page:
            params["page"] = page
        data = self._get("/posts.json", params).json()
        posts = data.get("posts", data) if isinstance(data, dict) else data
        return posts or []

    def top_uploaders(self, limit: int = 10, exclude_names: tuple[str, ...] = ()) -> list[dict]:
        """Return user dicts sorted by post_upload_count desc.

        Exclusion is case-insensitive and ignores underscores so 'ayo_keito' matches 'AyoKeito'.
        """
        def norm(s: str) -> str:
            return s.lower().replace("_", "")

        data = self._get(
            "/users.json",
            params={"limit": min(limit + len(exclude_names) + 5, 100), "search[order]": "post_upload_count"},
        ).json()
        users = data if isinstance(data, list) else data.get("users", [])
        excluded = {norm(n) for n in exclude_names if n}
        out = [u for u in users if norm(u.get("name", "")) not in excluded]
        return out[:limit]

    def fetch_uploader_posts(self, name: str, max_posts: int) -> list[dict]:
        """Fetch up to max_posts most recent posts uploaded by `name`."""
        out: list[dict] = []
        for post in self.iter_posts_descending(f"user:{name}"):
            out.append(post)
            if len(out) >= max_posts:
                break
        return out

    def iter_posts_descending(self, tags: str) -> Iterator[dict]:
        """Walk from newest to oldest using page=b<min_id> cursor."""
        cursor: str | None = None
        while True:
            posts = self.search_posts_page(tags, page=cursor)
            if not posts:
                return
            for p in posts:
                yield p
            cursor = f"b{min(p['id'] for p in posts)}"

    def iter_posts_ascending_after(self, tags: str, after_id: int) -> Iterator[dict]:
        """Walk from oldest-newer-than-after_id upward using page=a<max_id> cursor."""
        cursor = f"a{after_id}"
        while True:
            posts = self.search_posts_page(tags, page=cursor)
            if not posts:
                return
            for p in posts:
                yield p
            cursor = f"a{max(p['id'] for p in posts)}"

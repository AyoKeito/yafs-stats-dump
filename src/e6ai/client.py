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
            r.raise_for_status()
            return r
        raise RuntimeError("unreachable")

    def search_posts_page(self, tags: str, page: str | None = None) -> list[dict]:
        params: dict = {"tags": tags, "limit": MAX_LIMIT}
        if page:
            params["page"] = page
        data = self._get("/posts.json", params).json()
        posts = data.get("posts", data) if isinstance(data, dict) else data
        return posts or []

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

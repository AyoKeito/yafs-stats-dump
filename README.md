# ai_suite

Personal toolkit for analyzing engagement stats on your own NSFW furry art across booru sites.
First supported site: **e6ai.net**.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for Python env / dependency management.

```bash
uv venv
uv pip install -r requirements.txt
copy .env.example .env
```

(Fallback without uv: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`.)

Fill in `.env`:

```
E6AI_LOGIN=AyoKeito
E6AI_API_KEY=...            # generate at https://e6ai.net/users/home
E6AI_PROXY=                 # optional, e.g. http://user:pass@host:port
AI_SUITE_PROXY=             # optional global fallback
```

> **Security:** if you pasted your API key anywhere public (chat, screenshots), rotate it
> at https://e6ai.net/users/home before running.

## Usage

```bash
# Fetch all posts tagged with your artist name
uv run python -m src.cli e6ai fetch --tag AyoKeito

# Analyze the cached data and produce report + charts
uv run python -m src.cli e6ai analyze --tag AyoKeito

# Do both
uv run python -m src.cli e6ai run --tag AyoKeito
```

Flags:

- `--mode full|incremental` — default: incremental if cache exists, else full.
- `--proxy http://host:port` — overrides env vars for this run.

Outputs:

- Raw cache: `data/e6ai/<tag>.json`
- Report:    `out/e6ai/<tag>/report.md`
- CSV:       `out/e6ai/<tag>/posts.csv`
- Charts:    `out/e6ai/<tag>/charts/*.png`

## Proxy priority

`--proxy` CLI flag → `<SITE>_PROXY` env var → `AI_SUITE_PROXY` env var → direct.

## Adding new sites later

Mirror `src/e6ai/` as `src/<site>/` with its own `client.py` + `fetch.py`, add `<SITE>_PROXY` to `.env.example`, and reuse `src/analysis/` for the correlation work.
# yafs-stats-dump

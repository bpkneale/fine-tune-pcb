"""Shared GitHub API helpers for the scrape_* scripts."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

GITHUB_API = "https://api.github.com"

PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "isc",
    "cc0-1.0",
    "unlicense",
    "0bsd",
}


class RateLimited(Exception):
    pass


@dataclass(slots=True)
class IndexRecord:
    sha: str
    repo: str
    path: str
    license: str
    bytes: int

    def to_json(self) -> bytes:
        return orjson.dumps(asdict(self)) + b"\n"


def session_from_env() -> requests.Session:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: set GITHUB_TOKEN (public_repo scope)", file=sys.stderr)
        raise SystemExit(2)
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "fine-tune-pcb/0.1",
        }
    )
    return s


def sleep_until_reset(resp: requests.Response) -> None:
    reset = resp.headers.get("X-RateLimit-Reset")
    if not reset:
        time.sleep(30)
        return
    delay = max(1, int(reset) - int(time.time()) + 2)
    print(f"  rate limited, sleeping {delay}s", file=sys.stderr)
    time.sleep(delay)


@retry(
    retry=retry_if_exception_type(RateLimited),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    reraise=True,
)
def gh_get(s: requests.Session, url: str, params: dict | None = None) -> dict | list:
    resp = s.get(url, params=params, timeout=30)
    if resp.status_code in (403, 429):
        sleep_until_reset(resp)
        raise RateLimited(resp.text[:200])
    resp.raise_for_status()
    return resp.json()


@retry(
    retry=retry_if_exception_type(RateLimited),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
def download_raw(s: requests.Session, repo_full: str, path: str, ref: str | None = None) -> bytes:
    url = f"{GITHUB_API}/repos/{repo_full}/contents/{path}"
    params = {"ref": ref} if ref else None
    resp = s.get(
        url,
        headers={"Accept": "application/vnd.github.raw"},
        params=params,
        timeout=60,
    )
    if resp.status_code in (403, 429):
        sleep_until_reset(resp)
        raise RateLimited(resp.text[:200])
    resp.raise_for_status()
    return resp.content


def load_index(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    seen: set[str] = set()
    with index_path.open("rb") as f:
        for line in f:
            try:
                seen.add(orjson.loads(line)["sha"])
            except Exception:
                continue
    return seen


def normalise_license(spdx: str | None) -> str:
    return (spdx or "").lower()


def is_permissive(spdx: str | None) -> bool:
    return normalise_license(spdx) in PERMISSIVE_LICENSES

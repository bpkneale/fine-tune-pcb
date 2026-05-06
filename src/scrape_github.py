"""Scrape *.kicad_pcb files via GitHub's broad code search API.

Resumable via data/raw/_index.jsonl. License filter accepts a permissive
whitelist only.

Auth: needs $GITHUB_TOKEN with public_repo scope (or finer-grained
equivalent). Code search rate limit is 30 req/min — the loop sleeps 6s
between pages to stay under it. Sharded by file size to push past the
1000-result-per-query cap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from src._gh import (
    GITHUB_API,
    IndexRecord,
    RateLimited,
    download_raw,
    gh_get,
    is_permissive,
    load_index,
    normalise_license,
    session_from_env,
)

SEARCH_PATH = "/search/code"


def search_kicad_pcb(s: requests.Session, max_results: int) -> list[dict]:
    """Yield search hits, sharded by file size (search api caps at 1000 per
    query; 4 brackets × 1000 = 4000 ceiling). Bias toward larger files —
    tiny .kicad_pcb files in the wild are overwhelmingly empty templates."""
    size_brackets = [
        "100000..400000",
        "40000..100000",
        ">400000",
        "15000..40000",
    ]
    seen: set[str] = set()
    out: list[dict] = []
    for bracket in size_brackets:
        if len(out) >= max_results:
            break
        q = f"extension:kicad_pcb size:{bracket}"
        for page in range(1, 11):
            data = gh_get(
                s,
                GITHUB_API + SEARCH_PATH,
                params={"q": q, "per_page": 100, "page": page},
            )
            items = data.get("items", []) if isinstance(data, dict) else []
            if not items:
                break
            for item in items:
                sha = item["sha"]
                if sha in seen:
                    continue
                seen.add(sha)
                out.append(item)
                if len(out) >= max_results:
                    return out
            time.sleep(6)  # 30 req/min limit on /search/code
    return out


def fetch_license(s: requests.Session, repo_full: str, cache: dict) -> str | None:
    if repo_full in cache:
        return cache[repo_full]
    try:
        data = gh_get(s, f"{GITHUB_API}/repos/{repo_full}")
    except requests.HTTPError:
        cache[repo_full] = None
        return None
    spdx = (data.get("license") or {}).get("spdx_id") if isinstance(data, dict) else None
    cache[repo_full] = normalise_license(spdx)
    return cache[repo_full]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--limit", type=int, default=200, help="max new files this run")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "_index.jsonl"
    already = load_index(index_path)
    print(f"already have {len(already)} blobs in index")

    s = session_from_env()
    hits = search_kicad_pcb(s, max_results=args.limit * 4)
    print(f"search returned {len(hits)} candidate hits")

    license_cache: dict[str, str | None] = {}
    new_count = 0

    with index_path.open("ab") as idx_f:
        for item in tqdm(hits, desc="download"):
            if new_count >= args.limit:
                break
            sha = item["sha"]
            if sha in already:
                continue
            repo = item["repository"]["full_name"]
            lic = fetch_license(s, repo, license_cache) or ""
            if not is_permissive(lic):
                continue
            try:
                blob = download_raw(s, repo, item["path"])
            except (RateLimited, requests.HTTPError) as e:
                tqdm.write(f"  skip {repo}/{item['path']}: {e}")
                continue
            if not blob.lstrip().startswith(b"(kicad_pcb"):
                continue
            (args.out / f"{sha}.kicad_pcb").write_bytes(blob)
            rec = IndexRecord(
                sha=sha,
                repo=repo,
                path=item["path"],
                license=lic,
                bytes=len(blob),
            )
            idx_f.write(rec.to_json())
            idx_f.flush()
            already.add(sha)
            new_count += 1

    print(f"downloaded {new_count} new blobs, total {len(already)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scrape *.kicad_pcb files from a curated list of high-quality
open-hardware GitHub orgs.

Why this exists alongside scrape_github.py:
  - Code search caps at 1000 hits per query. Even with size-bracket
    sharding we miss boards.
  - Targeted orgs (Adafruit, Sparkfun, ...) are denser in real designs
    and lower in junk than the open web.

Algorithm:
  for each org:
    list repos (paginated, includes license metadata for free)
    for each non-archived repo with a permissive license:
        fetch git tree at HEAD (recursive)
        for each *.kicad_pcb path:
            download via raw contents API, dedupe by blob sha
            append to data/raw/_index.jsonl

Reuses scrape_github's _index.jsonl so dedupe is automatic across both.
"""

from __future__ import annotations

import argparse
import sys
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

# Curated default — high-quality, mostly-KiCad open-hardware orgs.
# Override with --orgs at the CLI.
DEFAULT_ORGS = [
    "adafruit",
    "sparkfun",
    "Seeed-Studio",
    "pololu",
    "olimex",
    "pimoroni",
    "arturo182",
    "watterott",
    "particle-iot",
    "betaflight",
    "openinputlabs",
    "keebio",
    "1bitsquared",
    "adamgreig",
]


def list_org_repos(s: requests.Session, org: str) -> list[dict]:
    """Paginated /orgs/{org}/repos. Returns repo dicts including the
    `license`, `archived`, `default_branch`, `full_name`, and `size` fields."""
    out: list[dict] = []
    page = 1
    while True:
        data = gh_get(
            s,
            f"{GITHUB_API}/orgs/{org}/repos",
            params={"per_page": 100, "page": page, "type": "public"},
        )
        if not isinstance(data, list):
            break
        if not data:
            break
        out.extend(data)
        if len(data) < 100:
            break
        page += 1
    return out


def list_user_repos(s: requests.Session, user: str) -> list[dict]:
    """Some entries in DEFAULT_ORGS (e.g. arturo182) are user accounts, not
    orgs. /users/{user}/repos works for both but doesn't include private."""
    out: list[dict] = []
    page = 1
    while True:
        data = gh_get(
            s,
            f"{GITHUB_API}/users/{user}/repos",
            params={"per_page": 100, "page": page, "type": "owner"},
        )
        if not isinstance(data, list):
            break
        if not data:
            break
        out.extend(data)
        if len(data) < 100:
            break
        page += 1
    return out


def list_repos(s: requests.Session, owner: str) -> list[dict]:
    """Try org endpoint first; fall back to user endpoint on 404."""
    try:
        return list_org_repos(s, owner)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return list_user_repos(s, owner)
        raise


def get_tree(s: requests.Session, repo_full: str, branch: str) -> list[dict]:
    """Recursive git tree at the tip of the given branch. Returns list of
    {path, sha, type, size} entries, or [] on failure (empty repo, etc.)."""
    try:
        data = gh_get(
            s,
            f"{GITHUB_API}/repos/{repo_full}/git/trees/{branch}",
            params={"recursive": "1"},
        )
    except requests.HTTPError:
        return []
    if not isinstance(data, dict):
        return []
    return data.get("tree", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument(
        "--orgs",
        nargs="+",
        default=DEFAULT_ORGS,
        help="GitHub orgs/users to scrape (defaults to curated list)",
    )
    ap.add_argument("--limit", type=int, default=10_000, help="hard cap on new files")
    ap.add_argument(
        "--max-file-bytes",
        type=int,
        default=2_000_000,
        help="skip absurdly large .kicad_pcb files",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "_index.jsonl"
    already = load_index(index_path)
    print(f"already have {len(already)} blobs in index")

    s = session_from_env()

    new_count = 0
    with index_path.open("ab") as idx_f:
        for owner in args.orgs:
            if new_count >= args.limit:
                break
            try:
                repos = list_repos(s, owner)
            except requests.HTTPError as e:
                print(f"  skip {owner}: {e}", file=sys.stderr)
                continue
            print(f"{owner}: {len(repos)} repos")
            for repo in tqdm(repos, desc=owner, leave=False):
                if new_count >= args.limit:
                    break
                if repo.get("archived"):
                    continue
                if repo.get("fork"):
                    # forks usually duplicate upstream content
                    continue
                spdx = (repo.get("license") or {}).get("spdx_id")
                lic = normalise_license(spdx)
                if not is_permissive(lic):
                    continue
                full = repo["full_name"]
                branch = repo.get("default_branch") or "main"
                tree = get_tree(s, full, branch)
                kicad_paths = [
                    t["path"]
                    for t in tree
                    if t.get("type") == "blob"
                    and t.get("path", "").endswith(".kicad_pcb")
                    and (t.get("size") or 0) <= args.max_file_bytes
                    and t["sha"] not in already
                ]
                for path in kicad_paths:
                    if new_count >= args.limit:
                        break
                    try:
                        blob = download_raw(s, full, path, ref=branch)
                    except (RateLimited, requests.HTTPError) as e:
                        tqdm.write(f"  skip {full}/{path}: {e}")
                        continue
                    if not blob.lstrip().startswith(b"(kicad_pcb"):
                        continue
                    # Use the tree-blob sha as the canonical id (matches search API).
                    sha = next(
                        t["sha"] for t in tree if t.get("path") == path
                    )
                    if sha in already:
                        continue
                    (args.out / f"{sha}.kicad_pcb").write_bytes(blob)
                    rec = IndexRecord(
                        sha=sha,
                        repo=full,
                        path=path,
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

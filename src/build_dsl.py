"""Convert parsed board records into (prompt, completion) DSL pairs.

For each board with N components, emit N-1 examples — one per placement
step k=1..N-1. Components are placed in descending pin-count order
(important parts first); the model must predict component k+1 given the
netlist + outline + already-placed k components.

Coordinate system: origin shifted to outline bottom-left, quantised to
GRID_MM. Rotation snapped to {0, 90, 180, 270}.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import orjson
from tqdm import tqdm

GRID_MM = 0.05
ROT_BUCKETS = (0, 90, 180, 270)

# Orders that augmentation will produce per board. Keep these stable —
# `order` is recorded in each example for traceability.
ALL_ORDERS = ("pin_desc", "connectivity_bfs", "random_seeded")


@dataclass(slots=True)
class Comp:
    ref: str
    footprint: str
    x: float
    y: float
    rot: int
    layer: str
    pads: list[dict]  # [{"number": "1", "net": "VCC"}, ...]


def _quantise(v: float) -> float:
    return round(v / GRID_MM) * GRID_MM


def _snap_rot(r: float) -> int:
    r = r % 360
    return min(ROT_BUCKETS, key=lambda b: min(abs(r - b), 360 - abs(r - b)))


def _normalise(record: dict) -> tuple[list[Comp], tuple[float, float], tuple[float, float]]:
    x0, y0, x1, y1 = record["outline_bbox"]
    width = max(x1 - x0, 0.0)
    height = max(y1 - y0, 0.0)
    comps: list[Comp] = []
    for c in record["components"]:
        comps.append(
            Comp(
                ref=c["ref"],
                footprint=c["footprint"] or "?",
                x=_quantise(c["x"] - x0),
                y=_quantise(c["y"] - y0),
                rot=_snap_rot(c["rot"]),
                layer=c["layer"],
                pads=c["pads"],
            )
        )
    return comps, (width, height), (x0, y0)


def _order_pin_desc(comps: list[Comp]) -> list[Comp]:
    """Highest pin count first; tie-break by ref designator."""
    return sorted(comps, key=lambda c: (-len(c.pads), c.ref))


def _order_connectivity_bfs(comps: list[Comp]) -> list[Comp]:
    """Start at the highest-pin-count component; visit neighbours (sharing
    a real net) in descending pin-count order. Disconnected fragments are
    appended in pin-count order. Models a 'place outwards from the anchor
    IC' workflow that real PCB designers use."""
    if not comps:
        return []
    by_ref = {c.ref: c for c in comps}
    # Adjacency: two components are neighbours if they share at least one
    # named net (not None and not unconnected-*).
    adj: dict[str, set[str]] = defaultdict(set)
    by_net: dict[str, list[str]] = defaultdict(list)
    for c in comps:
        for pad in c.pads:
            n = pad.get("net")
            if not n or n.startswith("unconnected-"):
                continue
            by_net[n].append(c.ref)
    for refs in by_net.values():
        unique = set(refs)
        for r in unique:
            adj[r].update(unique - {r})

    seed_order = _order_pin_desc(comps)
    visited: set[str] = set()
    out: list[Comp] = []
    for seed in seed_order:
        if seed.ref in visited:
            continue
        q: deque[str] = deque([seed.ref])
        visited.add(seed.ref)
        while q:
            ref = q.popleft()
            out.append(by_ref[ref])
            neighbours = sorted(
                adj[ref] - visited,
                key=lambda r: (-len(by_ref[r].pads), r),
            )
            for n in neighbours:
                visited.add(n)
                q.append(n)
    return out


def _order_random_seeded(comps: list[Comp], seed: int) -> list[Comp]:
    """Deterministic shuffle for a board (seeded by board sha hash)."""
    rng = random.Random(seed)
    out = list(comps)
    rng.shuffle(out)
    return out


def _build_netlist(comps: list[Comp], limit: int = 80) -> list[tuple[str, list[str]]]:
    """Returns [(net_name, [ref.pad, ...]), ...] sorted by descending fanout.
    Only nets connecting >=2 pads are included; capped to `limit` nets."""
    by_net: dict[str, list[str]] = defaultdict(list)
    for c in comps:
        for pad in c.pads:
            net = pad.get("net")
            num = pad.get("number") or ""
            if not net:
                continue
            by_net[net].append(f"{c.ref}.{num}")
    nets = [(n, refs) for n, refs in by_net.items() if len(refs) >= 2]
    nets.sort(key=lambda nr: (-len(nr[1]), nr[0]))
    return nets[:limit]


def _fmt_coord(v: float) -> str:
    return f"{v:.2f}"


def _render_prompt(
    bbox: tuple[float, float],
    layer_count: int,
    netlist: list[tuple[str, list[str]]],
    placed: list[Comp],
    to_place: Comp,
) -> str:
    w, h = bbox
    lines: list[str] = []
    lines.append("<board>")
    lines.append(f"  <outline>RECT 0 0 {_fmt_coord(w)} {_fmt_coord(h)}</outline>")
    lines.append(f"  <layers>{layer_count}</layers>")
    lines.append("  <netlist>")
    for net, refs in netlist:
        lines.append(f"    NET {net}: {', '.join(refs)}")
    lines.append("  </netlist>")
    lines.append("  <placed>")
    for c in placed:
        lines.append(
            f"    {c.ref} {c.footprint} at ({_fmt_coord(c.x)}, {_fmt_coord(c.y)}) "
            f"rot {c.rot} layer {c.layer}"
        )
    lines.append("  </placed>")
    lines.append("  <to_place>")
    nets = sorted(
        {
            p["net"]
            for p in to_place.pads
            if p.get("net") and not p["net"].startswith("unconnected-")
        }
    )
    nets_s = ", ".join(nets) if nets else "none"
    lines.append(f"    {to_place.ref} {to_place.footprint} (nets: {nets_s})")
    lines.append("  </to_place>")
    lines.append("</board>")
    lines.append(f"Place {to_place.ref}.")
    return "\n".join(lines)


def _render_completion(c: Comp) -> str:
    return (
        f"PLACE {c.ref} at ({_fmt_coord(c.x)}, {_fmt_coord(c.y)}) "
        f"rot {c.rot} layer {c.layer}"
    )


def _orderings(comps: list[Comp], sha: str, orders: tuple[str, ...]) -> dict[str, list[Comp]]:
    """Compute each requested placement order. Random uses a seed derived
    from the board sha so the ordering is reproducible across runs."""
    out: dict[str, list[Comp]] = {}
    for name in orders:
        if name == "pin_desc":
            out[name] = _order_pin_desc(comps)
        elif name == "connectivity_bfs":
            out[name] = _order_connectivity_bfs(comps)
        elif name == "random_seeded":
            seed = int(sha[:8], 16) if sha else 0
            out[name] = _order_random_seeded(comps, seed)
        else:
            raise ValueError(f"unknown ordering: {name}")
    return out


def build_examples(record: dict, orders: tuple[str, ...] = ALL_ORDERS) -> list[dict]:
    comps, bbox, _ = _normalise(record)
    if bbox[0] <= 0 or bbox[1] <= 0:
        return []
    netlist = _build_netlist(comps)
    layer_count = record.get("layer_count", 2)
    sha = record["sha"]

    out: list[dict] = []
    for order_name, ordered in _orderings(comps, sha, orders).items():
        for k in range(1, len(ordered)):
            placed = ordered[:k]
            to_place = ordered[k]
            out.append(
                {
                    "sha": sha,
                    "order": order_name,
                    "step": k,
                    "prompt": _render_prompt(bbox, layer_count, netlist, placed, to_place),
                    "completion": _render_completion(to_place),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/parsed/boards.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/dsl/train.jsonl"))
    ap.add_argument(
        "--orders",
        nargs="+",
        default=list(ALL_ORDERS),
        choices=ALL_ORDERS,
        help="placement orderings to emit per board (each multiplies example count)",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_boards = 0
    n_examples = 0
    with args.inp.open("rb") as fin, args.out.open("wb") as fout:
        for line in tqdm(fin, desc="build"):
            try:
                rec = orjson.loads(line)
            except Exception:
                continue
            examples = build_examples(rec, orders=tuple(args.orders))
            for ex in examples:
                fout.write(orjson.dumps(ex) + b"\n")
            n_boards += 1
            n_examples += len(examples)

    print(f"boards={n_boards}  examples={n_examples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

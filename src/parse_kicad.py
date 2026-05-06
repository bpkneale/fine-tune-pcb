"""Parse .kicad_pcb files into structured JSON board records.

Uses kiutils for s-expression parsing. Applies the quality filters from
the plan (Phase 2): 10..200 components, has board outline, all components
have valid positions, <20% of components stacked at origin.

Output: JSONL where each line is one board record (see BoardRecord schema).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from tqdm import tqdm

try:
    from kiutils.board import Board
except ImportError:
    print("kiutils not installed — `pip install -e .`", file=sys.stderr)
    raise

MIN_COMPONENTS = 10
MAX_COMPONENTS = 200
MAX_ORIGIN_STACK_RATIO = 0.20
SYNTH_OUTLINE_MARGIN_MM = 5.0


@dataclass(slots=True)
class Pad:
    number: str
    net: str | None


@dataclass(slots=True)
class Component:
    ref: str
    footprint: str
    x: float
    y: float
    rot: float
    layer: str  # "F" or "B"
    pads: list[Pad] = field(default_factory=list)


@dataclass(slots=True)
class BoardRecord:
    sha: str
    outline_bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    layer_count: int
    components: list[Component]

    def to_json(self) -> bytes:
        return orjson.dumps(
            {
                "sha": self.sha,
                "outline_bbox": list(self.outline_bbox),
                "layer_count": self.layer_count,
                "components": [
                    {
                        "ref": c.ref,
                        "footprint": c.footprint,
                        "x": c.x,
                        "y": c.y,
                        "rot": c.rot,
                        "layer": c.layer,
                        "pads": [{"number": p.number, "net": p.net} for p in c.pads],
                    }
                    for c in self.components
                ],
            }
        ) + b"\n"


def _ref_from_footprint(fp) -> str | None:
    """In current kiutils, fp.properties is a {name: value} dict for newer
    KiCad files; older files use FpText graphic items with type=reference."""
    props = getattr(fp, "properties", None)
    if isinstance(props, dict):
        for k, v in props.items():
            if str(k).lower() == "reference" and v:
                return str(v)
    elif props:  # list-of-objects fallback
        for prop in props:
            key = getattr(prop, "key", None) or getattr(prop, "name", None)
            if key and str(key).lower() == "reference":
                val = getattr(prop, "value", None)
                if val:
                    return str(val)
    for txt in getattr(fp, "graphicItems", []) or []:
        if getattr(txt, "type", None) == "reference":
            v = getattr(txt, "text", None)
            if v:
                return str(v)
    return None


def _layer_short(layer: str | None) -> str:
    if not layer:
        return "F"
    return "B" if layer.startswith("B.") else "F"


def _outline_bbox(board) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for item in getattr(board, "graphicItems", []) or []:
        if getattr(item, "layer", None) != "Edge.Cuts":
            continue
        for attr in ("start", "end", "center", "midPoint"):
            pt = getattr(item, attr, None)
            if pt is None:
                continue
            xs.append(float(pt.X))
            ys.append(float(pt.Y))
    if len(xs) < 2:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _load_board_utf8(path: Path):
    """kiutils.Board.from_file uses open() with locale-default encoding, which
    on Windows is cp1252 and chokes on UTF-8 bytes in author/comment fields.
    Strip the file down to ASCII (PCB geometry doesn't depend on those bytes)
    and load via a temp file."""
    raw = path.read_bytes()
    ascii_bytes = raw.decode("utf-8", errors="replace").encode("ascii", errors="replace")
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".kicad_pcb", delete=False
    ) as tmp:
        tmp.write(ascii_bytes)
        tmp_path = tmp.name
    try:
        return Board.from_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _synth_outline_from_components(comps: list["Component"]) -> tuple[float, float, float, float]:
    xs = [c.x for c in comps]
    ys = [c.y for c in comps]
    m = SYNTH_OUTLINE_MARGIN_MM
    return (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)


def parse_board(path: Path, sha: str) -> tuple[BoardRecord | None, str]:
    board = _load_board_utf8(path)

    bbox = _outline_bbox(board)

    layer_count = len(getattr(board, "layers", []) or []) or 2
    # Heuristic: only count copper layers if `layers` includes all defined layers
    copper_layers = [
        ly for ly in (getattr(board, "layers", []) or [])
        if getattr(ly, "name", "").endswith(".Cu")
    ]
    if copper_layers:
        layer_count = len(copper_layers)

    comps: list[Component] = []
    for fp in getattr(board, "footprints", []) or []:
        ref = _ref_from_footprint(fp)
        if not ref:
            continue
        pos = getattr(fp, "position", None)
        if pos is None or pos.X is None or pos.Y is None:
            continue
        comp = Component(
            ref=ref,
            footprint=str(getattr(fp, "entryName", "") or getattr(fp, "libId", "")),
            x=float(pos.X),
            y=float(pos.Y),
            rot=float(getattr(pos, "angle", 0) or 0),
            layer=_layer_short(getattr(fp, "layer", None)),
        )
        for pad in getattr(fp, "pads", []) or []:
            net = getattr(pad, "net", None)
            net_name = getattr(net, "name", None) if net else None
            comp.pads.append(
                Pad(number=str(getattr(pad, "number", "") or ""), net=net_name or None)
            )
        comps.append(comp)

    if len(comps) == 0:
        return None, "no_components"
    if len(comps) < MIN_COMPONENTS:
        return None, "too_few_components"
    if len(comps) > MAX_COMPONENTS:
        return None, "too_many_components"

    if bbox is None:
        bbox = _synth_outline_from_components(comps)

    with_pads = sum(1 for c in comps if c.pads)
    if with_pads / len(comps) < 0.5:
        return None, "no_inline_pads"

    at_origin = sum(1 for c in comps if abs(c.x) < 0.01 and abs(c.y) < 0.01)
    if at_origin / len(comps) > MAX_ORIGIN_STACK_RATIO:
        return None, "origin_stacked"

    return BoardRecord(
        sha=sha,
        outline_bbox=bbox,
        layer_count=layer_count,
        components=comps,
    ), "kept"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/parsed/boards.jsonl"))
    ap.add_argument("--verbose-errors", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(args.inp.glob("*.kicad_pcb"))
    print(f"parsing {len(files)} files")

    from collections import Counter

    reasons: Counter[str] = Counter()
    errored = 0
    with args.out.open("wb") as f:
        for path in tqdm(files, desc="parse"):
            sha = path.stem
            try:
                rec, reason = parse_board(path, sha)
            except Exception as e:
                errored += 1
                reasons["error"] += 1
                if args.verbose_errors:
                    tqdm.write(f"  error {sha}: {e}")
                    traceback.print_exc()
                continue
            reasons[reason] += 1
            if rec is not None:
                f.write(rec.to_json())

    print(f"kept={reasons['kept']}  errored={errored}")
    print("rejection breakdown:")
    for reason, n in reasons.most_common():
        print(f"  {reason:24s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

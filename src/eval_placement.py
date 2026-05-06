"""Held-out evaluation of a placement LoRA.

Loads the base model + adapter, generates a completion for every example
in the eval split, parses the predicted PLACE statement, and computes:

  - coord_l2_mm        : Euclidean distance between predicted and ground-truth
                         placement (mm).
  - rot_acc            : fraction with exact-match rotation (in {0,90,180,270}).
  - layer_acc          : fraction with correct layer (F vs B).
  - overlap_rate       : fraction whose 2mm bounding box around the predicted
                         centre overlaps any already-placed component.
  - in_outline_rate    : fraction whose predicted centre falls within the
                         board outline rectangle.
  - parse_success_rate : fraction of completions whose PLACE was parseable.

Two baselines are reported alongside the model:

  - random_valid : uniform random point inside the outline.
  - net_centroid : centre of mass of already-placed components that share
                   a net with the target. Falls back to outline centre.

Run inside WSL2 after training.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import orjson
from tqdm import tqdm

PLACE_RE = re.compile(
    r"PLACE\s+(?P<ref>\S+)\s+at\s+\(\s*(?P<x>-?[\d.]+)\s*,\s*(?P<y>-?[\d.]+)\s*\)"
    r"\s+rot\s+(?P<rot>-?\d+)\s+layer\s+(?P<layer>[FB])",
    re.IGNORECASE,
)

OUTLINE_RE = re.compile(r"<outline>RECT\s+0\s+0\s+([\d.]+)\s+([\d.]+)</outline>")
PLACED_LINE_RE = re.compile(
    r"^\s+(\S+)\s+\S+\s+at\s+\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s+rot\s+(\d+)\s+layer\s+([FB])"
)


@dataclass(slots=True)
class ParsedPrompt:
    width: float
    height: float
    placed: list[tuple[str, float, float, int, str]]  # ref, x, y, rot, layer


def parse_prompt(prompt: str) -> ParsedPrompt:
    m = OUTLINE_RE.search(prompt)
    width, height = (float(m.group(1)), float(m.group(2))) if m else (100.0, 100.0)
    placed: list[tuple[str, float, float, int, str]] = []
    in_block = False
    for line in prompt.splitlines():
        s = line.strip()
        if s == "<placed>":
            in_block = True
            continue
        if s == "</placed>":
            in_block = False
            continue
        if not in_block:
            continue
        mm = PLACED_LINE_RE.match(line)
        if mm:
            placed.append(
                (mm.group(1), float(mm.group(2)), float(mm.group(3)), int(mm.group(4)), mm.group(5))
            )
    return ParsedPrompt(width=width, height=height, placed=placed)


def parse_completion(text: str) -> tuple[float, float, int, str] | None:
    m = PLACE_RE.search(text)
    if not m:
        return None
    return (
        float(m.group("x")),
        float(m.group("y")),
        int(m.group("rot")) % 360,
        m.group("layer").upper(),
    )


def overlaps_existing(x: float, y: float, placed: list, half: float = 1.0) -> bool:
    for _, px, py, _, _ in placed:
        if abs(px - x) < 2 * half and abs(py - y) < 2 * half:
            return True
    return False


def in_outline(x: float, y: float, w: float, h: float) -> bool:
    return 0 <= x <= w and 0 <= y <= h


def baseline_random(parsed: ParsedPrompt, rng: random.Random) -> tuple[float, float, int, str]:
    return (rng.uniform(0, parsed.width), rng.uniform(0, parsed.height), 0, "F")


def _prompt_completion(ex: dict) -> tuple[str, str]:
    """Pull (prompt, completion) strings from an eval example.

    The HF DatasetDict produced by tokenize_dataset.py only carries a
    `messages` column (chat-format pairs). For backwards compatibility we
    also accept dicts that have raw `prompt`/`completion` keys."""
    if "prompt" in ex and "completion" in ex:
        return ex["prompt"], ex["completion"]
    msgs = ex["messages"]
    user_msg = next(m for m in msgs if m["role"] == "user")
    asst_msg = next(m for m in msgs if m["role"] == "assistant")
    return user_msg["content"], asst_msg["content"]


def baseline_centroid(prompt: str, target_ref: str, parsed: ParsedPrompt) -> tuple[float, float, int, str]:
    """Place at centre of mass of placed components that share a named net
    with the target. Falls back to outline centre."""
    nets_block = re.search(r"<netlist>(.*?)</netlist>", prompt, re.DOTALL)
    target_nets: set[str] = set()
    if nets_block:
        for line in nets_block.group(1).splitlines():
            line = line.strip()
            if not line.startswith("NET "):
                continue
            head, _, refs = line.partition(":")
            net = head[4:].strip()
            if any(r.strip().split(".")[0] == target_ref for r in refs.split(",")):
                target_nets.add(net)
    relevant_refs: set[str] = set()
    if nets_block and target_nets:
        for line in nets_block.group(1).splitlines():
            line = line.strip()
            if not line.startswith("NET "):
                continue
            head, _, refs = line.partition(":")
            net = head[4:].strip()
            if net not in target_nets:
                continue
            for r in refs.split(","):
                ref = r.strip().split(".")[0]
                if ref != target_ref:
                    relevant_refs.add(ref)
    xs = [px for ref, px, _, _, _ in parsed.placed if ref in relevant_refs]
    ys = [py for ref, _, py, _, _ in parsed.placed if ref in relevant_refs]
    if not xs:
        return (parsed.width / 2, parsed.height / 2, 0, "F")
    return (statistics.mean(xs), statistics.mean(ys), 0, "F")


def evaluate(
    examples: list[dict],
    predictions: list[str | None],
) -> dict:
    metrics = {
        "n": len(examples),
        "parsed": 0,
        "coord_l2_mm": [],
        "rot_correct": 0,
        "layer_correct": 0,
        "overlap": 0,
        "in_outline": 0,
    }
    for ex, pred_text in zip(examples, predictions):
        prompt_text, completion_text = _prompt_completion(ex)
        gt = parse_completion(completion_text)
        if gt is None:
            continue
        gx, gy, grot, glayer = gt
        parsed_prompt = parse_prompt(prompt_text)
        if pred_text is None:
            continue
        pred = parse_completion(pred_text)
        if pred is None:
            continue
        metrics["parsed"] += 1
        px, py, prot, player = pred
        metrics["coord_l2_mm"].append(math.hypot(px - gx, py - gy))
        metrics["rot_correct"] += int(prot == grot)
        metrics["layer_correct"] += int(player == glayer)
        metrics["overlap"] += int(overlaps_existing(px, py, parsed_prompt.placed))
        metrics["in_outline"] += int(in_outline(px, py, parsed_prompt.width, parsed_prompt.height))

    n_parsed = max(metrics["parsed"], 1)
    return {
        "n": metrics["n"],
        "parse_success_rate": metrics["parsed"] / max(metrics["n"], 1),
        "coord_l2_mm_median": statistics.median(metrics["coord_l2_mm"]) if metrics["coord_l2_mm"] else float("nan"),
        "coord_l2_mm_mean": statistics.mean(metrics["coord_l2_mm"]) if metrics["coord_l2_mm"] else float("nan"),
        "rot_acc": metrics["rot_correct"] / n_parsed,
        "layer_acc": metrics["layer_correct"] / n_parsed,
        "overlap_rate": metrics["overlap"] / n_parsed,
        "in_outline_rate": metrics["in_outline"] / n_parsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", type=Path, required=True, help="LoRA adapter dir")
    ap.add_argument("--dataset", type=Path, default=Path("data/hf"))
    ap.add_argument("--max-examples", type=int, default=200, help="cap eval size for speed")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("data/eval/results.json"))
    args = ap.parse_args()

    import torch  # type: ignore
    from datasets import load_from_disk  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm unavailable")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    ds = load_from_disk(str(args.dataset))
    eval_rows = list(ds["eval"])[: args.max_examples]
    print(f"evaluating {len(eval_rows)} examples")

    rng_baseline = random.Random(0xBEEF)

    predictions: list[str | None] = []
    baseline_random_preds: list[str] = []
    baseline_centroid_preds: list[str] = []

    for ex in tqdm(eval_rows, desc="generate"):
        prompt_text, completion_text = _prompt_completion(ex)
        # Build the chat prompt (user turn only) and let the model continue.
        messages = [{"role": "user", "content": prompt_text}]
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predictions.append(gen)

        # Baselines: synthesise a PLACE line directly.
        gt = parse_completion(completion_text)
        if gt is None:
            baseline_random_preds.append("")
            baseline_centroid_preds.append("")
            continue
        target_ref = re.search(r"PLACE\s+(\S+)", completion_text).group(1)
        parsed_prompt = parse_prompt(prompt_text)
        rx, ry, rrot, rlayer = baseline_random(parsed_prompt, rng_baseline)
        cx, cy, crot, clayer = baseline_centroid(prompt_text, target_ref, parsed_prompt)
        baseline_random_preds.append(f"PLACE {target_ref} at ({rx:.2f}, {ry:.2f}) rot {rrot} layer {rlayer}")
        baseline_centroid_preds.append(f"PLACE {target_ref} at ({cx:.2f}, {cy:.2f}) rot {crot} layer {clayer}")

    results = {
        "model": evaluate(eval_rows, predictions),
        "baseline_random": evaluate(eval_rows, baseline_random_preds),
        "baseline_centroid": evaluate(eval_rows, baseline_centroid_preds),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    print(orjson.dumps(results, option=orjson.OPT_INDENT_2).decode())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

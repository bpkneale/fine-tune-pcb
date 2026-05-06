"""Convert data/dsl/train.jsonl into a Hugging Face DatasetDict ready for
TRL's SFTTrainer.

  - Board-level train/eval split (never split the same board across sets).
  - Each example becomes a chat message list: user=prompt, assistant=completion.
  - Optionally pre-rendered with the Gemma 4 chat template into a `text`
    column (some trainers prefer this), keeping the raw `messages` column too.
  - Long examples (over `--max-tokens`) are dropped rather than truncated —
    truncating completions corrupts the supervision signal.

Run inside WSL2 after Phase 1 setup, since it imports `transformers`.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import orjson
from tqdm import tqdm

EVAL_FRACTION = 0.05
SEED = 0xC0FFEE


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("rb") as f:
        for line in f:
            try:
                out.append(orjson.loads(line))
            except Exception:
                continue
    return out


def _board_split(records: list[dict], eval_frac: float, seed: int) -> tuple[set[str], set[str]]:
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_sha[r["sha"]].append(r)
    shas = list(by_sha.keys())
    rng = random.Random(seed)
    rng.shuffle(shas)
    n_eval = max(1, int(len(shas) * eval_frac))
    eval_shas = set(shas[:n_eval])
    train_shas = set(shas[n_eval:])
    return train_shas, eval_shas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/dsl/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/hf"))
    ap.add_argument(
        "--model",
        default="google/gemma-4-E4B-it",
        help="HF model id whose chat template / tokenizer will be applied",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="drop examples longer than this after chat templating",
    )
    args = ap.parse_args()

    # Heavy imports kept inside main so the data pipeline stays usable on
    # Windows where transformers isn't installed.
    from datasets import Dataset, DatasetDict  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    records = _read_jsonl(args.inp)
    print(f"read {len(records)} examples")
    if not records:
        print("nothing to do")
        return 0

    train_shas, eval_shas = _board_split(records, EVAL_FRACTION, SEED)
    print(f"split: {len(train_shas)} train boards, {len(eval_shas)} eval boards")

    tok = AutoTokenizer.from_pretrained(args.model)

    def render(example: dict) -> dict | None:
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
        text = tok.apply_chat_template(messages, tokenize=False)
        # Length check on the tokenised version (fast on the cached BPE).
        n = len(tok(text, add_special_tokens=False).input_ids)
        if n > args.max_tokens:
            return None
        return {
            "sha": example["sha"],
            "step": example["step"],
            "messages": messages,
            "text": text,
            "n_tokens": n,
        }

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    dropped = 0
    for r in tqdm(records, desc="render"):
        rendered = render(r)
        if rendered is None:
            dropped += 1
            continue
        (train_rows if r["sha"] in train_shas else eval_rows).append(rendered)

    print(f"rendered: train={len(train_rows)}  eval={len(eval_rows)}  dropped(>max_tokens)={dropped}")

    ds = DatasetDict(
        train=Dataset.from_list(train_rows),
        eval=Dataset.from_list(eval_rows),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out))
    print(f"saved DatasetDict to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

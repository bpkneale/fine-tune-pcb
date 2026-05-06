# fine-tune-pcb

Fine-tune Gemma 4 E4B as a next-component-placement assistant for KiCad PCBs.

Plan: `~/.claude/plans/i-would-like-to-sprightly-riddle.md`.

## Status

Phase 2 — data pipeline. Train inside WSL2 + ROCm (Phase 4); inference in
LM Studio on Windows.

## Quick start

### Phase 2 — data pipeline (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 1. Scrape (needs GITHUB_TOKEN env var with public_repo scope)
python -m src.scrape_github --limit 1000 --out data/raw   # broad code search
python -m src.scrape_orgs                                 # curated orgs

# 2. Parse to structured JSON
python -m src.parse_kicad --in data/raw --out data/parsed/boards.jsonl

# 3. Build DSL training pairs (3 placement orderings per board → 3× examples)
python -m src.build_dsl --in data/parsed/boards.jsonl --out data/dsl/train.jsonl
```

### Phases 3–6 — train / eval / export

Two paths:

- **Kaggle (free T4 + QLoRA, recommended)**: see [`docs/kaggle.md`](docs/kaggle.md).
  Notebook at `notebooks/kaggle_train.ipynb`. Avoids the local ROCm
  yak-shaving and runs at full `seq_len 4096`.
- **Local WSL2 + ROCm**: instructions below. Memory-pressured on
  consumer AMD cards (Gemma's 256k vocab); uses `seq_len 1536`.

#### Local WSL2 + ROCm

One-shot setup (idempotent, installs ROCm + librocdxg + venv + torch + HF):

```bash
bash env/wsl2-setup.sh
```

Manual prereqs the script can't do (Windows side):
- AMD Adrenalin >= 26.2.2.
- WSL2 + Ubuntu 24.04 (`wsl --install -d Ubuntu-24.04`).

Then from the repo root inside WSL2:

```bash
bash scripts/e2e_smoke.sh           # smoke run: 1 epoch, ~tens of minutes
```

Or run each phase by hand:

```bash
python -m src.tokenize_dataset --in data/dsl/train.jsonl --out data/hf
python -m src.train_lora --dataset data/hf --output-dir adapters/v1
python -m src.eval_placement --adapter adapters/v1 --dataset data/hf
python -m src.export_gguf --adapter adapters/v1 --out adapters/v1-gguf \
    --llama-cpp ~/code/llama.cpp
```

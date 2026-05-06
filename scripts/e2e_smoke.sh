#!/usr/bin/env bash
# End-to-end smoke run with whatever data is currently in data/dsl/train.jsonl.
#
# Purpose: prove the WSL2 + ROCm + transformers + peft + trl pipeline works
# wire-to-wire on this machine. NOT a real training run — overrides epochs
# and eval cadence to finish in tens of minutes, and the resulting LoRA is
# deliberately undertrained.
#
# Run inside WSL2 from the repo root, with the ~/venvs/pcb venv activated
# (see env/wsl2-rocm-setup.md).

set -euo pipefail

OUT_DIR="${OUT_DIR:-adapters/smoke-v1}"
HF_DATA_DIR="${HF_DATA_DIR:-data/hf-smoke}"
EVAL_DIR="${EVAL_DIR:-data/eval/smoke}"
GGUF_DIR="${GGUF_DIR:-adapters/smoke-v1-gguf}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/code/llama.cpp}"

echo "=== 1. Tokenise dataset ==="
python -m src.tokenize_dataset \
    --in data/dsl/train.jsonl \
    --out "$HF_DATA_DIR" \
    --max-tokens 1536

echo "=== 2. Train (smoke config: 1 epoch, freq eval) ==="
python -m src.train_lora \
    --dataset "$HF_DATA_DIR" \
    --output-dir "$OUT_DIR" \
    --epochs 1 \
    --per-device-batch-size 1 \
    --grad-accum 8 \
    --seq-len 1536

echo "=== 3. Evaluate against baselines ==="
mkdir -p "$EVAL_DIR"
python -m src.eval_placement \
    --adapter "$OUT_DIR" \
    --dataset "$HF_DATA_DIR" \
    --max-examples 100 \
    --out "$EVAL_DIR/results.json"

echo "=== 4. Export merged GGUF (skip if llama.cpp not built) ==="
if [[ -x "$LLAMA_CPP/build/bin/llama-quantize" ]] || [[ -x "$LLAMA_CPP/llama-quantize" ]]; then
    python -m src.export_gguf \
        --adapter "$OUT_DIR" \
        --out "$GGUF_DIR" \
        --llama-cpp "$LLAMA_CPP" \
        --quants Q4_K_M
else
    echo "  (skipped — set LLAMA_CPP to a built llama.cpp checkout)"
fi

echo "=== done ==="
echo "Adapter:  $OUT_DIR"
echo "Eval:     $EVAL_DIR/results.json"
echo "GGUFs:    $GGUF_DIR/"

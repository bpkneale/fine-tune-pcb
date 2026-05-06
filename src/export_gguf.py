"""Merge a LoRA adapter into the base model and export GGUF for LM Studio.

Pipeline:
  1. Load base + adapter, call merge_and_unload() → bf16 HF model.
  2. Save merged model to <out>/merged-hf/.
  3. Shell out to llama.cpp's convert_hf_to_gguf.py to make an f16 GGUF.
  4. Quantise with llama-quantize to Q4_K_M and Q5_K_M.

Requires llama.cpp checked out and built somewhere — pass --llama-cpp.
On Windows, point --llama-cpp at the directory containing llama-quantize.exe
and convert_hf_to_gguf.py.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output dir; merged HF weights and GGUFs go here",
    )
    ap.add_argument(
        "--llama-cpp",
        type=Path,
        required=True,
        help="path to llama.cpp checkout (must contain convert_hf_to_gguf.py and built llama-quantize)",
    )
    ap.add_argument(
        "--quants",
        nargs="+",
        default=["Q4_K_M", "Q5_K_M"],
        help="GGUF quantisations to produce",
    )
    ap.add_argument("--skip-merge", action="store_true", help="reuse existing merged-hf/")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    merged_dir = args.out / "merged-hf"

    if not args.skip_merge:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        print("loading base model...")
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        print("loading adapter...")
        peft_model = PeftModel.from_pretrained(base, str(args.adapter))
        print("merging...")
        merged = peft_model.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        AutoTokenizer.from_pretrained(args.model).save_pretrained(str(merged_dir))
        print(f"saved merged model to {merged_dir}")

    convert_script = args.llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise SystemExit(f"missing {convert_script}")

    f16_gguf = args.out / "model-f16.gguf"
    _run(
        [
            sys.executable,
            str(convert_script),
            str(merged_dir),
            "--outfile",
            str(f16_gguf),
            "--outtype",
            "f16",
        ]
    )

    quantize_bin = args.llama_cpp / ("llama-quantize.exe" if sys.platform == "win32" else "llama-quantize")
    if not quantize_bin.exists():
        # Fallback common build path
        alt = args.llama_cpp / "build" / "bin" / quantize_bin.name
        if alt.exists():
            quantize_bin = alt
        else:
            raise SystemExit(f"missing {quantize_bin}; build llama.cpp first")

    for quant in args.quants:
        out_gguf = args.out / f"model-{quant}.gguf"
        _run([str(quantize_bin), str(f16_gguf), str(out_gguf), quant])

    # f16 GGUF is only an intermediate; remove unless user wants to keep it.
    if not (args.out / "KEEP_F16").exists():
        try:
            f16_gguf.unlink()
        except OSError:
            pass

    print(f"GGUFs ready in {args.out}")
    print("To use in LM Studio, copy a Q*.gguf into:")
    print(r"  %USERPROFILE%\.lmstudio\models\local\gemma4-e4b-pcb\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

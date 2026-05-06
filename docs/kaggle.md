# Train on Kaggle (single T4 + QLoRA)

The local WSL2 + ROCm path works but is memory-pressured: Gemma 4 E4B's
256k vocab forces `seq_len ≤ 1536` on a 24 GB card without QLoRA. Kaggle's
free T4 has native CUDA bitsandbytes, so QLoRA fits the model at the full
`seq_len 4096` and uses the full ~8.7k example dataset.

This is a runbook. Most of the steps are interactive (browser clicks); the
actual training is one notebook.

## One-time setup

### 1. Publish this repo to GitHub

```bash
git init
# Sanity check what's about to be committed:
git status
# Make sure you DO NOT see: .env, data/raw/, data/parsed/, data/dsl/,
# adapters/, *.gguf, data/*.log
git add .
git commit -m "Initial commit"

# Create the empty repo on github.com first, then:
git remote add origin git@github.com:<you>/fine-tune-pcb.git
git branch -M main
git push -u origin main
```

If you used a different repo name, update the `git clone` URL in cell 1
of `notebooks/kaggle_train.ipynb`.

### 2. Upload the dataset to Kaggle

1. https://www.kaggle.com → Datasets → New Dataset.
2. Drag in `data/dsl/train.jsonl`.
3. Title: `fine-tune-pcb-dsl`. Set **Privacy: Private**.
4. Create.

The dataset will mount inside notebooks at
`/kaggle/input/fine-tune-pcb-dsl/train.jsonl`.

### 3. Hugging Face setup

1. https://huggingface.co/settings/tokens — create a token with **read +
   write** scope. Write is needed to push the trained adapter back at the
   end.
2. https://huggingface.co/google/gemma-4-E4B-it — click **Accept terms**
   on the same HF account whose token you just generated.
3. Optional: https://huggingface.co/new — create an empty private model
   repo `<you>/gemma4-e4b-pcb-lora-v1`. The notebook will create it
   on-the-fly via `create_repo(..., exist_ok=True)`, but you can do it up
   front for clarity.

### 4. Create the Kaggle notebook

1. Kaggle → Code → New Notebook.
2. Settings panel:
   - **Accelerator**: GPU T4 ×1
   - **Internet**: on
   - **Add-ons** → **Secrets** → Add Secret → Label `HF_TOKEN`, value =
     your HF token from step 3.1.
   - **Add-ons** → **Datasets** → Attach → search for your
     `fine-tune-pcb-dsl` dataset.
3. Notebook cells: paste the contents of `notebooks/kaggle_train.ipynb`
   from this repo. (Alternatively, in cell 1 just `!git clone` the repo
   and have the notebook itself drive everything — but Kaggle's notebook
   editor is the path of least resistance.)

## Running

Hit **Run all**. Approximate cell timings:

| Cell | What | Time |
|---|---|---|
| 1 | git clone | 5 s |
| 2 | pip install | 1–2 min |
| 3 | HF auth | 5 s |
| 4 | tokenise | 30 s |
| 5 | **train** | **2–3 hr** |
| 6 | eval | 2–5 min |
| 7 | push adapter | 30 s |

Kaggle sessions cap at 9 hours, so we have plenty of headroom. Free GPU
quota is 30 hr/week — one full run ≈ 1/10 of weekly quota.

If cell 5 OOMs (unlikely with QLoRA): drop `--per-device-batch-size 2`
to `1` and bump `--grad-accum 8` to `16` to keep the effective batch
size constant.

If `liger-kernel` fails to load: append `--no-liger` to the train
command. QLoRA alone gives enough memory headroom on T4.

## Pulling the adapter back local

After cell 7 succeeds, on your local WSL2:

```bash
source ~/venvs/pcb/bin/activate
huggingface-cli download <you>/gemma4-e4b-pcb-lora-v1 \
    --local-dir adapters/v1
```

Then convert to GGUF for LM Studio (assuming you have llama.cpp checked
out and built):

```bash
python -m src.export_gguf \
    --adapter adapters/v1 \
    --out adapters/v1-gguf \
    --llama-cpp ~/code/llama.cpp \
    --quants Q4_K_M Q5_K_M
```

Drop the resulting `.gguf` into
`%USERPROFILE%\.lmstudio\models\local\gemma4-e4b-pcb\` on Windows and
LM Studio picks it up.

## Troubleshooting

- **`401 Client Error: Unauthorized` from HF in cell 5**: token is
  read-only or you didn't accept Gemma 4 terms. Fix at
  https://huggingface.co/settings/tokens and
  https://huggingface.co/google/gemma-4-E4B-it.
- **Cell 2 fails on `bitsandbytes`**: Kaggle's CUDA version may need a
  specific bnb pin. Try `bitsandbytes==0.43.3` then `0.44.x`.
- **Long wall time on cell 5**: T4 is roughly 2–3× slower than an A100;
  if you want faster iteration, rent an A100 on RunPod / vast.ai for
  ~$1.50/hour.
- **Adapter push fails partway**: rerun cell 7 — `upload_folder` is
  resumable and `create_repo` is idempotent with `exist_ok=True`.

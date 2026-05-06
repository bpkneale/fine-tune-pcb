"""LoRA fine-tune of Gemma 4 E4B on the PCB placement DSL.

Run inside WSL2 with the env from env/wsl2-setup.sh:

    python -m src.train_lora \
        --dataset data/hf \
        --output-dir adapters/gemma4-e4b-pcb-lora-v1

Auto-detects 4-bit (QLoRA) availability. If bitsandbytes is importable
training runs in 4-bit; otherwise it falls back to bf16 LoRA, which fits
a 4B model + LoRA on the 7900 XTX at seq len 4096.

Loss is masked to assistant turns only via SFTConfig(assistant_only_loss=True)
— the chat template's role tokens tell the trainer which spans to compute
loss over. (TRL >=0.13 dropped DataCollatorForCompletionOnlyLM for this.)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Short-circuit transformers' is_wandb_available() before TRL eagerly
# imports wandb at module-load time. On Kaggle the preinstalled wandb has
# proto stubs that explode against protobuf<7; we don't use wandb anyway
# (report_to=[] in SFTConfig).
os.environ.setdefault("WANDB_DISABLED", "true")


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


# Gemma chat template patched to include the `{% generation %}` markers TRL
# needs to mask loss to assistant turns only. Functionally identical to the
# stock Gemma template otherwise — same role names, same delimiters.
GEMMA_TRAINING_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<start_of_turn>' + role + '\n' }}"
    "{% if (message['role'] == 'assistant') %}"
    "{% generation %}{{ message['content'] | trim + '<end_of_turn>\n' }}{% endgeneration %}"
    "{% else %}"
    "{{ message['content'] | trim + '<end_of_turn>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--dataset", type=Path, default=Path("data/hf"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--per-device-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument(
        "--no-4bit",
        action="store_true",
        help="force bf16 LoRA even if bitsandbytes loads",
    )
    ap.add_argument(
        "--no-liger",
        action="store_true",
        help="disable Liger Kernel even if importable (slower + more memory)",
    )
    args = ap.parse_args()

    import torch  # type: ignore
    from datasets import load_from_disk  # type: ignore
    from peft import LoraConfig, prepare_model_for_kbit_training  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from trl import SFTConfig, SFTTrainer  # type: ignore

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is False — ROCm/torch setup is wrong.")
    print(f"device: {torch.cuda.get_device_name(0)}")

    use_4bit = (not args.no_4bit) and _bnb_available()
    print(f"4-bit (QLoRA): {use_4bit}")

    # Liger Kernel: fused chunked cross-entropy + RMSNorm/RoPE kernels.
    # Critical for Gemma's 256k vocab — without it, the [seq, vocab] logit
    # tensor at training time blows out 24GB at seq_len > ~1k.
    use_liger = False
    if not args.no_liger:
        try:
            import liger_kernel  # noqa: F401
            use_liger = True
        except Exception:
            pass
    print(f"Liger Kernel: {use_liger}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = GEMMA_TRAINING_CHAT_TEMPLATE

    model_kwargs: dict = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",  # flash-attn isn't reliable on ROCm
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig  # type: ignore

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required when gradient checkpointing is on

    # Gemma 4 wraps each nn.Linear in Gemma4ClippableLinear (a clipping
    # wrapper that's incompatible with PEFT's LoRA injection). Target the
    # inner .linear instead via a regex on the full module path.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.linear$",
    )

    ds = load_from_disk(str(args.dataset))
    print(f"dataset: train={len(ds['train'])}  eval={len(ds['eval'])}")

    effective_batch = args.per_device_batch_size * args.grad_accum
    total_steps = max(1, int((len(ds["train"]) / effective_batch) * args.epochs))
    warmup_steps = max(1, int(total_steps * 0.03))

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        max_length=args.seq_len,        # was max_seq_length in older TRL
        packing=False,
        assistant_only_loss=True,       # mask loss to assistant turns
        use_liger_kernel=use_liger,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        report_to=[],
        seed=0,
    )

    # Newer TRL takes `processing_class=` and works with the `messages`
    # column directly; we drop the redundant `text` column to be explicit.
    train_ds = ds["train"].remove_columns(
        [c for c in ds["train"].column_names if c not in {"messages"}]
    )
    eval_ds = ds["eval"].remove_columns(
        [c for c in ds["eval"].column_names if c not in {"messages"}]
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tok.save_pretrained(str(args.output_dir))
    print(f"saved adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

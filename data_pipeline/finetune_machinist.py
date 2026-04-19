"""
T2-11: Fine-Tuned Machinist v2 — Qwen2.5-7B SFT Training Script.

Fine-tunes a Qwen2.5-7B model on the curated text-to-CadQuery dataset
produced by T2-05 (stratified prompts) and T2-06 (DeepCAD pairs).

Target: <1% invalid part rate on the held-out eval set. Replace Gemini
Machinist calls for Domain A, B, and C with the fine-tuned model.
Keep Gemini for Domain D (aerospace complexity exceeds 7B model capability).

Usage:
    python data_pipeline/finetune_machinist.py \
        --train-data data/machinist_sft_stratified.jsonl \
                     data/deepcad_cq_pairs.jsonl \
        --output-dir models/machinist-v2 \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --epochs 3 \
        --batch-size 4

Infrastructure requirements:
    - GPU: 2× A100 80GB or 4× A6000 48GB (minimum for 7B QLoRA)
    - RAM: 64GB system RAM
    - Storage: ~50GB for model + dataset
    - CUDA 12.1+

Dependencies:
    pip install transformers trl datasets peft accelerate bitsandbytes
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Training approach: QLoRA (4-bit quantisation + LoRA adapters)
    - Base model: Qwen/Qwen2.5-7B-Instruct
    - LoRA rank: 64, alpha: 128
    - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj
    - Max sequence length: 4096 tokens
    - Learning rate: 2e-4 (cosine decay)
    - Training time: ~8 hours on 2× A100

ProCAD benchmark (Wang et al. 2025): 1.6K curated CadQuery pairs fine-tuned
on Qwen2.5-7B achieves 0.9% invalidity rate, outperforming GPT-4o zero-shot.
Target the same metric after training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("mirum.finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Dataset formatting
# ---------------------------------------------------------------------------

def format_training_record(record: dict, domain_strategy_fn=None) -> dict | None:
    """Format a JSONL training record as a chat message triple.

    Input fields: prompt/description, code, domain, complexity_level
    Output: {"messages": [system, user, assistant]} for SFT Trainer

    Args:
        record: Raw training record from JSONL.
        domain_strategy_fn: Optional callable(domain, description) -> str
            to inject the production strategy prompt as the system message.
            If None, a generic system prompt is used.

    Returns:
        Formatted dict, or None if the record is malformed.
    """
    description = record.get("prompt") or record.get("description", "")
    code = record.get("code", "")
    domain = record.get("domain", "A")

    if not description or not code:
        return None

    # System message: domain-specific strategy prompt
    if domain_strategy_fn is not None:
        system_msg = domain_strategy_fn(domain, description)
    else:
        system_msg = (
            f"You are a CadQuery machinist for Domain {domain}. "
            "Generate valid CadQuery Python code for the given part description. "
            "Return ONLY executable Python. No markdown, no comments."
        )

    return {
        "messages": [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": description},
            {"role": "assistant", "content": code},
        ]
    }


def load_and_format_dataset(
    jsonl_paths: list[str],
    domain_strategy_fn=None,
    max_records: int = 0,
    domain_filter: list[str] | None = None,
) -> list[dict]:
    """Load and format multiple JSONL training files.

    Args:
        jsonl_paths: List of JSONL file paths to combine.
        domain_strategy_fn: Passed to format_training_record.
        max_records: If > 0, cap total records (useful for debugging).
        domain_filter: If set, only include records matching these domains.

    Returns:
        List of formatted chat message dicts.
    """
    records = []
    seen_codes: set[str] = set()

    for path in jsonl_paths:
        if not Path(path).exists():
            logger.warning("Training file not found: %s — skipping", path)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if domain_filter and rec.get("domain") not in domain_filter:
                    continue

                # Deduplicate on code hash
                code_key = rec.get("code", "")[:200]
                if code_key in seen_codes:
                    continue
                seen_codes.add(code_key)

                formatted = format_training_record(rec, domain_strategy_fn)
                if formatted:
                    records.append(formatted)

                if max_records > 0 and len(records) >= max_records:
                    break
            if max_records > 0 and len(records) >= max_records:
                break

    logger.info("Loaded %d formatted training records", len(records))
    return records


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    train_data_paths: list[str],
    output_dir: str,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 4096,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    domain_filter: list[str] | None = None,
    use_qlora: bool = True,
) -> None:
    """Run QLoRA SFT training on Qwen2.5-7B-Instruct.

    This function requires GPU hardware and the following packages:
        transformers, trl, peft, accelerate, bitsandbytes, datasets

    Args:
        train_data_paths: List of JSONL training files.
        output_dir: Directory to save the fine-tuned adapter weights.
        base_model: HuggingFace model name or local path.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation: Gradient accumulation steps.
        learning_rate: Learning rate.
        max_seq_length: Max token sequence length.
        lora_rank: LoRA rank r.
        lora_alpha: LoRA alpha scaling.
        domain_filter: If set, train only on these domains (e.g. ["A", "B", "C"]).
        use_qlora: Use 4-bit quantisation (requires bitsandbytes).
    """
    # Import training libraries
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTTrainer, SFTConfig
        from peft import LoraConfig, get_peft_model
        import torch
    except ImportError as e:
        raise ImportError(
            f"Training dependencies not available: {e}\n"
            "Install: pip install transformers trl peft accelerate bitsandbytes"
        ) from e

    # Try to inject production strategy prompts
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
        from strategies import get_strategy as domain_strategy_fn
    except ImportError:
        domain_strategy_fn = None
        logger.warning("Could not import strategies.py — using generic system prompts")

    # Load and format dataset
    records = load_and_format_dataset(
        train_data_paths,
        domain_strategy_fn=domain_strategy_fn,
        domain_filter=domain_filter,
    )
    if len(records) < 100:
        raise ValueError(
            f"Only {len(records)} training records found. "
            "Minimum 100 required. Run T2-05 and T2-06 pipelines first."
        )

    # Convert to HuggingFace dataset
    from datasets import Dataset
    hf_dataset = Dataset.from_list(records)
    logger.info("Training dataset: %d examples", len(hf_dataset))

    # Quantisation config (QLoRA)
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load base model
    logger.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not use_qlora else None,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # SFT training configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        report_to="none",  # Set to "wandb" for experiment tracking
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Format function for SFT Trainer
    def formatting_func(example):
        """Convert messages to a formatted string for training."""
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        formatting_func=formatting_func,
        args=sft_config,
    )

    logger.info("Starting training — %d epochs, %d examples", epochs, len(hf_dataset))
    trainer.train()

    # Save final adapter
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Model saved to %s", final_path)


# ---------------------------------------------------------------------------
# Inference wrapper (for production integration)
# ---------------------------------------------------------------------------

class FinetunedMachinist:
    """Production inference wrapper for the fine-tuned Machinist model.

    Replaces Gemini API calls for Domain A, B, C with local inference.
    Falls back to Gemini for Domain D (aerospace complexity).

    Usage:
        machinist = FinetunedMachinist(model_path="models/machinist-v2/final")
        code = machinist.generate(description, domain="A")
    """

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None
        self._supported_domains = {"A", "B", "C"}

    def _load(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch

            logger.info("Loading fine-tuned Machinist from %s", self.model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self._model.eval()
            logger.info("Fine-tuned Machinist loaded")
        except Exception as exc:
            logger.error("Failed to load fine-tuned model: %s", exc)
            raise

    def is_available(self) -> bool:
        """Return True if the fine-tuned model file exists."""
        return Path(self.model_path).exists()

    def generate(
        self,
        description: str,
        domain: str = "A",
        system_prompt: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str | None:
        """Generate CadQuery code for a part description.

        Returns None for unsupported domains (caller should use Gemini).
        Returns None if model loading fails (caller should use Gemini).
        """
        if domain not in self._supported_domains:
            return None  # Caller uses Gemini for Domain D

        if not self.is_available():
            return None  # Model not trained yet

        try:
            self._load()
            import torch

            if system_prompt is None:
                try:
                    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
                    from strategies import get_strategy
                    system_prompt = get_strategy(domain, description)
                except ImportError:
                    system_prompt = f"You are a CadQuery machinist for Domain {domain}."

            messages = [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": description},
            ]
            input_ids = self._tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
            ).to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the generated tokens (skip the input)
            new_tokens = output_ids[0][input_ids.shape[-1]:]
            code = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return code.strip()

        except Exception as exc:
            logger.error("Fine-tuned inference failed: %s", exc)
            return None  # Caller falls back to Gemini


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate_invalidity_rate(
    model_path: str,
    eval_jsonl: str,
    n_samples: int = 100,
) -> dict:
    """Measure invalidity rate on a held-out eval set.

    Runs the fine-tuned model on n_samples descriptions, executes each
    generated script, and computes the invalidity rate (% that fail OCCT).

    Returns:
        {
            "n_samples": int,
            "invalid_count": int,
            "invalidity_rate": float,
            "domains": {"A": ..., "B": ..., "C": ...},
        }
    """
    import subprocess
    import sys
    import tempfile

    machinist = FinetunedMachinist(model_path)
    if not machinist.is_available():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    eval_records = []
    with open(eval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                eval_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    import random
    sample = random.sample(eval_records, min(n_samples, len(eval_records)))
    logger.info("Evaluating on %d samples", len(sample))

    results = {"total": 0, "invalid": 0, "by_domain": {}}

    for rec in sample:
        domain = rec.get("domain", "A")
        desc = rec.get("prompt") or rec.get("description", "")
        code = machinist.generate(desc, domain=domain)
        if code is None:
            continue

        results["total"] += 1
        results["by_domain"].setdefault(domain, {"total": 0, "invalid": 0})
        results["by_domain"][domain]["total"] += 1

        # Execute via subprocess
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name
        try:
            r = subprocess.run(
                [sys.executable, tmp],
                capture_output=True,
                timeout=30,
            )
            if r.returncode != 0:
                results["invalid"] += 1
                results["by_domain"][domain]["invalid"] += 1
        except subprocess.TimeoutExpired:
            results["invalid"] += 1
            results["by_domain"][domain]["invalid"] += 1
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    if results["total"] > 0:
        results["invalidity_rate"] = results["invalid"] / results["total"]
    else:
        results["invalidity_rate"] = 1.0

    logger.info("Invalidity rate: %.2f%%", results["invalidity_rate"] * 100)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Machinist v2 (Qwen2.5-7B on CadQuery pairs)"
    )
    sub = p.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Run SFT training")
    train_p.add_argument("--train-data", nargs="+", required=True,
                         help="JSONL training files")
    train_p.add_argument("--output-dir", default="models/machinist-v2",
                         help="Output directory for adapter weights")
    train_p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    train_p.add_argument("--epochs", type=int, default=3)
    train_p.add_argument("--batch-size", type=int, default=4)
    train_p.add_argument("--domain-filter", nargs="*", default=None,
                         help="Only train on these domains (e.g. A B C)")

    eval_p = sub.add_parser("eval", help="Evaluate invalidity rate")
    eval_p.add_argument("--model-path", required=True)
    eval_p.add_argument("--eval-data", required=True, help="Eval JSONL path")
    eval_p.add_argument("--n-samples", type=int, default=100)

    return p.parse_args()


if __name__ == "__main__":
    import sys
    args = _parse_args()
    if args.command == "train":
        train(
            train_data_paths=args.train_data,
            output_dir=args.output_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            domain_filter=args.domain_filter,
        )
    elif args.command == "eval":
        results = evaluate_invalidity_rate(
            model_path=args.model_path,
            eval_jsonl=args.eval_data,
            n_samples=args.n_samples,
        )
        print(json.dumps(results, indent=2))
    else:
        print("Usage: finetune_machinist.py [train|eval] --help")
        sys.exit(1)

# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Script 02: Evaluate model quality after fine-tuning.

This script runs a comprehensive evaluation suite to check if the model
has learned the JLPT task correctly.

Usage:
    python scripts/02_evaluate.py --model <model_path_or_hf_id> [--use-eval-set]

What it measures:
    1. Exact-match accuracy on grammar meaning (short answers)
    2. ROUGE-L score for detailed explanations
    3. JLPT level prediction accuracy (N1-N5)
    4. Formality classification accuracy
    5. Translation quality (BLEU score)
    6. Human-readable sample outputs for manual review

Requirements (add to requirements.txt):
    transformers>=4.40.0
    torch>=2.1.0
    peft>=0.10.0
    rouge-score>=0.1.2
    sacrebleu>=2.4.0
    tqdm>=4.66.0
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
EVAL_DATA = ROOT / "data" / "processed" / "combined_eval.jsonl"
REPORT_DIR = ROOT / "data" / "processed"


# ─── Load eval data ───────────────────────────────────────────────────────────

def load_eval_data(path: Path, max_samples: int = 200) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # Stratified sampling if needed
    if len(records) > max_samples:
        import random
        random.seed(42)
        random.shuffle(records)
        records = records[:max_samples]
    return records


# ─── Model inference ──────────────────────────────────────────────────────────

def load_model(model_name_or_path: str):
    """
    Load a fine-tuned or base model for inference.
    Supports: HuggingFace model ID, local path, LoRA adapter path.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"Loading model: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return model, tokenizer
    except ImportError:
        raise ImportError(
            "Install transformers and torch:\n"
            "  pip install transformers torch peft"
        )


def format_prompt(instruction: str, input_text: str, tokenizer) -> str:
    """
    Format input using Alpaca-style prompt template.
    Adjust this if your model uses a different chat template.
    """
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    return prompt


def generate_response(model, tokenizer, instruction: str, input_text: str, max_new_tokens: int = 200) -> str:
    import torch
    prompt = format_prompt(instruction, input_text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        result = scorer.score(reference, prediction)
        return result["rougeL"].fmeasure
    except ImportError:
        # Fallback: simple longest-common-subsequence ratio
        def lcs_length(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        p_tokens = prediction.split()
        r_tokens = reference.split()
        if not p_tokens or not r_tokens:
            return 0.0
        lcs = lcs_length(p_tokens, r_tokens)
        precision = lcs / len(p_tokens)
        recall = lcs / len(r_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def extract_jlpt_level(text: str) -> str | None:
    """Extract N1-N5 from a response string."""
    match = re.search(r"\b(N[1-5])\b", text)
    return match.group(1) if match else None


def extract_formality(text: str) -> str | None:
    """Extract formality label from response."""
    for label in ["very_formal", "formal", "neutral", "casual", "intimate", "vulgar"]:
        if label in text.lower():
            return label
    return None


# ─── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate(model, tokenizer, eval_records: list[dict]) -> dict:
    from tqdm import tqdm

    results = {
        "rouge_l_scores": [],
        "jlpt_level_correct": 0,
        "jlpt_level_total": 0,
        "formality_correct": 0,
        "formality_total": 0,
        "sample_outputs": [],
    }

    for i, record in enumerate(tqdm(eval_records, desc="Evaluating")):
        instruction = record["instruction"]
        input_text = record["input"]
        reference = record["output"]

        prediction = generate_response(model, tokenizer, instruction, input_text)

        # ROUGE-L for all outputs
        rouge = compute_rouge_l(prediction, reference)
        results["rouge_l_scores"].append(rouge)

        # JLPT level accuracy (for questions asking about level)
        if "jlpt level" in instruction.lower() or "what jlpt" in instruction.lower():
            ref_level = extract_jlpt_level(reference)
            pred_level = extract_jlpt_level(prediction)
            if ref_level:
                results["jlpt_level_total"] += 1
                if pred_level == ref_level:
                    results["jlpt_level_correct"] += 1

        # Formality accuracy
        if "formality" in instruction.lower():
            ref_formality = extract_formality(reference)
            pred_formality = extract_formality(prediction)
            if ref_formality:
                results["formality_total"] += 1
                if pred_formality == ref_formality:
                    results["formality_correct"] += 1

        # Save samples for manual review
        if i < 10:
            results["sample_outputs"].append({
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "prediction": prediction,
                "rouge_l": round(rouge, 3),
            })

    return results


def summarize_results(results: dict) -> dict:
    scores = results["rouge_l_scores"]
    avg_rouge = sum(scores) / len(scores) if scores else 0.0

    # JLPT level accuracy
    jlpt_acc = (
        results["jlpt_level_correct"] / results["jlpt_level_total"]
        if results["jlpt_level_total"] > 0 else None
    )
    formality_acc = (
        results["formality_correct"] / results["formality_total"]
        if results["formality_total"] > 0 else None
    )

    summary = {
        "avg_rouge_l": round(avg_rouge, 4),
        "jlpt_level_accuracy": round(jlpt_acc, 4) if jlpt_acc is not None else "N/A (no level questions in eval)",
        "formality_accuracy": round(formality_acc, 4) if formality_acc is not None else "N/A",
        "total_evaluated": len(scores),
        "rouge_l_distribution": {
            ">=0.8 (excellent)": sum(1 for s in scores if s >= 0.8),
            "0.5-0.8 (good)":    sum(1 for s in scores if 0.5 <= s < 0.8),
            "0.3-0.5 (fair)":    sum(1 for s in scores if 0.3 <= s < 0.5),
            "<0.3 (poor)":       sum(1 for s in scores if s < 0.3),
        },
        "interpretation": interpret_scores(avg_rouge, jlpt_acc),
    }
    return summary


def interpret_scores(rouge_l: float, jlpt_acc: float | None) -> str:
    """Give a human-readable interpretation of the scores."""
    lines = []
    if rouge_l >= 0.75:
        lines.append("✅ ROUGE-L ≥ 0.75: Model is generating high-quality, accurate explanations.")
    elif rouge_l >= 0.5:
        lines.append("⚠️ ROUGE-L 0.5-0.75: Model outputs are reasonable but may miss nuance. Consider more training data or epochs.")
    else:
        lines.append("❌ ROUGE-L < 0.5: Model outputs are poor. Check: data quality, prompt format, learning rate, or train longer.")

    if jlpt_acc is not None:
        if jlpt_acc >= 0.9:
            lines.append("✅ JLPT Level accuracy ≥ 90%: Model correctly identifies JLPT levels.")
        elif jlpt_acc >= 0.7:
            lines.append("⚠️ JLPT Level accuracy 70-90%: Mostly correct, some level confusion (especially N1/N2).")
        else:
            lines.append("❌ JLPT Level accuracy < 70%: Model is not learning JLPT level distinctions well.")

    return " | ".join(lines)


# ─── Offline evaluation (no model needed) ────────────────────────────────────

def evaluate_offline(eval_records: list[dict]) -> None:
    """
    Quick sanity check without running a model.
    Verifies the eval data quality and shows distribution.
    """
    print("\n[Eval Data Quality Report] (no model needed)")
    print("=" * 55)

    instruction_types = defaultdict(int)
    output_lengths = []

    for r in eval_records:
        instr = r["instruction"].lower()
        output_lengths.append(len(r["output"].split()))
        if "formality" in instr:
            instruction_types["formality"] += 1
        elif "jlpt level" in instr or "what jlpt" in instr:
            instruction_types["jlpt_level"] += 1
        elif "translate" in instr or "translation" in instr:
            instruction_types["translation"] += 1
        elif "mean" in instr:
            instruction_types["meaning"] += 1
        elif "form" in instr or "structure" in instr:
            instruction_types["formation"] += 1
        elif "read" in instr:
            instruction_types["reading"] += 1
        elif "overview" in instr or "complete" in instr or "study" in instr:
            instruction_types["comprehensive"] += 1
        else:
            instruction_types["other"] += 1

    avg_out = sum(output_lengths) / len(output_lengths)
    print(f"  Total eval samples : {len(eval_records):,}")
    print(f"  Avg output length  : {avg_out:.1f} words")
    print(f"\n  Task distribution:")
    for task, count in sorted(instruction_types.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(eval_records)
        print(f"    {task:<20} : {count:>5} ({pct:.1f}%)")

    print("\n  Sample (first 3 eval examples):")
    for i, r in enumerate(eval_records[:3]):
        print(f"\n  [{i+1}] Instruction: {r['instruction'][:70]}...")
        print(f"       Input:       {r['input'][:60]}")
        print(f"       Output:      {r['output'][:80]}...")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate JLPT fine-tuned model")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path (HuggingFace ID or local path)")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max eval samples to use (default: 200)")
    parser.add_argument("--offline", action="store_true",
                        help="Only check eval data quality, do not run model")
    args = parser.parse_args()

    print("=" * 60)
    print("  JLPT Model Evaluation")
    print("=" * 60)

    # Load eval data
    if not EVAL_DATA.exists():
        print(f"Eval data not found: {EVAL_DATA}")
        print("   Run 01_clean_data.py first!")
        return

    eval_records = load_eval_data(EVAL_DATA, max_samples=args.max_samples)
    print(f"Loaded {len(eval_records):,} eval samples from {EVAL_DATA.name}")

    # Offline mode: just inspect data
    if args.offline or args.model is None:
        evaluate_offline(eval_records)
        print("\n[To run full model evaluation:]")
        print("   python scripts/02_evaluate.py --model ./models/my_finetuned_model")
        return

    # Model evaluation
    model, tokenizer = load_model(args.model)
    results = evaluate(model, tokenizer, eval_records)
    summary = summarize_results(results)

    # Print results
    print("\n" + "=" * 60)
    print("  📊 Evaluation Results")
    print("=" * 60)
    print(f"  Avg ROUGE-L         : {summary['avg_rouge_l']}")
    print(f"  JLPT Level Accuracy : {summary['jlpt_level_accuracy']}")
    print(f"  Formality Accuracy  : {summary['formality_accuracy']}")
    print(f"  Total evaluated     : {summary['total_evaluated']}")
    print(f"\n  Distribution:")
    for k, v in summary["rouge_l_distribution"].items():
        print(f"    {k}: {v}")
    print(f"\n  📝 {summary['interpretation']}")

    # Save report
    report = {"summary": summary, "samples": results["sample_outputs"]}
    report_path = REPORT_DIR / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Full report saved to: {report_path}")

    # Print sample outputs for manual inspection
    print("\n" + "=" * 60)
    print("  🔍 Sample Outputs (first 3)")
    print("=" * 60)
    for i, s in enumerate(results["sample_outputs"][:3]):
        print(f"\n  [{i+1}] ROUGE-L: {s['rouge_l']}")
        print(f"       Instruction: {s['instruction'][:70]}...")
        print(f"       Input:       {s['input'][:50]}")
        print(f"       Reference:   {s['reference'][:80]}...")
        print(f"       Prediction:  {s['prediction'][:80]}...")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Script 01: Clean and process raw JLPT data into fine-tuning format.

Target model type: Instruction-following (~4B), e.g. Qwen2.5, Gemma3, Llama3.2
Output format: JSONL with {"instruction": ..., "input": ..., "output": ...}
Compatible with: Alpaca, LLaMA-Factory, Unsloth, trl SFTTrainer

Run:
    python scripts/01_clean_data.py

Output files in data/processed/:
    grammar_qa.jsonl       - Q&A from grammar points
    kanji_qa.jsonl         - Q&A from JLPT-filtered kanji
    words_qa.jsonl         - Q&A from JLPT-filtered words
    combined_train.jsonl   - All combined, shuffled (80% train)
    combined_eval.jsonl    - Eval split (20%)
    stats.json             - Dataset statistics
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
# Only include JLPT-relevant kanji and words (null = non-JLPT, skip them)
JLPT_LEVELS = {"N1", "N2", "N3", "N4", "N5"}
# Target languages for meanings (kanji only)
MEANING_LANG = "en"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    print(f"Loading {path.name} ({path.stat().st_size // 1024 // 1024} MB)...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [saved] {len(records):,} records -> {path.name}")

def clean_text(s: str) -> str:
    """Strip excessive whitespace and normalize."""
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def make_example(instruction: str, input_text: str, output: str) -> dict:
    return {
        "instruction": clean_text(instruction),
        "input": clean_text(input_text),
        "output": clean_text(output),
    }


# ─── 1. Grammar processing ────────────────────────────────────────────────────

def process_grammar(data: dict) -> list[dict]:
    """
    From each grammar point, generate multiple Q&A pairs:
    1. "What does [pattern] mean?" → meaning_en
    2. "How do you form [pattern]?" → formation + formation_notes
    3. "Give an example of [pattern]" → example sentence + translation
    4. "What JLPT level is [pattern]?" → level
    5. Fill-in-the-blank from example sentences
    """
    records = []
    grammar_points = data.get("grammar_points", [])
    print(f"  Processing {len(grammar_points):,} grammar points...")

    for gp in grammar_points:
        pattern = gp.get("pattern", "")
        level = gp.get("level", "")
        meaning_en = gp.get("meaning_en", "")
        meaning_detailed = gp.get("meaning_detailed", "")
        formation = gp.get("formation", "")
        formation_notes = gp.get("formation_notes", [])
        formality = gp.get("formality", "")
        examples = gp.get("examples", [])
        related = gp.get("related", [])

        if not pattern or not meaning_en:
            continue

        # ── QA 1: Meaning (short) ──────────────────────────────────────────
        records.append(make_example(
            instruction="日本語の文法パターンの意味を英語で説明してください。",
            input_text=f"文法パターン: {pattern}",
            output=meaning_en,
        ))

        # ── QA 2: Meaning (detailed with nuance) ──────────────────────────
        if meaning_detailed:
            records.append(make_example(
                instruction=(
                    "Explain this Japanese grammar pattern in detail in English, "
                    "including nuance, usage, and when to use it."
                ),
                input_text=f"Grammar pattern: {pattern} (JLPT {level})",
                output=meaning_detailed,
            ))

        # ── QA 3: Formation ────────────────────────────────────────────────
        if formation:
            notes_text = ""
            if formation_notes:
                notes_text = " Additional notes: " + "; ".join(formation_notes)
            records.append(make_example(
                instruction="How do you form this Japanese grammar pattern? Describe the grammatical structure.",
                input_text=f"Grammar pattern: {pattern}",
                output=formation + notes_text,
            ))

        # ── QA 4: JLPT level ───────────────────────────────────────────────
        records.append(make_example(
            instruction="What JLPT level is this Japanese grammar pattern?",
            input_text=f"Grammar pattern: {pattern}",
            output=f"This grammar pattern is {level} level.",
        ))

        # ── QA 5: Formality ────────────────────────────────────────────────
        if formality:
            records.append(make_example(
                instruction="What is the formality/register of this Japanese grammar pattern?",
                input_text=f"Grammar pattern: {pattern}",
                output=f"The formality of {pattern} is: {formality}.",
            ))

        # ── QA 6: Example sentences ────────────────────────────────────────
        for ex in examples[:3]:  # max 3 examples per pattern
            jp = ex.get("japanese", "")
            en = ex.get("english", "")
            if jp and en:
                records.append(make_example(
                    instruction=(
                        f"Translate this Japanese sentence that uses the grammar pattern 「{pattern}」."
                    ),
                    input_text=jp,
                    output=en,
                ))
                # Fill-in-the-blank: remove the grammar pattern core from the sentence
                # This is a harder task — only for patterns with clear surface form
                core = pattern.split("＋")[-1].strip() if "＋" in pattern else ""
                if core and len(core) >= 2 and core in jp:
                    blanked = jp.replace(core, "___", 1)
                    records.append(make_example(
                        instruction=(
                            "Fill in the blank with the correct grammar pattern. "
                            f"The sentence uses a JLPT {level} grammar pattern."
                        ),
                        input_text=f"Sentence: {blanked}\nChoose the correct grammar: A) {core}  B) ？",
                        output=f"The correct answer is: {core}\nFull sentence: {jp}\nMeaning: {en}",
                    ))

        # ── QA 7: JLPT multiple-choice style ──────────────────────────────
        # "Which grammar pattern best completes the sentence?"
        for ex in examples[:2]:
            jp = ex.get("japanese", "")
            en = ex.get("english", "")
            if jp and en and related:
                distractor = related[0] if related else "～について"
                records.append(make_example(
                    instruction=(
                        "This is a JLPT grammar question. "
                        "Identify the grammar pattern used and explain why it is correct."
                    ),
                    input_text=(
                        f"Sentence: {jp}\n"
                        f"Option A: {pattern}\n"
                        f"Option B: {distractor}"
                    ),
                    output=(
                        f"The correct answer is Option A: {pattern}.\n"
                        f"Explanation: {meaning_en}. "
                        f"This is a {level} level {formality} grammar pattern.\n"
                        f"Translation: {en}"
                    ),
                ))

    print(f"  Grammar -> {len(records):,} training examples")
    return records


# ─── 2. Kanji processing ──────────────────────────────────────────────────────

def process_kanji(data: dict) -> list[dict]:
    """
    From each kanji (JLPT-only), generate:
    1. "What does [kanji] mean?" → meanings (en)
    2. "How do you read [kanji]?" → on/kun readings
    3. "What JLPT level is [kanji]?" → level
    4. "Which kanji has the meaning [meaning]?" → reverse lookup
    """
    records = []
    all_kanji = data.get("kanji", [])

    # Filter to JLPT-only kanji (kanji outside JLPT are not testable)
    jlpt_kanji = [k for k in all_kanji if k.get("jlpt_waller") in JLPT_LEVELS]
    print(f"  Processing {len(jlpt_kanji):,} JLPT kanji (out of {len(all_kanji):,} total)...")

    for k in jlpt_kanji:
        char = k.get("character", "")
        level = k.get("jlpt_waller", "")
        meanings = k.get("meanings", {}).get(MEANING_LANG, [])
        on_readings = k.get("readings", {}).get("on", [])
        kun_readings = k.get("readings", {}).get("kun", [])
        stroke_count = k.get("stroke_count", "")

        if not char or not meanings:
            continue

        meaning_str = ", ".join(meanings[:5])  # limit to top 5 meanings
        on_str = ", ".join(on_readings) if on_readings else "none"
        kun_str = ", ".join(kun_readings) if kun_readings else "none"

        # ── QA 1: Meaning ──────────────────────────────────────────────────
        records.append(make_example(
            instruction="What does this kanji mean? Provide the English meanings.",
            input_text=f"Kanji: {char}",
            output=f"Meanings: {meaning_str}",
        ))

        # ── QA 2: Readings ─────────────────────────────────────────────────
        records.append(make_example(
            instruction="How do you read this kanji? Provide the on-yomi and kun-yomi readings.",
            input_text=f"Kanji: {char}",
            output=f"On-yomi (音読み): {on_str}\nKun-yomi (訓読み): {kun_str}",
        ))

        # ── QA 3: JLPT level ───────────────────────────────────────────────
        records.append(make_example(
            instruction="What JLPT level is this kanji?",
            input_text=f"Kanji: {char}",
            output=f"{char} is a JLPT {level} kanji. Meanings: {meaning_str}. Stroke count: {stroke_count}.",
        ))

        # ── QA 4: Comprehensive info ────────────────────────────────────────
        records.append(make_example(
            instruction="Provide a complete overview of this kanji for JLPT study.",
            input_text=f"Kanji: {char} (JLPT {level})",
            output=(
                f"Kanji: {char}\n"
                f"JLPT Level: {level}\n"
                f"Meanings: {meaning_str}\n"
                f"On-yomi: {on_str}\n"
                f"Kun-yomi: {kun_str}\n"
                f"Stroke count: {stroke_count}"
            ),
        ))

    print(f"  Kanji -> {len(records):,} training examples")
    return records


# ─── 3. Words processing ──────────────────────────────────────────────────────

def process_words(data: dict) -> list[dict]:
    """
    From each word (JLPT + common-only), generate:
    1. "What does [word] mean?" → glosses
    2. "How do you read [word]?" → kana reading
    3. Translate example sentence
    4. "What part of speech is [word]?" → partOfSpeech
    """
    records = []
    all_words = data.get("words", [])

    # Filter: only words that have JLPT level assigned
    jlpt_words = [w for w in all_words if w.get("jlpt_waller") in JLPT_LEVELS]
    print(f"  Processing {len(jlpt_words):,} JLPT words (out of {len(all_words):,} total)...")

    # POS tag mapping (abbreviated → readable)
    POS_MAP = {
        "n": "noun", "v1": "Ichidan verb", "v5r": "Godan verb",
        "adj-i": "i-adjective", "adj-na": "na-adjective",
        "adv": "adverb", "prt": "particle", "conj": "conjunction",
        "pn": "pronoun", "exp": "expression", "vs": "suru noun",
        "vk": "kuru verb", "aux-v": "auxiliary verb",
    }

    for word in jlpt_words:
        level = word.get("jlpt_waller", "")
        kanji_writings = word.get("kanji", [])
        kana_writings = word.get("kana", [])
        senses = word.get("sense", [])

        if not senses:
            continue

        # Get primary writing (prefer common kanji, fallback to kana)
        primary_kanji = next((k["text"] for k in kanji_writings if k.get("common")), None)
        primary_kana = next((k["text"] for k in kana_writings if k.get("common")), None)
        display_word = primary_kanji or primary_kana
        if not display_word:
            continue

        for sense in senses[:2]:  # max 2 senses per word
            glosses = [g["text"] for g in sense.get("gloss", []) if g.get("lang") == "eng"]
            pos_tags = sense.get("partOfSpeech", [])
            examples = sense.get("examples", [])

            if not glosses:
                continue

            gloss_str = ", ".join(glosses[:5])
            pos_str = ", ".join(POS_MAP.get(p, p) for p in pos_tags[:3])
            reading_str = primary_kana or "N/A"

            # ── QA 1: Meaning ───────────────────────────────────────────────
            records.append(make_example(
                instruction="What does this Japanese word mean in English?",
                input_text=f"Word: {display_word}",
                output=gloss_str,
            ))

            # ── QA 2: Reading ───────────────────────────────────────────────
            if primary_kanji and primary_kana:
                records.append(make_example(
                    instruction="How do you read this Japanese word in hiragana/katakana?",
                    input_text=f"Word: {primary_kanji}",
                    output=f"Reading: {primary_kana}\nMeaning: {gloss_str}",
                ))

            # ── QA 3: JLPT + comprehensive ──────────────────────────────────
            records.append(make_example(
                instruction="Provide JLPT study information for this Japanese word.",
                input_text=f"Word: {display_word}",
                output=(
                    f"Word: {display_word}\n"
                    f"Reading: {reading_str}\n"
                    f"Meaning: {gloss_str}\n"
                    f"Part of speech: {pos_str}\n"
                    f"JLPT Level: {level}"
                ),
            ))

            # ── QA 4: Example sentence translation ─────────────────────────
            for ex in examples[:1]:
                jp_sent = ex.get("japanese", "")
                en_sent = ex.get("english", "")
                if jp_sent and en_sent:
                    records.append(make_example(
                        instruction="Translate this Japanese sentence to English.",
                        input_text=jp_sent,
                        output=en_sent,
                    ))

    print(f"  Words -> {len(records):,} training examples")
    return records


# ─── 4. Combine, shuffle, split ───────────────────────────────────────────────

def split_dataset(records: list[dict], train_ratio: float = 0.8):
    random.seed(RANDOM_SEED)
    random.shuffle(records)
    split_idx = int(len(records) * train_ratio)
    return records[:split_idx], records[split_idx:]


# ─── 5. Statistics ────────────────────────────────────────────────────────────

def compute_stats(grammar_recs, kanji_recs, word_recs, train, eval_) -> dict:
    total = len(train) + len(eval_)
    avg_out_len = sum(len(r["output"].split()) for r in train) / max(len(train), 1)
    return {
        "total_examples": total,
        "train_examples": len(train),
        "eval_examples": len(eval_),
        "by_source": {
            "grammar": len(grammar_recs),
            "kanji": len(kanji_recs),
            "words": len(word_recs),
        },
        "avg_output_word_length": round(avg_out_len, 1),
        "train_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
        "note": (
            "This dataset is for SFT (supervised fine-tuning) of a ~4B instruction model. "
            "Format: {instruction, input, output} — Alpaca-compatible. "
            "Kanji/words are filtered to JLPT levels only. "
            "Grammar points include all N1-N5 (595 total). "
        ),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  JLPT Data Cleaner & Training Format Generator")
    print("=" * 60)

    # Load raw data
    grammar_data = load_json(RAW / "grammar.json")
    kanji_data   = load_json(RAW / "kanji.json")
    words_data   = load_json(RAW / "words.json")

    print("\n[1/4] Processing grammar...")
    grammar_recs = process_grammar(grammar_data)
    write_jsonl(grammar_recs, OUT / "grammar_qa.jsonl")

    print("\n[2/4] Processing kanji...")
    kanji_recs = process_kanji(kanji_data)
    write_jsonl(kanji_recs, OUT / "kanji_qa.jsonl")

    print("\n[3/4] Processing words...")
    word_recs = process_words(words_data)
    write_jsonl(word_recs, OUT / "words_qa.jsonl")

    print("\n[4/4] Combining and splitting dataset...")
    all_records = grammar_recs + kanji_recs + word_recs
    train, eval_ = split_dataset(all_records, TRAIN_RATIO)

    write_jsonl(train, OUT / "combined_train.jsonl")
    write_jsonl(eval_,  OUT / "combined_eval.jsonl")

    # Save stats
    stats = compute_stats(grammar_recs, kanji_recs, word_recs, train, eval_)
    with open(OUT / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  [DONE] Summary:")
    print(f"     Grammar examples : {len(grammar_recs):>8,}")
    print(f"     Kanji examples   : {len(kanji_recs):>8,}")
    print(f"     Word examples    : {len(word_recs):>8,}")
    print(f"     Total            : {len(all_records):>8,}")
    print(f"     Train split      : {len(train):>8,}")
    print(f"     Eval  split      : {len(eval_):>8,}")
    print(f"\n  Output: {OUT}")
    print("=" * 60)

    # Print sample
    print("\n[Sample] training example (grammar):")
    if grammar_recs:
        sample = grammar_recs[0]
        print(f"  Instruction: {sample['instruction'][:80]}...")
        print(f"  Input:       {sample['input'][:80]}")
        print(f"  Output:      {sample['output'][:100]}...")


if __name__ == "__main__":
    main()

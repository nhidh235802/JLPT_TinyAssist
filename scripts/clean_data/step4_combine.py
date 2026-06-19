# step4_combine.py
# =============================================================================
# BƯỚC 4: Gộp data + Chia train/eval
#
# ĐỌC TRƯỚC: Chọn MODE phù hợp
#
# Cấu trúc bài thi JLPT thật:
#   問題1: Cho kanji → chọn cách ĐỌC đúng  (step2b_kanji_reading_mcq.py)
#   問題2: Cho kana  → chọn cách VIẾT kanji (step2c_kanji_writing_mcq.py)
#   問題3-5: Từ vựng (ý nghĩa, cách dùng)   (step3_words.py)
#   問題6-7: Ngữ pháp fill-in-blank          (step1b_grammar_mcq.py)
#   問題8:   Sắp xếp câu (ngữ pháp)          (step1_grammar.py)
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ MODE = "grammar_both"    Q&A + trắc nghiệm ngữ pháp  (~7,735 ex.)       │
# │                          Train ~45 phút Colab                            │
# │                                                                          │
# │ MODE = "vocab_mcq"       問題1+2: Reading + Writing MCQ (~25,628 ex.)   │
# │                          Train ~2 giờ Colab                             │
# │                                                                          │
# │ MODE = "jlpt_full" ← KHUYẾN NGHỊ                                        │
# │                          Tất cả dạng bài JLPT (~33,363 ex.)             │
# │                          Grammar Q&A + Grammar MCQ + Vocab MCQ          │
# │                          Train ~2-3 giờ Colab                           │
# │                                                                          │
# │ MODE = "full"            Mọi thứ kể cả Q&A mở về kanji/từ (~71,000 ex.)│
# │                          Train ~5-7 giờ Colab                           │
# └──────────────────────────────────────────────────────────────────────────┘

MODE = "jlpt_full"   # ← ĐỔI TẠI ĐÂY

# =============================================================================
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import OUT_DIR, save_jsonl

TRAIN_RATIO = 0.80   # 80% train, 20% eval
RANDOM_SEED = 42     # seed cố định để kết quả reproducible


def doc_jsonl(filename: str) -> list[dict]:
    """
    Đọc file JSONL từ thư mục data/processed/.
    Mỗi dòng là 1 JSON object → trả về list các dict.
    """
    path = OUT_DIR / filename
    if not path.exists():
        print(f"  [!] Khong tim thay: {filename} — hay chay buoc tuong ung truoc.")
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Doc {filename}: {len(records):,} examples")
    return records


def gop_va_chia(grammar_recs, grammar_mcq_recs, kanji_recs, word_recs,
                reading_mcq_recs, writing_mcq_recs,
                fill_mcq_recs, synonym_mcq_recs, usage_mcq_recs) -> tuple[list, list]:
    """Gộp các nguồn data theo MODE, shuffle, chia train/eval."""

    if MODE == "grammar_only":
        print(f"\n  [MODE: grammar_only] — Q&A mo ve ngu phap")
        tat_ca = grammar_recs
        print(f"  Grammar Q&A  : {len(grammar_recs):,}")

    elif MODE == "grammar_mcq":
        print(f"\n  [MODE: grammar_mcq] — Trac nghiem ngu phap (Mon 6-7)")
        tat_ca = grammar_mcq_recs
        print(f"  Grammar MCQ  : {len(grammar_mcq_recs):,}")

    elif MODE == "grammar_both":
        print(f"\n  [MODE: grammar_both] — Q&A mo + trac nghiem ngu phap")
        tat_ca = grammar_recs + grammar_mcq_recs
        print(f"  Grammar Q&A  : {len(grammar_recs):,}")
        print(f"  Grammar MCQ  : {len(grammar_mcq_recs):,}")

    elif MODE == "vocab_mcq":
        print(f"\n  [MODE: vocab_mcq] — Trac nghiem tu vung (Mon 1-5)")
        tat_ca = reading_mcq_recs + writing_mcq_recs + fill_mcq_recs + synonym_mcq_recs + usage_mcq_recs
        print(f"  Reading MCQ  : {len(reading_mcq_recs):,}  (Mon 1)")
        print(f"  Writing MCQ  : {len(writing_mcq_recs):,}  (Mon 2)")
        print(f"  Fill MCQ     : {len(fill_mcq_recs):,}  (Mon 3)")
        print(f"  Synonym MCQ  : {len(synonym_mcq_recs):,}  (Mon 4)")
        print(f"  Usage MCQ    : {len(usage_mcq_recs):,}  (Mon 5)")

    elif MODE == "jlpt_full":
        print(f"\n  [MODE: jlpt_full] — Tat ca dang bai JLPT (KHUYEN NGHI)")
        tat_ca = grammar_recs + grammar_mcq_recs + reading_mcq_recs + writing_mcq_recs + fill_mcq_recs + synonym_mcq_recs + usage_mcq_recs
        print(f"  Grammar Q&A  : {len(grammar_recs):,}   (hieu sau ngu phap)")
        print(f"  Grammar MCQ  : {len(grammar_mcq_recs):,}   (Mon 6-7)")
        print(f"  Reading MCQ  : {len(reading_mcq_recs):,}  (Mon 1)")
        print(f"  Writing MCQ  : {len(writing_mcq_recs):,}  (Mon 2)")
        print(f"  Fill MCQ     : {len(fill_mcq_recs):,}  (Mon 3)")
        print(f"  Synonym MCQ  : {len(synonym_mcq_recs):,}  (Mon 4)")
        print(f"  Usage MCQ    : {len(usage_mcq_recs):,}  (Mon 5)")

    elif MODE == "full":
        print(f"\n  [MODE: full] — Moi thu")
        tat_ca = (grammar_recs + grammar_mcq_recs + kanji_recs + word_recs
                  + reading_mcq_recs + writing_mcq_recs + fill_mcq_recs + synonym_mcq_recs + usage_mcq_recs)
        print(f"  Grammar Q&A  : {len(grammar_recs):,}")
        print(f"  Grammar MCQ  : {len(grammar_mcq_recs):,}")
        print(f"  Kanji Q&A    : {len(kanji_recs):,}")
        print(f"  Words Q&A    : {len(word_recs):,}")
        print(f"  Reading MCQ  : {len(reading_mcq_recs):,}")
        print(f"  Writing MCQ  : {len(writing_mcq_recs):,}")
        print(f"  Fill MCQ     : {len(fill_mcq_recs):,}")
        print(f"  Synonym MCQ  : {len(synonym_mcq_recs):,}")
        print(f"  Usage MCQ    : {len(usage_mcq_recs):,}")

    else:
        raise ValueError(f"MODE khong hop le: '{MODE}'")

    print(f"  Tong cong    : {len(tat_ca):,} examples")

    random.seed(RANDOM_SEED)
    random.shuffle(tat_ca)

    vi_tri_cat = int(len(tat_ca) * TRAIN_RATIO)
    train = tat_ca[:vi_tri_cat]
    eval_ = tat_ca[vi_tri_cat:]

    print(f"\n  Chia data (seed={RANDOM_SEED}):")
    print(f"    Train ({TRAIN_RATIO*100:.0f}%): {len(train):,} examples")
    print(f"    Eval  ({(1-TRAIN_RATIO)*100:.0f}%): {len(eval_):,} examples")

    return train, eval_


def luu_thong_ke(grammar_recs, grammar_mcq_recs, kanji_recs, word_recs,
                 reading_mcq_recs, writing_mcq_recs,
                 fill_mcq_recs, synonym_mcq_recs, usage_mcq_recs,
                 train, eval_):
    """Lưu file stats.json để ghi nhớ cấu hình dataset."""
    trung_binh = (
        sum(len(r["output"].split()) for r in train) / len(train)
        if train else 0
    )
    stats = {
        "mode"                   : MODE,
        "tong_examples"          : len(train) + len(eval_),
        "train_examples"         : len(train),
        "eval_examples"          : len(eval_),
        "theo_nguon": {
            "grammar_qa"         : len(grammar_recs),
            "grammar_mcq"        : len(grammar_mcq_recs),
            "kanji_qa"           : len(kanji_recs),
            "words_qa"           : len(word_recs),
            "reading_mcq"        : len(reading_mcq_recs),
            "writing_mcq"        : len(writing_mcq_recs),
            "fill_mcq"           : len(fill_mcq_recs),
            "synonym_mcq"        : len(synonym_mcq_recs),
            "usage_mcq"          : len(usage_mcq_recs),
        },
        "trung_binh_do_dai_output_tu": round(trung_binh, 1),
        "train_ratio"            : TRAIN_RATIO,
        "random_seed"            : RANDOM_SEED,
        "format"                 : "Alpaca (instruction / input / output)",
    }
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Da luu -> data/processed/stats.json")
    return stats


# =============================================================================
# CHẠY TRỰC TIẾP: python scripts/clean_data/step4_combine.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print(f"BUOC 4: Gop data va chia train/eval  [MODE={MODE}]")
    print("=" * 55)

    # Xác định những file nào cần load
    need_grammar_qa  = MODE in ("grammar_only", "grammar_both", "jlpt_full", "full")
    need_grammar_mcq = MODE in ("grammar_mcq",  "grammar_both", "jlpt_full", "full")
    need_kanji_qa    = MODE == "full"
    need_words_qa    = MODE == "full"
    need_vocab_mcq   = MODE in ("vocab_mcq", "jlpt_full", "full")

    grammar_recs     = doc_jsonl("grammar_qa.jsonl")        if need_grammar_qa  else []
    grammar_mcq_recs = doc_jsonl("grammar_mcq.jsonl")       if need_grammar_mcq else []
    kanji_recs       = doc_jsonl("kanji_qa.jsonl")          if need_kanji_qa    else []
    word_recs        = doc_jsonl("words_qa.jsonl")          if need_words_qa    else []
    reading_mcq_recs = doc_jsonl("kanji_reading_mcq.jsonl") if need_vocab_mcq   else []
    writing_mcq_recs = doc_jsonl("kanji_writing_mcq.jsonl") if need_vocab_mcq   else []
    fill_mcq_recs    = doc_jsonl("vocab_fill_mcq.jsonl")    if need_vocab_mcq   else []
    synonym_mcq_recs = doc_jsonl("vocab_synonym_mcq.jsonl") if need_vocab_mcq   else []
    usage_mcq_recs   = doc_jsonl("vocab_usage_mcq.jsonl")   if need_vocab_mcq   else []

    if not any([grammar_recs, grammar_mcq_recs, reading_mcq_recs, writing_mcq_recs, fill_mcq_recs, synonym_mcq_recs, usage_mcq_recs]):
        print("[!] Khong co data. Chay cac buoc step1-3 truoc.")
        exit(1)

    train, eval_ = gop_va_chia(
        grammar_recs, grammar_mcq_recs, kanji_recs, word_recs,
        reading_mcq_recs, writing_mcq_recs, fill_mcq_recs, synonym_mcq_recs, usage_mcq_recs
    )

    print("\nLuu ra file...")
    save_jsonl(train, "combined_train.jsonl")
    save_jsonl(eval_,  "combined_eval.jsonl")
    stats = luu_thong_ke(
        grammar_recs, grammar_mcq_recs, kanji_recs, word_recs,
        reading_mcq_recs, writing_mcq_recs, fill_mcq_recs, synonym_mcq_recs, usage_mcq_recs, train, eval_
    )

    print("\n" + "=" * 55)
    print("HOAN THANH!")
    print(f"  combined_train.jsonl : {stats['train_examples']:,} examples <- dung de TRAIN")
    print(f"  combined_eval.jsonl  : {stats['eval_examples']:,} examples <- dung de DANH GIA")
    print("\nBuoc tiep theo: chay step5_check.py de kiem tra data")


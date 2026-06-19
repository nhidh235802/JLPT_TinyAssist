# step2b_kanji_reading_mcq.py
# =============================================================================
# BƯỚC 2B: Tạo MCQ dạng 問題1 — Chọn CÁCH ĐỌC đúng của từ được gạch chân
#
# INPUT : data/raw/words.json
# OUTPUT: data/processed/kanji_reading_mcq.jsonl
#
# Đây là dạng bài JLPT thật (問題1):
#   "___の言葉の読み方として最もよいものを、1・2・3・4から一つ選びなさい。"
#
#   Ví dụ:
#   その会社の【情報】は、インターネットで見ました。
#   1) じょうほう  2) ちょうほう  3) じょうぼう  4) ちょうぼう
#
# Cách tạo 3 đáp án sai (distractor):
#   → Lấy cách đọc của các TỪ KHÁC có cùng JLPT level
#   → Tránh trùng với đáp án đúng
#   → Ưu tiên từ có cách đọc nghe tương tự (cùng âm đầu, cùng độ dài)
# =============================================================================

import sys
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from helpers import load_json, save_jsonl, make_qa

JLPT_LEVELS = {"N1", "N2", "N3", "N4", "N5"}
RANDOM_SEED  = 42
LABELS       = ["1", "2", "3", "4"]   # JLPT dùng số 1-4, không dùng A-D


def xay_dung_index_words(all_words: list) -> dict:
    """
    Tạo index từ theo level để tìm distractor nhanh:
    { "N3": [ {word_dict}, ... ], "N1": [...], ... }
    """
    by_level = defaultdict(list)
    for w in all_words:
        lv = w.get("jlpt_waller")
        if lv in JLPT_LEVELS:
            by_level[lv].append(w)
    return by_level


def lay_reading_distractors(correct_kana: str, level: str, by_level: dict,
                             so_luong: int = 3) -> list[str]:
    """
    Lấy 3 cách đọc SAI từ các từ cùng level.

    Chiến lược:
    1. Ưu tiên từ có độ dài kana giống nhau (nghe tương tự)
    2. Không được trùng với đáp án đúng
    3. Không lấy đáp án trùng nhau
    """
    do_dai = len(correct_kana)
    candidates_same_len = []
    candidates_other    = []

    for w in by_level.get(level, []):
        kana_list = w.get("kana", [])
        primary_kana = next((k["text"] for k in kana_list if k.get("common")), None)
        if not primary_kana:
            primary_kana = kana_list[0]["text"] if kana_list else None
        if not primary_kana or primary_kana == correct_kana:
            continue
        if len(primary_kana) == do_dai:
            candidates_same_len.append(primary_kana)
        else:
            candidates_other.append(primary_kana)

    random.shuffle(candidates_same_len)
    random.shuffle(candidates_other)

    pool = candidates_same_len + candidates_other

    # Lọc trùng
    seen    = {correct_kana}
    results = []
    for kana in pool:
        if kana not in seen:
            seen.add(kana)
            results.append(kana)
        if len(results) >= so_luong:
            break

    return results


def tao_reading_mcq(data: dict) -> list[dict]:
    """
    Tạo câu hỏi 問題1: "Từ này đọc là gì?" dạng trắc nghiệm 4 lựa chọn.
    """
    results   = []
    all_words = data.get("words", [])

    # Chỉ dùng từ có JLPT level VÀ có cả kanji lẫn kana
    jlpt_words = [
        w for w in all_words
        if w.get("jlpt_waller") in JLPT_LEVELS
        and w.get("kanji")
        and w.get("kana")
    ]

    print(f"  Tu co kanji + kana + JLPT level: {len(jlpt_words):,}")

    by_level = xay_dung_index_words(all_words)
    random.seed(RANDOM_SEED)

    for word in jlpt_words:
        level        = word.get("jlpt_waller", "")
        kanji_list   = word.get("kanji", [])
        kana_list    = word.get("kana", [])
        senses       = word.get("sense", [])

        # Lấy dạng viết kanji chính và cách đọc kana chính
        kanji_chinh = next((k["text"] for k in kanji_list if k.get("common")), None)
        kana_chinh  = next((k["text"] for k in kana_list  if k.get("common")), None)
        if not kanji_chinh or not kana_chinh:
            continue

        # Lấy nghĩa tiếng Anh (để đưa vào explanation)
        nghia = ""
        if senses:
            glosses = [g["text"] for g in senses[0].get("gloss", []) if g.get("lang") == "eng"]
            nghia   = ", ".join(glosses[:3])

        # Lấy câu ví dụ (nếu có) để tạo câu hỏi có ngữ cảnh
        cau_vi_du_jp = ""
        cau_vi_du_en = ""
        for sense in senses:
            for ex in sense.get("examples", []):
                if ex.get("japanese") and kanji_chinh in ex.get("japanese", ""):
                    cau_vi_du_jp = ex["japanese"]
                    cau_vi_du_en = ex.get("english", "")
                    break
            if cau_vi_du_jp:
                break

        # Lấy 3 distractor
        distractors = lay_reading_distractors(kana_chinh, level, by_level, so_luong=3)
        if len(distractors) < 3:
            continue

        # ── Xáo trộn thứ tự 1/2/3/4 ─────────────────────────────────────────
        cac_lua_chon = [
            {"label": None, "reading": kana_chinh, "la_dung": True},
            {"label": None, "reading": distractors[0], "la_dung": False},
            {"label": None, "reading": distractors[1], "la_dung": False},
            {"label": None, "reading": distractors[2], "la_dung": False},
        ]
        random.shuffle(cac_lua_chon)
        for i, lc in enumerate(cac_lua_chon):
            lc["label"] = LABELS[i]

        dung_label = next(lc["label"] for lc in cac_lua_chon if lc["la_dung"])

        # ── Phần INPUT ───────────────────────────────────────────────────────
        # Có 2 dạng câu hỏi:
        #   1. Có câu ngữ cảnh: dùng câu ví dụ thật
        #   2. Không có câu: chỉ hỏi từ đơn lẻ
        if cau_vi_du_jp:
            # Đánh dấu từ đang hỏi bằng【】giống đề JLPT thật
            cau_hien_thi = cau_vi_du_jp.replace(kanji_chinh, f"【{kanji_chinh}】", 1)
            input_text = (
                f"Sentence: {cau_hien_thi}\n"
                f"Question: How do you read 【{kanji_chinh}】?\n"
            )
        else:
            input_text = f"Question: How do you read the word 【{kanji_chinh}】?\n"

        for lc in cac_lua_chon:
            input_text += f"{lc['label']}) {lc['reading']}\n"
        input_text = input_text.strip()

        # ── Phần OUTPUT (giải thích CoT) ─────────────────────────────────────
        cot = "<thinking>\n"
        cot += f"1. Target Word Analysis: The underlined word is 【{kanji_chinh}】. We need to determine its correct furigana reading.\n"
        if cau_vi_du_en:
            cot += f"   - The sentence translates to '{cau_vi_du_en}', which provides context.\n"
        cot += f"   - The word 【{kanji_chinh}】 means '{nghia}'.\n"
        cot += "2. Option Analysis:\n"
        for lc in cac_lua_chon:
            status = "Correct" if lc["la_dung"] else "Incorrect"
            reason = f"The correct reading of {kanji_chinh} is {kana_chinh}." if lc["la_dung"] else f"This is an incorrect reading or belongs to a different word."
            cot += f"   - {lc['label']}) {lc['reading']}: {status}. {reason}\n"
        cot += f"</thinking>\n<answer> {dung_label} </answer>"

        instruction_text = (
            "問題1: ＿＿＿の言葉の読み方として最もよいものを、1・2・3・4から一つ選びなさい。\n"
            "Solve this JLPT kanji reading question step-by-step. Provide your reasoning in a <thinking> block, and output the final number in an <answer> block."
        )

        results.append(make_qa(
            instruction = instruction_text,
            input_text  = input_text,
            output      = cot,
        ))

    print(f"  Ket qua: {len(results):,} reading MCQ examples")
    return results


# =============================================================================
# CHẠY TRỰC TIẾP: python scripts/clean_data/step2b_kanji_reading_mcq.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 2B: Tao MCQ doc kanji (Mon 1 - Reading)")
    print("=" * 55)
    print()
    print("Dang bai: Chon cach DOC dung cua tu duoc gach chan")
    print("Vi du:  【情報】 → 1)じょうほう  2)ちょうほう  3)じょうぼう  4)ちょうぼう")
    print()

    raw_data = load_json("words.json")
    records  = tao_reading_mcq(raw_data)
    save_jsonl(records, "kanji_reading_mcq.jsonl")

    print("\n--- Vi du 1 Reading MCQ ---")
    if records:
        s = records[0]
        print(f"\nInstruction: {s['instruction']}")
        print(f"\nInput:\n{s['input']}")
        print(f"\nOutput:\n{s['output']}")
    print("\nHoan thanh! -> data/processed/kanji_reading_mcq.jsonl")

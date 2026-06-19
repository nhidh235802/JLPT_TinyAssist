# step2c_kanji_writing_mcq.py
# =============================================================================
# BƯỚC 2C: Tạo MCQ dạng 問題2 — Chọn cách VIẾT KANJI đúng từ chữ kana
#
# INPUT : data/raw/words.json
# OUTPUT: data/processed/kanji_writing_mcq.jsonl
#
# Đây là dạng bài JLPT thật (問題2):
#   "___のことばを漢字で書くとき、最もよいものを、1・2・3・4からえらびなさい。"
#
#   Ví dụ:
#   私は毎日、どのくらい歩いたか【きろく】している。
#   1) 基緑  2) 記録  3) 基録  4) 記録
#
# Cách tạo 3 đáp án sai:
#   → Dùng các TỪ KHÁC có cách đọc GIỐNG NHAU (đồng âm dị nghĩa / dị tự)
#     Ví dụ: きろく → 記録 (đúng), 基録 (sai - ghép sai kanji), ...
#   → Nếu không đủ từ đồng âm, dùng từ có cách đọc tương tự (cùng âm đầu)
# =============================================================================

import sys
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from helpers import load_json, save_jsonl, make_qa

JLPT_LEVELS = {"N1", "N2", "N3", "N4", "N5"}
RANDOM_SEED  = 42
LABELS       = ["1", "2", "3", "4"]


def xay_dung_index_kana(all_words: list) -> dict:
    """
    Tạo index từ theo cách đọc kana:
    { "きろく": [ {word với kanji 記録}, ... ],
      "けいかく": [ {word với kanji 計画}, ... ] }

    Dùng để tìm từ ĐỒNG ÂM (cùng cách đọc, khác chữ kanji) → distractor tốt nhất.
    """
    by_kana = defaultdict(list)
    for w in all_words:
        for k in w.get("kana", []):
            kana = k.get("text", "")
            if kana:
                by_kana[kana].append(w)
    return by_kana


def xay_dung_index_level(all_words: list) -> dict:
    """Index theo JLPT level để fallback khi không có từ đồng âm."""
    by_level = defaultdict(list)
    for w in all_words:
        lv = w.get("jlpt_waller")
        if lv in JLPT_LEVELS:
            by_level[lv].append(w)
    return by_level


def lay_kanji_cua_word(word: dict) -> str | None:
    """Lấy cách viết kanji chính (common=True) của 1 từ."""
    kanji_list = word.get("kanji", [])
    kanji_chinh = next((k["text"] for k in kanji_list if k.get("common")), None)
    if not kanji_chinh and kanji_list:
        kanji_chinh = kanji_list[0].get("text")
    return kanji_chinh


def lay_writing_distractors(correct_kanji: str, correct_kana: str, level: str,
                             by_kana: dict, by_level: dict,
                             so_luong: int = 3) -> list[str]:
    """
    Lấy 3 cách viết KANJI SAI làm distractor.

    Chiến lược (theo thứ tự ưu tiên):
    1. Từ ĐỒNG ÂM: cùng cách đọc kana nhưng viết kanji khác
       → Đây là distractor tốt nhất, giống đề JLPT thật
    2. Từ cùng JLPT level nhưng có kanji (bất kỳ)
    """
    seen     = {correct_kanji}
    results  = []

    # ── Ưu tiên 1: từ đồng âm (cùng kana, khác kanji) ───────────────────────
    for word in by_kana.get(correct_kana, []):
        kanji = lay_kanji_cua_word(word)
        if kanji and kanji not in seen:
            seen.add(kanji)
            results.append(kanji)
        if len(results) >= so_luong:
            return results

    # ── Ưu tiên 2: cùng level ────────────────────────────────────────────────
    pool = list(by_level.get(level, []))
    random.shuffle(pool)
    for word in pool:
        kanji = lay_kanji_cua_word(word)
        if kanji and kanji not in seen:
            seen.add(kanji)
            results.append(kanji)
        if len(results) >= so_luong:
            return results

    return results


def tao_writing_mcq(data: dict) -> list[dict]:
    """
    Tạo câu hỏi 問題2: "Từ kana này viết bằng kanji nào?" dạng trắc nghiệm.
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

    by_kana  = xay_dung_index_kana(all_words)
    by_level = xay_dung_index_level(all_words)
    random.seed(RANDOM_SEED)

    for word in jlpt_words:
        level      = word.get("jlpt_waller", "")
        kanji_list = word.get("kanji", [])
        kana_list  = word.get("kana", [])
        senses     = word.get("sense", [])

        kanji_chinh = next((k["text"] for k in kanji_list if k.get("common")), None)
        kana_chinh  = next((k["text"] for k in kana_list  if k.get("common")), None)
        if not kanji_chinh or not kana_chinh:
            continue

        # Lấy nghĩa
        nghia = ""
        if senses:
            glosses = [g["text"] for g in senses[0].get("gloss", []) if g.get("lang") == "eng"]
            nghia   = ", ".join(glosses[:3])

        # Lấy câu ví dụ có chứa kana này
        cau_vi_du_jp = ""
        cau_vi_du_en = ""
        for sense in senses:
            for ex in sense.get("examples", []):
                jp = ex.get("japanese", "")
                # Câu ví dụ có thể dùng kanji hoặc kana
                if jp and (kanji_chinh in jp or kana_chinh in jp):
                    cau_vi_du_jp = jp
                    cau_vi_du_en = ex.get("english", "")
                    break
            if cau_vi_du_jp:
                break

        # Lấy 3 distractor (kanji sai)
        distractors = lay_writing_distractors(
            kanji_chinh, kana_chinh, level, by_kana, by_level, so_luong=3
        )
        if len(distractors) < 3:
            continue

        # ── Xáo trộn thứ tự 1/2/3/4 ─────────────────────────────────────────
        cac_lua_chon = [
            {"label": None, "kanji": kanji_chinh, "la_dung": True},
            {"label": None, "kanji": distractors[0], "la_dung": False},
            {"label": None, "kanji": distractors[1], "la_dung": False},
            {"label": None, "kanji": distractors[2], "la_dung": False},
        ]
        random.shuffle(cac_lua_chon)
        for i, lc in enumerate(cac_lua_chon):
            lc["label"] = LABELS[i]

        dung_label = next(lc["label"] for lc in cac_lua_chon if lc["la_dung"])

        # ── Phần INPUT ───────────────────────────────────────────────────────
        # Trong câu hỏi, phần kana được gạch chân → hiển thị bằng【】
        if cau_vi_du_jp:
            # Ưu tiên dùng câu có kana (vì đây là câu hỏi về cách VIẾT)
            if kana_chinh in cau_vi_du_jp:
                cau_hien_thi = cau_vi_du_jp.replace(kana_chinh, f"【{kana_chinh}】", 1)
            else:
                # Câu dùng kanji → thay kanji bằng kana để tạo ngữ cảnh
                cau_hien_thi = cau_vi_du_jp.replace(kanji_chinh, f"【{kana_chinh}】", 1)
            input_text = (
                f"Sentence: {cau_hien_thi}\n"
                f"Question: Which kanji correctly writes 【{kana_chinh}】?\n"
            )
        else:
            input_text = f"Question: Which kanji correctly writes the word 【{kana_chinh}】?\n"

        for lc in cac_lua_chon:
            input_text += f"{lc['label']}) {lc['kanji']}\n"
        input_text = input_text.strip()

        # ── Phần OUTPUT (giải thích CoT) ─────────────────────────────────────
        cot = "<thinking>\n"
        cot += f"1. Target Word Analysis: The underlined word is 【{kana_chinh}】. We need to determine its correct kanji spelling.\n"
        if cau_vi_du_en:
            cot += f"   - The sentence translates to '{cau_vi_du_en}', which provides context.\n"
        cot += f"   - The word means '{nghia}'.\n"
        cot += "2. Option Analysis:\n"
        for lc in cac_lua_chon:
            status = "Correct" if lc["la_dung"] else "Incorrect"
            reason = f"The kanji {kanji_chinh} is the standard way to write this word." if lc["la_dung"] else f"This kanji is either incorrect or belongs to a homophone with a different meaning."
            cot += f"   - {lc['label']}) {lc['kanji']}: {status}. {reason}\n"
        cot += f"</thinking>\n<answer> {dung_label} </answer>"

        instruction_text = (
            "問題2: ＿＿＿のことばを漢字で書くとき、最もよいものを、1・2・3・4から一つえらびなさい。\n"
            "Solve this JLPT kanji writing question step-by-step. Provide your reasoning in a <thinking> block, and output the final number in an <answer> block."
        )

        results.append(make_qa(
            instruction = instruction_text,
            input_text  = input_text,
            output      = cot,
        ))

    print(f"  Ket qua: {len(results):,} writing MCQ examples")
    return results


# =============================================================================
# CHẠY TRỰC TIẾP: python scripts/clean_data/step2c_kanji_writing_mcq.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 2C: Tao MCQ viet kanji (Mon 2 - Writing)")
    print("=" * 55)
    print()
    print("Dang bai: Cho chu kana → chon kanji dung")
    print("Vi du:  【きろく】 → 1)基緑  2)記録  3)基録  4)記録")
    print()

    raw_data = load_json("words.json")
    records  = tao_writing_mcq(raw_data)
    save_jsonl(records, "kanji_writing_mcq.jsonl")

    print("\n--- Vi du 1 Writing MCQ ---")
    if records:
        s = records[0]
        print(f"\nInstruction: {s['instruction']}")
        print(f"\nInput:\n{s['input']}")
        print(f"\nOutput:\n{s['output']}")
    print("\nHoan thanh! -> data/processed/kanji_writing_mcq.jsonl")

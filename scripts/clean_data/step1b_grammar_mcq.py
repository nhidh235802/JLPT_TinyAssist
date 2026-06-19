# step1b_grammar_mcq.py
# =============================================================================
# BƯỚC 1B: Tạo câu hỏi TRẮC NGHIỆM 4 ĐÁP ÁN từ grammar.json
#          (giống format đề thi JLPT thật)
#
# INPUT : data/raw/grammar.json
# OUTPUT: data/processed/grammar_mcq.jsonl
#
# Tại sao cần file này?
#   → Step1 (step1_grammar.py) tạo câu hỏi MỞ: "Mẫu X có nghĩa là gì?"
#   → Nhưng JLPT là trắc nghiệm 4 lựa chọn: chọn A/B/C/D
#   → Model cần học cả 2 kỹ năng:
#       1. Hiểu sâu ngữ pháp (từ step1)
#       2. Loại trừ đáp án sai trong 4 lựa chọn (từ file này)
#
# Cách tạo 3 ĐÁP ÁN SAI (distractors):
#   - Ưu tiên 1: Dùng trường "related" trong data (các mẫu liên quan)
#     Vì chúng gần nghĩa → khó phân biệt → giống đề thi thật nhất
#   - Ưu tiên 2: Lấy ngẫu nhiên các mẫu cùng JLPT level
#   - Ưu tiên 3: Lấy ngẫu nhiên bất kỳ mẫu nào
#
# Ví dụ output (1 training example):
# {
#   "instruction": "Choose the correct grammar pattern to complete the sentence.",
#   "input": "Sentence: 彼の苦労は（　）。\nA) 想像に難くない\nB) 想像にたえない\nC) 想像するほかない\nD) 想像にすぎない",
#   "output": "Answer: A\nCorrect pattern: 想像に難くない\nMeaning: easy to imagine / not hard to\nWhy A is correct: ...\nWhy others are wrong: ..."
# }
# =============================================================================

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import load_json, save_jsonl, make_qa

RANDOM_SEED = 42
# Nhãn 4 đáp án
LABELS = ["A", "B", "C", "D"]


def xay_dung_index(grammar_points: list) -> dict:
    """
    Tạo 2 cấu trúc để tra cứu nhanh khi tìm distractors:
      - by_id   : { "id_slug": grammar_point_dict }  → tra theo ID
      - by_level: { "N1": [gp, gp, ...] }            → tra theo level
    """
    by_id    = {gp["id"]: gp for gp in grammar_points if gp.get("id")}
    by_level = {}
    for gp in grammar_points:
        lv = gp.get("level", "")
        if lv:
            by_level.setdefault(lv, []).append(gp)
    return by_id, by_level


def lay_distractors(gp: dict, by_id: dict, by_level: dict, so_luong: int = 3) -> list:
    """
    Lấy 'so_luong' đáp án sai cho câu hỏi về grammar point 'gp'.

    Chiến lược (theo thứ tự ưu tiên):
    1. Dùng các mẫu trong trường "related" (liên quan, khó phân biệt)
    2. Bổ sung bằng các mẫu cùng JLPT level
    3. Nếu vẫn thiếu, lấy từ bất kỳ mẫu nào
    """
    dung_id    = gp.get("id", "")
    level      = gp.get("level", "")
    related_ids = gp.get("related", [])

    candidates = []

    # ── Ưu tiên 1: related patterns ──────────────────────────────────────────
    # "related" là danh sách ID của các mẫu ngữ pháp liên quan
    # Đây là distractor tốt nhất vì chúng gần nghĩa, dễ nhầm lẫn
    for rid in related_ids:
        if rid in by_id and rid != dung_id:
            candidates.append(by_id[rid])

    # ── Ưu tiên 2: cùng JLPT level ───────────────────────────────────────────
    # Lấy các mẫu cùng level nhưng chưa có trong candidates
    da_co_ids = {c["id"] for c in candidates} | {dung_id}
    cung_level = [
        p for p in by_level.get(level, [])
        if p.get("id") not in da_co_ids
    ]
    # Xáo trộn để lấy ngẫu nhiên
    random.shuffle(cung_level)
    candidates.extend(cung_level)

    # ── Ưu tiên 3: bất kỳ mẫu nào (fallback) ────────────────────────────────
    if len(candidates) < so_luong:
        da_co_ids = {c["id"] for c in candidates} | {dung_id}
        tat_ca_con_lai = [
            p for p in by_id.values()
            if p.get("id") not in da_co_ids
        ]
        random.shuffle(tat_ca_con_lai)
        candidates.extend(tat_ca_con_lai)

    # Trả về đúng số lượng cần
    return candidates[:so_luong]


def tao_mcq_grammar(data: dict) -> list[dict]:
    """
    Với mỗi grammar point có ví dụ câu:
      → Tạo câu hỏi trắc nghiệm 4 đáp án (A/B/C/D)
      → Đáp án đúng là grammar point hiện tại
      → 3 đáp án sai là các mẫu distractors

    Output format (Alpaca):
      instruction: "Choose the correct grammar pattern..."
      input:       câu hỏi + 4 lựa chọn A/B/C/D
      output:      đáp án đúng + giải thích tại sao đúng/sai
    """
    results = []
    grammar_points = data.get("grammar_points", [])

    # Xây dựng index để tìm distractors nhanh
    by_id, by_level = xay_dung_index(grammar_points)

    random.seed(RANDOM_SEED)

    print(f"  Tim thay {len(grammar_points)} diem ngu phap, bat dau tao MCQ...")
    dem_co_vi_du = 0

    for gp in grammar_points:
        pattern       = gp.get("pattern", "")
        level         = gp.get("level", "")
        meaning_en    = gp.get("meaning_en", "")
        meaning_detail = gp.get("meaning_detailed", "")
        examples      = gp.get("examples", [])
        formality     = gp.get("formality", "")

        # Bỏ qua nếu không có câu ví dụ — không thể tạo câu hỏi fill-in-blank
        if not examples or not pattern or not meaning_en:
            continue

        dem_co_vi_du += 1

        # Lấy 3 distractors
        distractors = lay_distractors(gp, by_id, by_level, so_luong=3)
        if len(distractors) < 3:
            continue  # bỏ qua nếu không đủ distractors

        # ─────────────────────────────────────────────────────────────────────
        # Tạo MCQ cho từng câu ví dụ (tối đa 2 câu/pattern)
        # ─────────────────────────────────────────────────────────────────────
        for ex in examples[:2]:
            jp  = ex.get("japanese", "")
            en  = ex.get("english", "")
            if not jp or not en:
                continue

            # ── Tạo câu hỏi dạng fill-in-the-blank ──────────────────────────
            # Thay phần cốt lõi của pattern bằng （　）trong câu ví dụ.
            # Ví dụ: "彼の苦労は想像に難くない。" → "彼の苦労は（　）。"
            #
            # Chiến lược: tìm chuỗi ký tự Nhật dài nhất trong pattern,
            # rồi tìm chuỗi đó trong câu ví dụ và thay bằng blank.
            import re as _re
            # Lấy tất cả đoạn ký tự Nhật liên tục từ pattern
            # Ví dụ: "Noun / V dict + に難くない" → ["に難くない"]
            jp_tokens = _re.findall(r'[\u3000-\u9fff\uff00-\uffef]+', pattern)
            jp_co_blank = jp  # mặc định nếu không tìm được
            for token in reversed(jp_tokens):
                # Xóa dấu okurigana "." (vd: "いと.しい" → "いとしい")
                token_clean = token.replace(".", "")
                if len(token_clean) >= 2 and token_clean in jp:
                    jp_co_blank = jp.replace(token_clean, "（　）", 1)
                    break

            # ── Xáo trộn thứ tự A/B/C/D ──────────────────────────────────────
            # Đáp án đúng không được luôn là A!
            cac_lua_chon = [
                {"label": None, "gp": gp, "la_dung": True},
                {"label": None, "gp": distractors[0], "la_dung": False},
                {"label": None, "gp": distractors[1], "la_dung": False},
                {"label": None, "gp": distractors[2], "la_dung": False},
            ]
            random.shuffle(cac_lua_chon)

            # Gán nhãn A/B/C/D
            for i, lua_chon in enumerate(cac_lua_chon):
                lua_chon["label"] = LABELS[i]

            # Tìm nhãn của đáp án đúng
            dap_an_dung = next(lc for lc in cac_lua_chon if lc["la_dung"])
            dung_label  = dap_an_dung["label"]

            # ── Tạo phần INPUT: câu hỏi + 4 lựa chọn ─────────────────────────
            input_text = f"Sentence: {jp_co_blank}\n"
            for lc in cac_lua_chon:
                input_text += f"{lc['label']}) {lc['gp']['pattern']}\n"
            input_text = input_text.strip()

            # ── Tạo phần OUTPUT: đáp án + giải thích CoT ───────────────────────
            cot = "<thinking>\n"
            cot += f"1. Context Analysis: The full sentence translates to '{en}'. We need a grammar pattern that fits into the blank '（　）' to convey this meaning naturally.\n"
            cot += "2. Option Analysis:\n"
            
            for lc in cac_lua_chon:
                if lc["la_dung"]:
                    cot += f"   - {lc['label']}) {lc['gp']['pattern']}: Correct. This pattern means '{meaning_en}', which perfectly fits the intended meaning and context."
                    if meaning_detail:
                        detail_short = meaning_detail[:200] + "..." if len(meaning_detail) > 200 else meaning_detail
                        cot += f" ({detail_short})"
                    cot += "\n"
                else:
                    sai_gp = lc["gp"]
                    cot += f"   - {lc['label']}) {sai_gp['pattern']}: Incorrect. This pattern typically means '{sai_gp.get('meaning_en', 'different meaning')}', which does not fit the context.\n"
            
            cot += f"</thinking>\n<answer> {dung_label} </answer>"

            # JLPT instruction + CoT prompt
            instruction_text = (
                "問題: 次の文の（　）に入れるのに最もよいものを、1・2・3・4から一つえらびなさい。\n"
                "Solve this JLPT grammar question step-by-step. Provide your reasoning in a <thinking> block, and output the final number in an <answer> block."
            )

            results.append(make_qa(
                instruction = instruction_text,
                input_text  = input_text,
                output      = cot,
            ))

    print(f"  {dem_co_vi_du}/{len(grammar_points)} patterns co vi du cau")
    print(f"  Ket qua: {len(results):,} MCQ training examples")
    return results


# =============================================================================
# CHẠY TRỰC TIẾP: python scripts/clean_data/step1b_grammar_mcq.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 1B: Tao cau hoi trac nghiem 4 dap an (MCQ)")
    print("=" * 55)
    print()
    print("Day la dang cau hoi JLPT THAT:")
    print("  Cho cau co blank → chon 1 trong 4 mau ngu phap")
    print("  1 dap an dung + 3 dap an sai (distractors)")
    print()

    raw_data = load_json("grammar.json")
    records  = tao_mcq_grammar(raw_data)
    save_jsonl(records, "grammar_mcq.jsonl")

    # In ví dụ để kiểm tra
    print("\n--- Vi du 1 MCQ example ---")
    if records:
        s = records[0]
        print(f"\nInstruction:\n  {s['instruction']}")
        print(f"\nInput:\n{s['input']}")
        print(f"\nOutput:\n{s['output']}")

    print("\nHoan thanh!")
    print("File da tao: data/processed/grammar_mcq.jsonl")
    print()
    print("Buoc tiep theo: chay step4_combine.py (doi MODE thanh 'grammar_mcq')")

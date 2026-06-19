# step1_grammar.py
# =============================================================================
# BƯỚC 1: Xử lý grammar.json → tạo training examples về NGỮ PHÁP
#
# INPUT : data/raw/grammar.json  (595 điểm ngữ pháp N1-N5)
# OUTPUT: data/processed/grammar_qa.jsonl
#
# Mỗi điểm ngữ pháp sẽ tạo ra NHIỀU câu hỏi khác nhau, ví dụ:
#   - "Mẫu ngữ pháp X có nghĩa là gì?"
#   - "Cách chia mẫu X như thế nào?"
#   - "Dịch câu này sang tiếng Anh" (dùng câu ví dụ trong data)
#   - "X thuộc JLPT level nào?"
#
# Tại sao làm vậy? Vì model cần học được NHIỀU kiểu câu hỏi về cùng 1 chủ đề.
# Nếu chỉ hỏi 1 kiểu, model sẽ "thuộc lòng" thay vì "hiểu".
# =============================================================================

import sys
from pathlib import Path

# Thêm thư mục cha vào sys.path để import helpers.py
sys.path.insert(0, str(Path(__file__).parent))
from helpers import load_json, save_jsonl, make_qa


def xu_ly_ngu_phap(data: dict) -> list[dict]:
    """
    Đọc toàn bộ grammar points và tạo ra các Q&A training examples.

    Với MỖI điểm ngữ pháp, hàm này tạo ra tối đa 7 loại câu hỏi.
    """
    results = []  # danh sách tích lũy kết quả

    # grammar.json có cấu trúc: { "metadata": {...}, "grammar_points": [...] }
    # Ta chỉ cần phần "grammar_points"
    grammar_points = data.get("grammar_points", [])
    print(f"  Tim thay {len(grammar_points)} diem ngu phap (N1-N5)")

    for gp in grammar_points:
        # Lấy các trường quan trọng từ mỗi grammar point
        pattern         = gp.get("pattern", "")           # vd: "Noun + に難くない"
        level           = gp.get("level", "")             # vd: "N1"
        meaning_en      = gp.get("meaning_en", "")        # vd: "easy to imagine"
        meaning_detail  = gp.get("meaning_detailed", "")  # giải thích dài hơn
        formation       = gp.get("formation", "")         # cách chia động từ
        formation_notes = gp.get("formation_notes", [])   # ghi chú thêm
        formality       = gp.get("formality", "")         # "formal", "casual"...
        examples        = gp.get("examples", [])          # câu ví dụ tiếng Nhật
        related         = gp.get("related", [])           # các mẫu liên quan

        # Bỏ qua nếu thiếu thông tin cơ bản
        if not pattern or not meaning_en:
            continue

        # ─────────────────────────────────────────────────────────────────────
        # LOẠI CÂU HỎI 1: Nghĩa ngắn gọn
        # Model cần trả lời ngắn, súc tích về nghĩa của mẫu ngữ pháp
        # ─────────────────────────────────────────────────────────────────────
        results.append(make_qa(
            instruction = "日本語の文法パターンの意味を英語で説明してください。",
            input_text  = f"文法パターン: {pattern}",
            output      = meaning_en,
        ))

        # ─────────────────────────────────────────────────────────────────────
        # LOẠI CÂU HỎI 2: Giải thích chi tiết (có sắc thái, trường hợp đặc biệt)
        # Model học cách giải thích sâu hơn, phân biệt với mẫu tương tự
        # ─────────────────────────────────────────────────────────────────────
        if meaning_detail:
            results.append(make_qa(
                instruction = (
                    "Explain this Japanese grammar pattern in detail in English, "
                    "including nuance, usage, and when to use it."
                ),
                input_text  = f"Grammar pattern: {pattern} (JLPT {level})",
                output      = meaning_detail,
            ))

        # ─────────────────────────────────────────────────────────────────────
        # LOẠI CÂU HỎI 3: Cách chia / cấu trúc ngữ pháp
        # Ví dụ: "V-て form + ください" → model học cách ghép với các từ loại
        # formation_notes là list các ghi chú, ta nối lại bằng "; "
        # ─────────────────────────────────────────────────────────────────────
        if formation:
            notes = ""
            if formation_notes:
                notes = " Notes: " + "; ".join(formation_notes)
            results.append(make_qa(
                instruction = "How do you form this Japanese grammar pattern? Describe the grammatical structure.",
                input_text  = f"Grammar pattern: {pattern}",
                output      = formation + notes,
            ))

        # ─────────────────────────────────────────────────────────────────────
        # LOẠI CÂU HỎI 4: JLPT level
        # Câu hỏi đơn giản: "Mẫu X thuộc level nào?"
        # ─────────────────────────────────────────────────────────────────────
        results.append(make_qa(
            instruction = "What JLPT level is this Japanese grammar pattern?",
            input_text  = f"Grammar pattern: {pattern}",
            output      = f"This grammar pattern is JLPT {level} level.",
        ))

        # ─────────────────────────────────────────────────────────────────────
        # LOẠI CÂU HỎI 5: Formality (văn phong)
        # Ví dụ: "formal", "casual", "very_formal"
        # ─────────────────────────────────────────────────────────────────────
        if formality:
            results.append(make_qa(
                instruction = "What is the formality/register of this Japanese grammar pattern?",
                input_text  = f"Grammar pattern: {pattern}",
                output      = f"The register of {pattern} is: {formality}.",
            ))

        # ─────────────────────────────────────────────────────────────────────
        # LOẠI CÂU HỎI 6: Dịch câu ví dụ (Nhật → Anh)
        # Mỗi grammar point có 3-5 câu ví dụ, ta dùng tối đa 3 câu
        # ─────────────────────────────────────────────────────────────────────
        for ex in examples[:3]:
            jp_sentence = ex.get("japanese", "")
            en_sentence = ex.get("english", "")
            if jp_sentence and en_sentence:
                results.append(make_qa(
                    instruction = f"Translate this Japanese sentence that uses the grammar pattern [{pattern}].",
                    input_text  = jp_sentence,
                    output      = en_sentence,
                ))

        # ─────────────────────────────────────────────────────────────────────
        # LOẠI CÂU HỎI 7: Trắc nghiệm (giống đề JLPT thật)
        # Cho 2 lựa chọn: 1 đúng (pattern hiện tại) + 1 sai (pattern liên quan)
        # Đây là dạng câu hỏi thực tế trong bài thi JLPT
        # ─────────────────────────────────────────────────────────────────────
        if related and examples:
            ex = examples[0]
            jp = ex.get("japanese", "")
            en = ex.get("english", "")
            if jp and en:
                distractor = related[0]  # dùng mẫu liên quan làm đáp án sai
                results.append(make_qa(
                    instruction = (
                        "This is a JLPT grammar question. "
                        "Choose the correct grammar pattern and explain why."
                    ),
                    input_text  = (
                        f"Sentence: {jp}\n"
                        f"A) {pattern}\n"
                        f"B) {distractor}"
                    ),
                    output      = (
                        f"The correct answer is A) {pattern}.\n"
                        f"Reason: {meaning_en}. "
                        f"This is JLPT {level} ({formality} register).\n"
                        f"Translation: {en}"
                    ),
                ))

    print(f"  Ket qua: {len(results):,} training examples tu {len(grammar_points)} diem ngu phap")
    return results


# =============================================================================
# CHẠY FILE NÀY TRỰC TIẾP
# Ví dụ: python scripts/clean_data/step1_grammar.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 1: Xu ly grammar.json")
    print("=" * 55)

    raw_data = load_json("grammar.json")
    records  = xu_ly_ngu_phap(raw_data)
    save_jsonl(records, "grammar_qa.jsonl")

    # In thử 1 ví dụ để kiểm tra
    print("\n--- Vi du 1 training example (cau hoi dau tien) ---")
    sample = records[0]
    print(f"  instruction: {sample['instruction']}")
    print(f"  input      : {sample['input']}")
    print(f"  output     : {sample['output']}")
    print("\nHoan thanh! Xem ket qua o: data/processed/grammar_qa.jsonl")

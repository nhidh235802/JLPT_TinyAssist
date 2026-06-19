# step2_kanji.py
# =============================================================================
# BƯỚC 2: Xử lý kanji.json → tạo training examples về KANJI
#
# INPUT : data/raw/kanji.json  (13,108 kanji, nhưng ta chỉ lấy kanji có JLPT level)
# OUTPUT: data/processed/kanji_qa.jsonl
#
# Tại sao chỉ lấy kanji có JLPT level?
#   → kanji.json có 13,108 ký tự, nhưng JLPT N1-N5 chỉ test khoảng 2,136 kanji.
#   → Các kanji không có JLPT level là kanji cực hiếm / tên riêng,
#     không xuất hiện trong đề thi → không cần train.
#
# Mỗi kanji tạo ra 4 loại câu hỏi:
#   1. "Kanji X có nghĩa là gì?"
#   2. "Đọc kanji X như thế nào?" (on-yomi, kun-yomi)
#   3. "Kanji X thuộc JLPT level nào?"
#   4. "Cho tôi biết tất cả về kanji X" (tổng hợp)
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import load_json, save_jsonl, make_qa

# Chỉ giữ kanji thuộc các level này
JLPT_LEVELS = {"N1", "N2", "N3", "N4", "N5"}


def xu_ly_kanji(data: dict) -> list[dict]:
    """
    Đọc toàn bộ kanji và tạo training examples.
    Chỉ xử lý kanji có JLPT level (bỏ qua kanji không liên quan đến JLPT).
    """
    results = []

    # kanji.json có cấu trúc: { "metadata": {...}, "kanji": [...] }
    tat_ca_kanji = data.get("kanji", [])

    # Lọc chỉ lấy kanji JLPT
    # Trường "jlpt_waller" chứa level N1-N5 hoặc null (không thuộc JLPT)
    jlpt_kanji = [k for k in tat_ca_kanji if k.get("jlpt_waller") in JLPT_LEVELS]

    print(f"  Tong so kanji trong file : {len(tat_ca_kanji):,}")
    print(f"  Kanji co JLPT level      : {len(jlpt_kanji):,}  (se dung de train)")
    print(f"  Kanji khong co level     : {len(tat_ca_kanji) - len(jlpt_kanji):,}  (bo qua)")

    for kanji in jlpt_kanji:
        # Lấy thông tin từ mỗi kanji
        ky_tu       = kanji.get("character", "")         # ký tự kanji, vd: "愛"
        level       = kanji.get("jlpt_waller", "")       # "N1", "N2"...
        nghia       = kanji.get("meanings", {}).get("en", [])  # ["love", "affection"]
        on_yomi     = kanji.get("readings", {}).get("on", [])  # ["アイ"]  (cách đọc Hán)
        kun_yomi    = kanji.get("readings", {}).get("kun", []) # ["いと.しい"] (cách đọc Nhật)
        so_net      = kanji.get("stroke_count", "?")     # số nét bút

        # Bỏ qua nếu thiếu thông tin quan trọng
        if not ky_tu or not nghia:
            continue

        # Format lại cho dễ đọc
        # Giới hạn 5 nghĩa để output không quá dài
        nghia_str   = ", ".join(nghia[:5])
        on_str      = ", ".join(on_yomi)  if on_yomi  else "khong co"
        kun_str     = ", ".join(kun_yomi) if kun_yomi else "khong co"

        # ─────────────────────────────────────────────────────────────────────
        # CÂU HỎI 1: Nghĩa của kanji
        # Đây là câu hỏi đơn giản nhất, phổ biến trong JLPT
        # ─────────────────────────────────────────────────────────────────────
        results.append(make_qa(
            instruction = "What does this kanji mean? Provide the English meanings.",
            input_text  = f"Kanji: {ky_tu}",
            output      = f"Meanings: {nghia_str}",
        ))

        # ─────────────────────────────────────────────────────────────────────
        # CÂU HỎI 2: Cách đọc (on-yomi và kun-yomi)
        #
        # Lưu ý về format kun-yomi:
        #   "いと.しい" → dấu "." phân cách phần gốc và okurigana
        #   Ví dụ: 愛 đọc là "いとしい" nhưng viết là "いと.しい" trong từ điển
        # ─────────────────────────────────────────────────────────────────────
        results.append(make_qa(
            instruction = "How do you read this kanji? Provide the on-yomi (音読み) and kun-yomi (訓読み).",
            input_text  = f"Kanji: {ky_tu}",
            output      = (
                f"On-yomi  (音読み / Sino-Japanese) : {on_str}\n"
                f"Kun-yomi (訓読み / Native Japanese): {kun_str}"
            ),
        ))

        # ─────────────────────────────────────────────────────────────────────
        # CÂU HỎI 3: JLPT level
        # ─────────────────────────────────────────────────────────────────────
        results.append(make_qa(
            instruction = "What JLPT level is this kanji?",
            input_text  = f"Kanji: {ky_tu}",
            output      = (
                f"{ky_tu} is a JLPT {level} kanji.\n"
                f"Meanings: {nghia_str}\n"
                f"Stroke count: {so_net}"
            ),
        ))

        # ─────────────────────────────────────────────────────────────────────
        # CÂU HỎI 4: Tổng hợp thông tin (để review toàn diện)
        # Câu hỏi này cho model học cách tổng hợp nhiều thông tin cùng lúc
        # ─────────────────────────────────────────────────────────────────────
        results.append(make_qa(
            instruction = "Give me a complete study summary of this kanji for JLPT preparation.",
            input_text  = f"Kanji: {ky_tu}",
            output      = (
                f"Kanji       : {ky_tu}\n"
                f"JLPT Level  : {level}\n"
                f"Meanings    : {nghia_str}\n"
                f"On-yomi     : {on_str}\n"
                f"Kun-yomi    : {kun_str}\n"
                f"Stroke count: {so_net}"
            ),
        ))

    print(f"  Ket qua: {len(results):,} training examples")
    return results


# =============================================================================
# CHẠY FILE NÀY TRỰC TIẾP
# Ví dụ: python scripts/clean_data/step2_kanji.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 2: Xu ly kanji.json")
    print("=" * 55)

    raw_data = load_json("kanji.json")
    records  = xu_ly_kanji(raw_data)
    save_jsonl(records, "kanji_qa.jsonl")

    # In thử ví dụ
    print("\n--- Vi du (kanji dau tien trong ket qua) ---")
    sample = records[0]
    print(f"  instruction: {sample['instruction']}")
    print(f"  input      : {sample['input']}")
    print(f"  output     : {sample['output']}")
    print("\nHoan thanh! Xem ket qua o: data/processed/kanji_qa.jsonl")

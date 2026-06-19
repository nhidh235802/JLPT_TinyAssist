# step3_words.py
# =============================================================================
# BƯỚC 3: Xử lý words.json → tạo training examples về TỪ VỰNG
#
# INPUT : data/raw/words.json  (23,119 từ phổ biến, nhưng chỉ ~7,747 có JLPT level)
# OUTPUT: data/processed/words_qa.jsonl
#
# Cấu trúc của words.json phức tạp hơn grammar và kanji.
# Mỗi từ có thể có:
#   - Nhiều cách viết kanji (ví dụ: "見る" và "観る" đều là "miru")
#   - Nhiều cách đọc kana (ví dụ: "あそこ" và "あすこ")
#   - Nhiều nghĩa (sense) khác nhau (ví dụ: "橋" có thể là "bridge" hoặc tên người)
#   - Câu ví dụ từ Tatoeba (trang web câu ví dụ miễn phí)
#
# Ví dụ cấu trúc 1 từ trong words.json:
#   {
#     "id": "1234567",
#     "kanji": [{"common": true, "text": "勉強"}],
#     "kana":  [{"common": true, "text": "べんきょう"}],
#     "sense": [
#       {
#         "partOfSpeech": ["vs"],        ← loại từ (suru-verb)
#         "gloss": [{"lang": "eng", "text": "study"}],  ← nghĩa tiếng Anh
#         "examples": [{"japanese": "...", "english": "..."}]
#       }
#     ],
#     "jlpt_waller": "N5"
#   }
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import load_json, save_jsonl, make_qa

JLPT_LEVELS = {"N1", "N2", "N3", "N4", "N5"}

# Bảng dịch viết tắt loại từ (POS = Part Of Speech) sang tiếng Anh dễ hiểu
# Đây là các tag trong JMdict (từ điển tiếng Nhật mở)
POS_MAP = {
    "n"     : "noun (danh từ)",
    "v1"    : "Ichidan verb - verbs ending in -る (động từ nhóm 2)",
    "v5r"   : "Godan verb ending in -る (động từ nhóm 1)",
    "v5k"   : "Godan verb ending in -く",
    "v5s"   : "Godan verb ending in -す",
    "v5t"   : "Godan verb ending in -つ",
    "v5n"   : "Godan verb ending in -ぬ",
    "v5b"   : "Godan verb ending in -ぶ",
    "v5m"   : "Godan verb ending in -む",
    "v5g"   : "Godan verb ending in -ぐ",
    "v5u"   : "Godan verb ending in -う",
    "vk"    : "irregular verb くる (kuru)",
    "vs"    : "suru-verb (する verb, dùng với する)",
    "vs-i"  : "suru-verb (built-in する)",
    "adj-i" : "i-adjective (tính từ đuôi -い)",
    "adj-na": "na-adjective (tính từ dùng với な)",
    "adv"   : "adverb (trạng từ)",
    "prt"   : "particle (trợ từ như は、が、を...)",
    "conj"  : "conjunction (liên từ như そして、でも...)",
    "pn"    : "pronoun (đại từ như 私、あなた...)",
    "exp"   : "expression / set phrase (thành ngữ, cụm cố định)",
    "aux-v" : "auxiliary verb (động từ phụ trợ như ている、ている...)",
    "int"   : "interjection (thán từ như あ、ええ...)",
}


def xu_ly_tu_vung(data: dict) -> list[dict]:
    """
    Đọc toàn bộ words và tạo training examples.
    Chỉ xử lý từ có JLPT level.
    """
    results = []

    tat_ca_tu = data.get("words", [])

    # Lọc chỉ lấy từ có JLPT level
    jlpt_tu = [w for w in tat_ca_tu if w.get("jlpt_waller") in JLPT_LEVELS]

    print(f"  Tong so tu trong file : {len(tat_ca_tu):,}")
    print(f"  Tu co JLPT level      : {len(jlpt_tu):,}  (se dung)")
    print(f"  Tu khong co level     : {len(tat_ca_tu) - len(jlpt_tu):,}  (bo qua)")

    for tu in jlpt_tu:
        level         = tu.get("jlpt_waller", "")
        kanji_list    = tu.get("kanji", [])  # danh sách cách viết kanji
        kana_list     = tu.get("kana", [])   # danh sách cách đọc kana
        sense_list    = tu.get("sense", [])  # danh sách nghĩa

        if not sense_list:
            continue

        # ── Xác định cách viết chính ──────────────────────────────────────────
        # Ưu tiên: cách viết kanji phổ biến (common=True)
        # Nếu không có, dùng cách đọc kana phổ biến
        kanji_chinh = next(
            (k["text"] for k in kanji_list if k.get("common")),
            None
        )
        kana_chinh = next(
            (k["text"] for k in kana_list if k.get("common")),
            None
        )

        # Tu hien thi la dang chinh cua tu
        tu_hien_thi = kanji_chinh or kana_chinh
        if not tu_hien_thi:
            continue

        # Xử lý từng nghĩa (lấy tối đa 2 nghĩa đầu)
        # Một từ có thể có nhiều nghĩa rất khác nhau, ví dụ:
        #   "橋" nghĩa 1: bridge (cây cầu)
        #   "橋" nghĩa 2: họ người Nhật
        for sense in sense_list[:2]:
            # Lấy nghĩa tiếng Anh (lọc theo lang="eng")
            nghia_list = [
                g["text"] for g in sense.get("gloss", [])
                if g.get("lang") == "eng"
            ]
            if not nghia_list:
                continue

            pos_tags = sense.get("partOfSpeech", [])  # loại từ
            vi_du    = sense.get("examples", [])       # câu ví dụ

            # Format output
            nghia_str = ", ".join(nghia_list[:5])
            # Dịch tag loại từ sang tên dài hơn nếu có trong POS_MAP
            pos_str = ", ".join(POS_MAP.get(p, p) for p in pos_tags[:2])
            doc_str = kana_chinh or "N/A"

            # ─────────────────────────────────────────────────────────────────
            # CÂU HỎI 1: Từ này có nghĩa là gì?
            # ─────────────────────────────────────────────────────────────────
            results.append(make_qa(
                instruction = "What does this Japanese word mean in English?",
                input_text  = f"Word: {tu_hien_thi}",
                output      = nghia_str,
            ))

            # ─────────────────────────────────────────────────────────────────
            # CÂU HỎI 2: Đọc từ này như thế nào?
            # Chỉ tạo câu hỏi này khi từ có dạng kanji (vì kana tự đọc được)
            # ─────────────────────────────────────────────────────────────────
            if kanji_chinh and kana_chinh:
                results.append(make_qa(
                    instruction = "How do you read this Japanese word? Give the hiragana/katakana reading.",
                    input_text  = f"Word: {kanji_chinh}",
                    output      = f"Reading: {kana_chinh}\nMeaning: {nghia_str}",
                ))

            # ─────────────────────────────────────────────────────────────────
            # CÂU HỎI 3: Thông tin JLPT đầy đủ
            # Câu hỏi tổng hợp: từ này ở đâu trong JLPT? Loại từ là gì?
            # ─────────────────────────────────────────────────────────────────
            results.append(make_qa(
                instruction = "Give me JLPT study information for this Japanese word.",
                input_text  = f"Word: {tu_hien_thi}",
                output      = (
                    f"Word        : {tu_hien_thi}\n"
                    f"Reading     : {doc_str}\n"
                    f"Meaning     : {nghia_str}\n"
                    f"Part of speech: {pos_str}\n"
                    f"JLPT Level  : {level}"
                ),
            ))

            # ─────────────────────────────────────────────────────────────────
            # CÂU HỎI 4: Dịch câu ví dụ (nếu có)
            # Câu ví dụ lấy từ Tatoeba - trang web câu ví dụ song ngữ Nhật-Anh
            # ─────────────────────────────────────────────────────────────────
            for ex in vi_du[:1]:   # chỉ lấy 1 câu ví dụ để tránh lặp quá nhiều
                jp_cau = ex.get("japanese", "")
                en_cau = ex.get("english", "")
                if jp_cau and en_cau:
                    results.append(make_qa(
                        instruction = "Translate this Japanese sentence to English.",
                        input_text  = jp_cau,
                        output      = en_cau,
                    ))

    print(f"  Ket qua: {len(results):,} training examples")
    return results


# =============================================================================
# CHẠY FILE NÀY TRỰC TIẾP
# Ví dụ: python scripts/clean_data/step3_words.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 3: Xu ly words.json")
    print("=" * 55)

    raw_data = load_json("words.json")
    records  = xu_ly_tu_vung(raw_data)
    save_jsonl(records, "words_qa.jsonl")

    print("\n--- Vi du (tu dau tien trong ket qua) ---")
    sample = records[0]
    print(f"  instruction: {sample['instruction']}")
    print(f"  input      : {sample['input']}")
    print(f"  output     : {sample['output']}")
    print("\nHoan thanh! Xem ket qua o: data/processed/words_qa.jsonl")

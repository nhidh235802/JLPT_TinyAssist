# step5_check.py
# =============================================================================
# BƯỚC 5: KIỂM TRA DATA ĐÃ CLEAN
#
# File này KHÔNG cần model, chỉ đọc file JSONL và kiểm tra:
#   1. Data có đúng format không? (instruction/input/output)
#   2. Phân phối các loại câu hỏi như thế nào?
#   3. Output trung bình dài bao nhiêu?
#   4. In vài ví dụ để đọc bằng mắt
#
# Đây là bước QUAN TRỌNG trước khi train: nếu data bẩn → model sẽ học sai.
# Luôn kiểm tra data bằng mắt trước khi train!
#
# Chạy: python scripts/clean_data/step5_check.py
# =============================================================================

import sys
import json
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from helpers import OUT_DIR


def doc_jsonl(filename: str, gioi_han: int = None) -> list[dict]:
    """
    Đọc file JSONL, trả về list.
    gioi_han: nếu set, chỉ đọc tối đa n dòng (để kiểm tra nhanh file lớn)
    """
    path = OUT_DIR / filename
    if not path.exists():
        print(f"  [!] Khong tim thay: {filename}")
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if gioi_han and i >= gioi_han:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def kiem_tra_format(records: list[dict], ten_file: str):
    """
    Kiểm tra xem mỗi record có đủ 3 trường bắt buộc không:
    instruction, input, output

    Nếu thiếu → báo lỗi ngay, không để lọt vào training.
    """
    loi = 0
    for i, r in enumerate(records):
        if not r.get("instruction"):
            print(f"  [LOI] Record #{i}: thieu 'instruction'")
            loi += 1
        if "input" not in r:
            print(f"  [LOI] Record #{i}: thieu 'input'")
            loi += 1
        if not r.get("output"):
            print(f"  [LOI] Record #{i}: thieu 'output'")
            loi += 1

    if loi == 0:
        print(f"  [OK] {ten_file}: tat ca {len(records):,} records deu hop le")
    else:
        print(f"  [!!] {ten_file}: co {loi} loi can sua!")

    return loi == 0


def phan_tich_phan_phoi(records: list[dict]) -> dict:
    """
    Phân loại các câu hỏi theo kiểu (dựa vào từ khóa trong instruction).

    Tại sao quan trọng?
    → Nếu 90% data là "translate" và chỉ 10% là câu hỏi grammar,
      model sẽ giỏi dịch nhưng kém trả lời về JLPT level, formality...
    → Cần data phân phối đều các loại task.
    """
    phan_loai = defaultdict(int)
    do_dai_output = []

    for r in records:
        instr = r.get("instruction", "").lower()
        out   = r.get("output", "")
        do_dai_output.append(len(out.split()))

        # Phân loại dựa theo từ khóa
        if "translate" in instr:
            phan_loai["Dich thuat (Nhat->Anh)"] += 1
        elif "jlpt level" in instr or "what jlpt" in instr:
            phan_loai["JLPT level"] += 1
        elif "formality" in instr or "register" in instr:
            phan_loai["Van phong (formal/casual)"] += 1
        elif "form" in instr or "structure" in instr:
            phan_loai["Cau truc ngu phap"] += 1
        elif "read" in instr or "hiragana" in instr or "yomi" in instr:
            phan_loai["Cach doc (yomi)"] += 1
        elif "mean" in instr:
            phan_loai["Nghia (meaning)"] += 1
        elif "explain" in instr or "detail" in instr:
            phan_loai["Giai thich chi tiet"] += 1
        elif "summary" in instr or "overview" in instr or "study" in instr:
            phan_loai["Tong hop (summary)"] += 1
        elif "問題1" in instr:
            phan_loai["Trac nghiem Mon 1 (Reading)"] += 1
        elif "問題2" in instr:
            phan_loai["Trac nghiem Mon 2 (Writing)"] += 1
        elif "問題3" in instr:
            phan_loai["Trac nghiem Mon 3 (Fill-in)"] += 1
        elif "問題4" in instr:
            phan_loai["Trac nghiem Mon 4 (Synonyms)"] += 1
        elif "問題5" in instr:
            phan_loai["Trac nghiem Mon 5 (Usage)"] += 1
        elif "問題" in instr or "grammar question" in instr or "choose" in instr:
            phan_loai["Trac nghiem Ngu phap"] += 1
        else:
            phan_loai["Khac"] += 1

    # Tính độ dài trung bình output
    trung_binh = sum(do_dai_output) / len(do_dai_output) if do_dai_output else 0

    return {
        "phan_loai": dict(phan_loai),
        "trung_binh_do_dai_output": round(trung_binh, 1),
        "output_ngan_nhat": min(do_dai_output) if do_dai_output else 0,
        "output_dai_nhat": max(do_dai_output) if do_dai_output else 0,
    }


def in_vi_du(records: list[dict], so_luong: int = 3, ngau_nhien: bool = True):
    """
    In một số ví dụ để đọc bằng mắt.
    Đây là bước quan trọng để phát hiện data bẩn mà code không phát hiện được.
    """
    print(f"\n  --- {so_luong} vi du {'ngau nhien' if ngau_nhien else 'dau tien'} ---")

    if ngau_nhien and len(records) > so_luong:
        random.seed(0)
        mau = random.sample(records, so_luong)
    else:
        mau = records[:so_luong]

    for i, r in enumerate(mau, 1):
        print(f"\n  [{i}]")
        print(f"  instruction : {r.get('instruction', '')[:90]}")
        print(f"  input       : {r.get('input', '')[:80]}")
        # In 300 ký tự đầu tiên để hiển thị được block <thinking> của CoT
        output_preview = r.get('output', '')[:300]
        if len(r.get('output', '')) > 300:
            output_preview += "..."
        # Thêm xuống dòng cho dễ đọc CoT
        output_preview = output_preview.replace("\\n", "\n              ")
        print(f"  output      :\n              {output_preview}")


def kiem_tra_file(filename: str):
    """Chạy toàn bộ kiểm tra cho 1 file JSONL."""
    print(f"\n{'='*55}")
    print(f"  Kiem tra: {filename}")
    print(f"{'='*55}")

    records = doc_jsonl(filename)
    if not records:
        return

    print(f"  So luong records: {len(records):,}")

    # Kiểm tra format
    ok = kiem_tra_format(records, filename)

    # Phân tích phân phối
    if ok:
        phan_tich = phan_tich_phan_phoi(records)

        print(f"\n  Phan phoi loai cau hoi:")
        for loai, dem in sorted(phan_tich["phan_loai"].items(), key=lambda x: -x[1]):
            phan_tram = 100 * dem / len(records)
            print(f"    {loai:<40}: {dem:>6,} ({phan_tram:.1f}%)")

        print(f"\n  Do dai output (tinh theo so tu):")
        print(f"    Trung binh : {phan_tich['trung_binh_do_dai_output']} tu")
        print(f"    Ngan nhat  : {phan_tich['output_ngan_nhat']} tu")
        print(f"    Dai nhat   : {phan_tich['output_dai_nhat']} tu")

        # In vài ví dụ
        in_vi_du(records, so_luong=2, ngau_nhien=True)


# =============================================================================
# CHẠY FILE NÀY TRỰC TIẾP
# Ví dụ: python scripts/clean_data/step5_check.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 5: Kiem tra data da clean")
    print("=" * 55)
    print("File nay khong can model - chi kiem tra data.")

    # Kiểm tra từng file nguồn
    kiem_tra_file("grammar_qa.jsonl")
    kiem_tra_file("grammar_mcq.jsonl")
    kiem_tra_file("kanji_qa.jsonl")
    kiem_tra_file("words_qa.jsonl")
    kiem_tra_file("kanji_reading_mcq.jsonl")
    kiem_tra_file("kanji_writing_mcq.jsonl")
    kiem_tra_file("vocab_fill_mcq.jsonl")
    kiem_tra_file("vocab_synonym_mcq.jsonl")
    kiem_tra_file("vocab_usage_mcq.jsonl")

    # Kiểm tra file tổng hợp
    kiem_tra_file("combined_train.jsonl")
    kiem_tra_file("combined_eval.jsonl")

    print("\n" + "=" * 55)
    print("Ket luan:")
    print("  - Neu tat ca [OK] → data san sang de train")
    print("  - Neu co [LOI]   → sua lai step1/2/3 truoc")
    print("  - Doc cac vi du bang mat → dam bao noi dung hop ly")
    print("=" * 55)

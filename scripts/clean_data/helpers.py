# helpers.py
# =============================================================================
# Các hàm DÙNG CHUNG cho tất cả các bước xử lý data
#
# File này không chạy trực tiếp, chỉ import vào các file step1, step2...
# =============================================================================

import sys
import io
import json
import re
from pathlib import Path

# Fix lỗi hiển thị tiếng Nhật trên Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# Tất cả các file đều dùng chung 2 thư mục này:
#   - RAW_DIR : chứa data gốc tải về (grammar.json, kanji.json, words.json)
#   - OUT_DIR : chứa data đã xử lý (output sau khi clean)
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent.parent   # thư mục gốc ML_DL/
RAW_DIR  = ROOT_DIR / "data" / "raw"             # data/raw/
OUT_DIR  = ROOT_DIR / "data" / "processed"       # data/processed/
OUT_DIR.mkdir(parents=True, exist_ok=True)        # tạo thư mục nếu chưa có


# =============================================================================
# HÀM ĐỌC FILE JSON
# Dùng để load grammar.json, kanji.json, words.json
# =============================================================================

def load_json(filename: str) -> dict:
    """
    Đọc file JSON từ thư mục data/raw/.

    Ví dụ: load_json("grammar.json") sẽ đọc data/raw/grammar.json
    Trả về: dict (toàn bộ nội dung file dưới dạng Python dict)
    """
    path = RAW_DIR / filename
    size_mb = path.stat().st_size // 1024 // 1024
    print(f"  Dang doc {filename} ({size_mb} MB)...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# HÀM GHI FILE JSONL
#
# JSONL = JSON Lines = mỗi dòng là 1 JSON object
# Đây là format chuẩn để train LLM, vì:
#   - Đọc từng dòng rất nhanh (không cần load cả file vào RAM)
#   - Mỗi dòng là 1 training example độc lập
#
# Ví dụ 1 dòng trong file JSONL:
#   {"instruction": "What does に難くない mean?", "input": "...", "output": "..."}
# =============================================================================

def save_jsonl(records: list[dict], filename: str):
    """
    Lưu danh sách records ra file JSONL trong thư mục data/processed/.

    Ví dụ: save_jsonl(my_list, "grammar_qa.jsonl")
    """
    path = OUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            # ensure_ascii=False để giữ nguyên chữ Nhật (không bị encode thành \u1234...)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  [Luu] {len(records):,} examples -> data/processed/{filename}")
    return path


# =============================================================================
# HÀM TẠO 1 TRAINING EXAMPLE
#
# Format "Alpaca" - đây là format phổ biến nhất để fine-tune LLM:
#   {
#     "instruction": "câu lệnh cho model" (KHÔNG thay đổi theo từng ví dụ),
#     "input":       "dữ liệu đầu vào cụ thể" (thay đổi theo từng ví dụ),
#     "output":      "câu trả lời đúng mà model cần học"
#   }
#
# Khi fine-tune, model sẽ học: "khi thấy instruction + input này, phải trả lời output này"
# =============================================================================

def make_qa(instruction: str, input_text: str, output: str) -> dict:
    """
    Tạo 1 training example theo format Alpaca.

    instruction : nhiệm vụ (ví dụ: "Translate this sentence")
    input_text  : dữ liệu vào (ví dụ: "彼女は走った。")
    output      : câu trả lời đúng (ví dụ: "She ran.")
    """
    def clean(s: str) -> str:
        """Xóa khoảng trắng thừa đầu/cuối và giữa các từ."""
        return re.sub(r"\s+", " ", s.strip()) if s else ""

    return {
        "instruction": clean(instruction),
        "input":       clean(input_text),
        "output":      clean(output),
    }

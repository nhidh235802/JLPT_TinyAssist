"""
Script 04: Sinh dữ liệu tổng hợp (Synthetic Data Generation) cho Train V2.

Dùng Gemini API để tự động tạo ra câu hỏi JLPT cho 2 dạng bài khó nhất:
  - Dạng Bài 2: Sắp xếp câu (Dấu sao *) - Tìm từ ở vị trí thứ 3 trong 4 mảnh
  - Dạng Bài 3: Đoạn văn đục lỗ - Chọn từ nối / ngữ pháp điền vào chỗ trống

Nguyên liệu gốc: data/raw/grammar.json (595 cấu trúc ngữ pháp N5~N1)
Kết quả: data/processed/synthetic_part2.jsonl và synthetic_part3.jsonl

Cách chạy:
    # Chạy thử 10 câu Bài 2 và 5 đoạn văn Bài 3 (để kiểm tra chất lượng)
    python scripts/04_generate_synthetic.py --mode test

    # Chạy full (sinh toàn bộ data)
    python scripts/04_generate_synthetic.py --mode full

    # Chỉ sinh 1 dạng bài
    python scripts/04_generate_synthetic.py --mode full --part 2
    python scripts/04_generate_synthetic.py --mode full --part 3

Yêu cầu:
    pip install google-generativeai python-dotenv tqdm
"""

import json
import time
import random
import argparse
import sys
import io
from pathlib import Path
from dotenv import load_dotenv
import os

# Fix lỗi hiển thị tiếng Nhật trên Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─── CẤU HÌNH ĐƯỜNG DẪN ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRAMMAR_FILE    = RAW_DIR / "grammar.json"
OUTPUT_PART2    = OUT_DIR / "synthetic_part2.jsonl"
OUTPUT_PART3    = OUT_DIR / "synthetic_part3.jsonl"

# ─── CẤU HÌNH API ────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY", "")
DELAY_SECONDS = 2


# ─── 1. TẢI DATA GRAMMAR GỐC ─────────────────────────────────────────────────

def load_grammar_points() -> list[dict]:
    """Đọc file grammar.json và trả về danh sách các điểm ngữ pháp."""
    print(f"Đang đọc {GRAMMAR_FILE}...")
    with open(GRAMMAR_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("grammar_points", [])
    print(f"  Đã load {len(points)} điểm ngữ pháp (N5~N1).")
    return points


# ─── 2. PROMPT TEMPLATES ─────────────────────────────────────────────────────

PROMPT_PART2 = """Bạn là giáo viên tiếng Nhật JLPT chuyên nghiệp. Hãy tạo ra 1 câu hỏi dạng Bài 2 (sắp xếp câu - tìm từ ở vị trí ★) theo đúng định dạng thi JLPT thật.

THÔNG TIN NGỮ PHÁP:
- Cấu trúc: {pattern}
- Cấp độ: {level}
- Ý nghĩa: {meaning_en}
- Ví dụ câu: {example_jp} ({example_en})

QUY TẮC TẠO CÂU HỎI:
1. Viết 1 câu tiếng Nhật tự nhiên sử dụng cấu trúc ngữ pháp này. Câu phải có ngữ cảnh rõ ràng.
2. Chọn 1 đoạn trong câu và chia thành ĐÚNG 4 mảnh (A, B, C, D).
3. Các mảnh khi ghép lại đúng thứ tự sẽ tạo thành đoạn câu hoàn chỉnh.
4. Đáp án là mảnh ở VỊ TRÍ SỐ 3 (mảnh thứ 3 trong thứ tự đúng).
5. Viết phần <thinking> giải thích: nghĩa câu, tại sao thứ tự đó đúng về mặt ngữ pháp.

ĐỊNH DẠNG OUTPUT (JSON THUẦN TÚY, không có markdown):
{{
  "full_sentence": "câu hoàn chỉnh tiếng Nhật",
  "sentence_with_blank": "phần đầu câu ＿＿＿★＿＿＿ phần cuối câu",
  "choices": {{
    "1": "mảnh A",
    "2": "mảnh B",
    "3": "mảnh C (ĐÂY LÀ ĐÁP ÁN - VỊ TRÍ ★)",
    "4": "mảnh D"
  }},
  "correct_order": [số thứ tự mảnh 1, số thứ tự mảnh 2, số thứ tự mảnh 3★, số thứ tự mảnh 4],
  "answer": "số của mảnh ở vị trí ★",
  "thinking": "Phân tích: [nghĩa câu]. Về thứ tự: [giải thích ngữ pháp chi tiết tại sao phải ghép theo đúng thứ tự đó]. Loại trừ: [giải thích tại sao các thứ tự khác sai].",
  "level": "{level}"
}}"""


PROMPT_PART3 = """Bạn là giáo viên tiếng Nhật JLPT chuyên nghiệp. Hãy tạo ra 1 đoạn văn ngắn theo đúng định dạng bài đọc Bài 3 của kỳ thi JLPT thật.

CHỦ ĐỀ GỢI Ý: {topic}
CẤP ĐỘ: {level}

QUY TẮC:
1. Viết đoạn văn tiếng Nhật khoảng 80-120 chữ. Văn phong rõ ràng, logic mạch lạc.
2. Tạo ĐÚNG 3 lỗ trống (đánh số {lp1}, {lp2}, {lp3}) ở những vị trí cần từ nối, phó từ, hoặc cấu trúc ngữ pháp.
3. Với MỖI lỗ trống, tạo 4 lựa chọn (A/B/C/D). Chỉ 1 đáp án đúng, 3 đáp án sai phải nghe "hợp lý" nhưng sai về ngữ cảnh.
4. Viết phần <thinking> giải thích từng lỗ trống dựa vào ngữ cảnh câu trước/sau.

ĐỊNH DẠNG OUTPUT (JSON THUẦN TÚY, không có markdown):
{{
  "passage": "đoạn văn tiếng Nhật với ({lp1}) ({lp2}) ({lp3}) là chỗ trống",
  "questions": [
    {{
      "blank_id": {lp1},
      "choices": {{"1": "lựa chọn 1", "2": "lựa chọn 2", "3": "lựa chọn 3", "4": "lựa chọn 4"}},
      "answer": "số đáp án đúng",
      "thinking": "Phân tích ngữ cảnh: [câu trước nói gì]. Câu này cần từ [loại từ] vì [lý do]. Các đáp án sai [X] vì [lý do]."
    }},
    {{
      "blank_id": {lp2},
      "choices": {{"1": "...", "2": "...", "3": "...", "4": "..."}},
      "answer": "...",
      "thinking": "..."
    }},
    {{
      "blank_id": {lp3},
      "choices": {{"1": "...", "2": "...", "3": "...", "4": "..."}},
      "answer": "...",
      "thinking": "..."
    }}
  ],
  "level": "{level}"
}}"""


# ─── 3. HÀM GỌI GEMINI API ────

def init_gemini(api_key: str):
    """Khởi tạo client Gemini dùng SDK mới (google.genai)."""
    from google import genai
    client = genai.Client(api_key=api_key)
    return client

GENERATE_MODEL = "gemini-2.0-flash-lite"


def call_gemini(client, prompt: str, retries: int = 3) -> str | None:
    """
    Gọi Gemini API với cơ chế thử lại (retry) khi bị lỗi mạng hoặc rate limit.
    Trả về chuỗi text từ Gemini, hoặc None nếu thất bại sau nhiều lần thử.
    """
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GENERATE_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            print(f"  [Lần {attempt+1}/{retries}] Lỗi gọi API: {e}")
            # Nếu bị 429 (Rate Limit), đợi lâu hơn theo gợi ý của API
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 35  # Đợi 35s như API gợi ý
            else:
                wait = (attempt + 1) * 10
            if attempt < retries - 1:
                print(f"  Đợi {wait}s rồi thử lại...")
                time.sleep(wait)
    return None


def parse_json_response(text: str) -> dict | None:
    """Bóc tách JSON từ phản hồi của Gemini (có thể lẫn markdown)."""
    if not text:
        return None
    # Xóa markdown code block nếu có
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Thử tìm phần JSON trong chuỗi
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


# ─── 4. SINH DỮ LIỆU BÀI 2 (DẤU SAO) ──────

def generate_part2_record(gemini_model, grammar_point: dict) -> dict | None:
    """
    Gọi Gemini để tạo 1 câu hỏi dạng Bài 2 từ 1 điểm ngữ pháp.
    Trả về dict training record (Alpaca format) hoặc None nếu thất bại.
    """
    # Lấy 1 câu ví dụ ngẫu nhiên từ điểm ngữ pháp này
    examples = grammar_point.get("examples", [])
    if not examples:
        return None
    ex = random.choice(examples)

    prompt = PROMPT_PART2.format(
        pattern    = grammar_point.get("pattern", ""),
        level      = grammar_point.get("level", "N3"),
        meaning_en = grammar_point.get("meaning_en", ""),
        example_jp = ex.get("japanese", ""),
        example_en = ex.get("english", ""),
    )

    raw = call_gemini(gemini_model, prompt)
    data = parse_json_response(raw)
    if not data:
        return None

    # Đóng gói thành định dạng Alpaca chuẩn để train
    level = data.get("level", grammar_point.get("level", "N3"))
    sentence_with_blank = data.get("sentence_with_blank", "")
    choices = data.get("choices", {})
    answer  = str(data.get("answer", ""))
    thinking = data.get("thinking", "")

    if not all([sentence_with_blank, choices, answer, thinking]):
        return None

    # Xây dựng chuỗi câu hỏi
    choices_str = "\n".join([f"{k}) {v}" for k, v in choices.items()])
    instruction = (
        f"問題2: 次の文の（ ）に入れるのに最もよいものを、1・2・3・4から一つえらびなさい。"
        f" Arrange the word fragments to fill the blank marked ★. "
        f"Provide your reasoning in a <thinking> block and the number of the ★ fragment in an <answer> block."
    )
    input_text = f"Sentence: {sentence_with_blank}\nFragments:\n{choices_str}"
    output_text = f"<thinking>\n{thinking}\n</thinking>\n<answer>{answer}</answer>"

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": {"type": "grammar_part2_star", "level": level, "pattern": grammar_point.get("pattern", "")}
    }


# ─── 5. SINH DỮ LIỆU BÀI 3 (ĐOẠN VĂN) ──────

TOPICS = [
    "công nghệ và cuộc sống hiện đại", "môi trường và biến đổi khí hậu",
    "giáo dục và học tập", "sức khỏe và lối sống", "giao thông đô thị",
    "thói quen ăn uống của người Nhật", "du lịch và văn hóa địa phương",
    "kinh tế và công việc", "mạng xã hội và truyền thông", "gia đình và xã hội",
    "nghệ thuật và âm nhạc", "thể thao và rèn luyện thân thể",
    "thiên nhiên và mùa màng ở Nhật Bản", "tình nguyện và cộng đồng",
]

LEVELS = ["N5", "N4", "N3", "N3", "N2", "N2", "N1"]  # Tỷ lệ: N3/N2 nhiều hơn

def generate_part3_records(gemini_model, topic: str, level: str) -> list[dict]:
    """
    Gọi Gemini để tạo 1 đoạn văn với 3 lỗ trống.
    Trả về list gồm 3 training records (1 record per blank).
    """
    blank_ids = random.sample(range(30, 50), 3)  # Số thứ tự lỗ trống ngẫu nhiên
    blank_ids.sort()

    prompt = PROMPT_PART3.format(
        topic=topic, level=level,
        lp1=blank_ids[0], lp2=blank_ids[1], lp3=blank_ids[2]
    )

    raw = call_gemini(gemini_model, prompt)
    data = parse_json_response(raw)
    if not data:
        return []

    passage   = data.get("passage", "")
    questions = data.get("questions", [])
    if not passage or len(questions) < 3:
        return []

    records = []
    for q in questions:
        blank_id = q.get("blank_id", "")
        choices  = q.get("choices", {})
        answer   = str(q.get("answer", ""))
        thinking = q.get("thinking", "")

        if not all([blank_id, choices, answer, thinking]):
            continue

        choices_str = "  ".join([f"{k} {v}" for k, v in choices.items()])
        instruction = (
            f"問題3: 次の文章を読んで、文章全体の内容を考えて、（{blank_id}）に入る最もよいものを"
            f"1・2・3・4から一つえらびなさい。"
            f" Choose the best word/phrase for blank ({blank_id}) based on context. "
            f"Provide your reasoning in a <thinking> block and the answer number in an <answer> block."
        )
        input_text  = f"Passage:\n{passage}\n\nBlank ({blank_id}) choices: {choices_str}"
        output_text = f"<thinking>\n{thinking}\n</thinking>\n<answer>{answer}</answer>"

        records.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {"type": "grammar_part3_passage", "level": level, "topic": topic, "blank_id": blank_id}
        })

    return records


# ─── 6. VÒNG LẶP CHÍNH ─────

def run_part2(gemini_model, grammar_points: list, count: int, output_file: Path):
    """Sinh dữ liệu Bài 2 và ghi vào file JSONL."""
    print(f"\n{'='*55}")
    print(f"  BẮT ĐẦU SINH DỮ LIỆU BÀI 2 (DẤU SAO) - {count} câu")
    print(f"{'='*55}")

    # Xáo trộn ngẫu nhiên để lấy đủ các level khác nhau
    pool = grammar_points.copy()
    random.shuffle(pool)
    selected = pool[:count]

    success = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for i, gp in enumerate(selected, 1):
            print(f"  [{i}/{count}] Đang tạo câu từ: {gp.get('pattern', '')} ({gp.get('level', '')})")
            record = generate_part2_record(gemini_model, gp)
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                success += 1
                print(f"    ✅ Thành công! ({success} câu đã lưu)")
            else:
                print(f"    ❌ Thất bại, bỏ qua.")
            time.sleep(DELAY_SECONDS)

    print(f"\nKết quả Bài 2: {success}/{count} câu thành công → {output_file}")
    return success


def run_part3(gemini_model, count_passages: int, level_filter: str | None, output_file: Path):
    """Sinh dữ liệu Bài 3 (đoạn văn) và ghi vào file JSONL."""
    print(f"\n{'='*55}")
    print(f"  BẮT ĐẦU SINH DỮ LIỆU BÀI 3 (ĐOẠN VĂN) - {count_passages} đoạn")
    print(f"{'='*55}")

    success_passages = 0
    total_records = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(1, count_passages + 1):
            topic = random.choice(TOPICS)
            level = level_filter if level_filter else random.choice(LEVELS)
            print(f"  [{i}/{count_passages}] Đang tạo đoạn văn: Chủ đề '{topic}' ({level})")

            records = generate_part3_records(gemini_model, topic, level)
            if records:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total_records += len(records)
                success_passages += 1
                print(f"    ✅ Tạo được {len(records)} câu hỏi! (Tổng: {total_records})")
            else:
                print(f"    ❌ Thất bại, bỏ qua.")
            time.sleep(DELAY_SECONDS)

    print(f"\nKết quả Bài 3: {success_passages}/{count_passages} đoạn văn → {total_records} câu hỏi → {output_file}")
    return total_records


# ─── 7. ĐIỂM BẮT ĐẦU ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh dữ liệu JLPT tổng hợp bằng Gemini API")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                        help="'test' = sinh thử 10+5, 'full' = sinh toàn bộ (mặc định: test)")
    parser.add_argument("--part", choices=["2", "3", "both"], default="both",
                        help="Chỉ sinh Bài 2, Bài 3, hoặc cả hai (mặc định: both)")
    parser.add_argument("--level", default=None,
                        help="Lọc theo level (N1/N2/N3/N4/N5). Mặc định: ngẫu nhiên")
    parser.add_argument("--api-key", default=API_KEY,
                        help="Gemini API Key (mặc định tự lấy từ .env)")
    args = parser.parse_args()

    if not args.api_key:
        print("LỖI: Không tìm thấy GEMINI_API_KEY. Hãy thêm vào file .env hoặc dùng --api-key")
        sys.exit(1)

    # Cấu hình số lượng tùy mode
    if args.mode == "test":
        N_PART2 = 10   # Sinh thử 10 câu Bài 2
        N_PART3 = 5    # Sinh thử 5 đoạn văn Bài 3 (= ~15 câu)
        print("🧪 CHẾ ĐỘ TEST: Sinh thử 10 câu Bài 2 và 5 đoạn văn Bài 3")
    else:
        N_PART2 = 595  # Full: 1 câu per điểm ngữ pháp
        N_PART3 = 500  # Full: 500 đoạn văn (≈ 1500 câu)
        print("🚀 CHẾ ĐỘ FULL: Sinh toàn bộ dữ liệu (có thể mất 2-3 giờ)")

    # Lọc theo level nếu cần
    grammar_points = load_grammar_points()
    if args.level:
        grammar_points = [g for g in grammar_points if g.get("level") == args.level]
        print(f"  Đã lọc: {len(grammar_points)} điểm ngữ pháp cấp {args.level}")

    # Khởi tạo Gemini client
    print(f"\nĐang khởi tạo Gemini API (gemini-2.5-flash)...")
    gemini_client = init_gemini(args.api_key)
    print("  ✅ Sẵn sàng!")

    # Chạy theo lựa chọn
    if args.part in ("2", "both"):
        n = min(N_PART2, len(grammar_points))
        run_part2(gemini_client, grammar_points, n, OUTPUT_PART2)

    if args.part in ("3", "both"):
        run_part3(gemini_client, N_PART3, args.level, OUTPUT_PART3)

    print("\n" + "="*55)
    print("  HOÀN TẤT! Hãy chạy scripts/clean_data/step4_combine.py")
    print("  để gộp data mới vào combined_train_v2.jsonl")
    print("="*55)

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Script 02: Đánh giá mô hình (Evaluate model) sau khi Fine-tune.

BƯỚC QUAN TRỌNG: Đây là kịch bản (script) để "chấm điểm" xem AI của bạn đã thông minh
lên được bao nhiêu phần trăm sau quá trình "luyện đan" (fine-tuning).

CÁC TIÊU CHÍ ĐÁNH GIÁ TRONG SCRIPT NÀY:
1. Độ chính xác trắc nghiệm (MCQ Accuracy): So sánh đáp án trong thẻ <answer>.
   - Đây là điểm số quan trọng nhất để biết model có làm bài JLPT đúng hay không.
2. Chất lượng suy luận (CoT ROUGE-L): Đo mức độ giống nhau giữa phần giải thích
   trong thẻ <thinking> của model so với đáp án gốc.
   - Thể hiện model có hiểu vì sao chọn đáp án đó không, hay chỉ là "đoán mò".
3. Độ trung thành với Format (Format Compliance): Xem model có làm đúng theo
   yêu cầu sinh ra <thinking> và <answer> hay không.

Cách chạy:
    # Chấm bằng ROUGE-L (chấm tự động không tốn phí)
    python scripts/02_evaluate.py --model <tên_model>

    # Chấm bằng Gemini API (Giám khảo AI - Chính xác nhất)
    python scripts/02_evaluate.py --model <tên_model> --gemini-key <API_KEY_CỦA_BẠN>

Yêu cầu thư viện:
    pip install transformers torch peft rouge-score tqdm google-generativeai
"""

import json
import re
import os
import argparse
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

# ─── CẤU HÌNH ĐƯỜNG DẪN ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
EVAL_DATA = ROOT / "data" / "processed" / "combined_eval.jsonl"
REPORT_DIR = ROOT / "data" / "processed"

# ─── 1. HÀM ĐỌC DỮ LIỆU ĐÁNH GIÁ ─────────────────────────────────────────────────
def load_eval_data(path: Path, max_samples: int = 200) -> list[dict]:
    """
    Đọc dữ liệu từ file eval (không dùng file train để đảm bảo tính khách quan).
    Mặc định chỉ lấy 200 câu hỏi để chấm nhanh (có thể tăng lên tùy ý).
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    # Xáo trộn ngẫu nhiên để lấy đủ các dạng bài khác nhau
    if len(records) > max_samples:
        import random
        random.seed(42)
        random.shuffle(records)
        records = records[:max_samples]
    return records


# ─── 2. HÀM LOAD MODEL ĐỂ INFERENCE (DỰ ĐOÁN) ──────────────────────────────────
def load_model(model_name_or_path: str):
    """
    Load mô hình ngôn ngữ (HuggingFace) để tự động giải bài.
    Sử dụng torch.float16 và device_map="auto" để tối ưu hóa VRAM trên Colab/GPU.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"Đang tải model từ: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto", # Tự động chia tải sang GPU nếu có
            trust_remote_code=True,
        )
        model.eval() # Chuyển sang chế độ inference (không train)
        return model, tokenizer
    except ImportError:
        raise ImportError(
            "Bạn cần cài đặt các thư viện Deep Learning trước:\n"
            "  pip install transformers torch peft"
        )


def format_prompt(instruction: str, input_text: str, tokenizer) -> str:
    """
    Đóng gói câu hỏi thành định dạng Alpaca.
    Lưu ý: Format này PHẢI GIỐNG HỆT lúc bạn train model.
    """
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    return prompt


def generate_response(model, tokenizer, instruction: str, input_text: str) -> str:
    """Đưa câu hỏi vào model và lấy câu trả lời."""
    prompt = format_prompt(instruction, input_text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Thiết lập tham số sinh văn bản (text generation)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,   # Cho phép model nghĩ (CoT) đủ dài (300 từ)
        temperature=0.3,      # Dùng nhiệt độ thấp (0.3) để model trả lời chính xác, logic
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Cắt bỏ phần prompt, chỉ lấy câu trả lời của model
    prompt_len = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return response.strip()


# ─── 3. CÁC HÀM TÍNH ĐIỂM (METRICS) ──────────────────────────────────────────────

def parse_cot(text: str):
    """
    Bóc tách phần suy luận <thinking> và đáp án <answer> từ chuỗi.
    """
    # Tìm nội dung trong <thinking>
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL | re.IGNORECASE)
    thinking = thinking_match.group(1).strip() if thinking_match else ""
    
    # Tìm đáp án trong <answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    return thinking, answer


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Tính điểm ROUGE-L: Đoạn văn bản tiên đoán giống đoạn văn mẫu đến mức nào.
    Điểm từ 0.0 (không giống chút nào) đến 1.0 (giống hệt).
    Dùng để chấm "chất lượng giải thích" của thẻ <thinking>.
    """
    if not prediction or not reference:
        return 0.0
        
    p_tokens = prediction.lower().split()
    r_tokens = reference.lower().split()
    
    if not p_tokens or not r_tokens:
        return 0.0

    # Tìm chuỗi con chung dài nhất (Longest Common Subsequence)
    # Đây là thuật toán cơ bản của ROUGE-L
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    lcs = lcs_length(p_tokens, r_tokens)
    precision = lcs / len(p_tokens)
    recall = lcs / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─── 4. GEMINI NHƯ MỘT GIÁM KHẢO (LLM-as-a-Judge) ─────────────────────────────

def evaluate_explanation_with_gemini(instruction: str, input_text: str, ref_thinking: str, pred_thinking: str, gemini_model) -> dict:
    """
    Sử dụng Gemini API để đọc và chấm điểm phần giải thích của model.
    Trả về một dictionary gồm 'score' (1-5) và 'reason' (lời nhận xét).
    """
    if not ref_thinking or not pred_thinking:
        return {"score": 0, "reason": "Không có phần giải thích để chấm."}

    prompt = f"""Bạn là một giáo viên tiếng Nhật JLPT chuyên nghiệp.
Nhiệm vụ của bạn là chấm điểm phần giải thích của học sinh (AI Model) so với đáp án gốc.
Hãy tập trung vào logic suy luận, độ chính xác của ngữ pháp/từ vựng tiếng Nhật.

[CÂU HỎI]
Instruction: {instruction}
Input: {input_text}

[GIẢI THÍCH ĐÁP ÁN GỐC (CHUẨN)]
{ref_thinking}

[GIẢI THÍCH CỦA HỌC SINH (MODEL)]
{pred_thinking}

Dựa vào giải thích gốc, hãy chấm điểm giải thích của học sinh theo thang điểm từ 1 đến 5:
1: Sai hoàn toàn logic, hiểu sai ngữ cảnh hoặc bịa đặt kiến thức.
2: Có nhắc đến từ khóa nhưng giải thích sai logic hoặc lạc đề.
3: Hiểu được ngữ cảnh nhưng giải thích thiếu ý hoặc hơi lủng củng.
4: Giải thích đúng logic, đúng ý nghĩa nhưng chưa thật sự sắc sảo bằng đáp án gốc.
5: Giải thích xuất sắc, logic chặt chẽ, chính xác tuyệt đối như đáp án gốc.

Yêu cầu đầu ra (Chỉ trả về định dạng JSON hợp lệ, không có markdown code block):
{{
  "score": <điểm số từ 1 đến 5>,
  "reason": "<1 câu nhận xét ngắn gọn tại sao cho điểm đó>"
}}
"""
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        return {"score": 0, "reason": f"Lỗi gọi Gemini API: {e}"}


# ─── 5. VÒNG LẶP CHẤM THI CHÍNH ─────────────────────────────────────────────────

def evaluate(model, tokenizer, eval_records: list[dict], gemini_api_key: str = None) -> dict:
    from tqdm import tqdm
    
    gemini_model = None
    if gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            # Dùng flash model vì nó rẻ và rất nhanh, đủ sức chấm điểm
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            print("Đã khởi tạo xong Giám khảo Gemini API!")
        except ImportError:
            print("LỖI: Bạn chưa cài thư viện google-generativeai. Bỏ qua chấm bằng Gemini.")
            gemini_model = None

    results = {
        "total": 0,
        "format_correct": 0,        # Bao nhiêu câu xuất ra đúng format <thinking>...<answer>
        "mcq_correct": 0,           # Trắc nghiệm: Bao nhiêu câu chọn đúng A/B/C/D
        "mcq_total": 0,
        "rouge_scores": [],         # Danh sách điểm ROUGE chấm phần suy luận
        "gemini_scores": [],        # Danh sách điểm Gemini (1-5) nếu dùng Gemini
        "sample_outputs": [],       # Lưu lại vài câu để người dùng đọc trực tiếp
    }

    print(f"\nBắt đầu chấm thi cho {len(eval_records)} câu hỏi JLPT...")
    
    for i, record in enumerate(tqdm(eval_records, desc="Đang làm bài")):
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        reference = record.get("output", "")

        results["total"] += 1
        
        # Gọi model làm bài
        prediction = generate_response(model, tokenizer, instruction, input_text)

        # Bóc tách đáp án của Model (Dự đoán) và Của Đề Bài (Mẫu)
        pred_thinking, pred_answer = parse_cot(prediction)
        ref_thinking, ref_answer = parse_cot(reference)

        # 1. Chấm Format
        # Nếu model có xuất ra được thẻ answer tức là nó tuân thủ format CoT
        if pred_answer:
            results["format_correct"] += 1

        # 2. Chấm Đáp Án Trắc Nghiệm (Exact Match)
        is_mcq_correct = False
        if ref_answer:  # Chỉ chấm trắc nghiệm nếu câu hỏi có đáp án A/B/C/D
            results["mcq_total"] += 1
            # Loại bỏ ký tự lạ, dấu câu để so sánh chính xác (Vd: "2" vs " 2 ")
            p_ans = re.sub(r"[^\w]", "", pred_answer.lower())
            r_ans = re.sub(r"[^\w]", "", ref_answer.lower())
            
            if p_ans == r_ans and p_ans != "":
                results["mcq_correct"] += 1
                is_mcq_correct = True

        # 3. Chấm Chất Lượng Giải Thích (ROUGE-L hoặc Gemini)
        rouge = 0.0
        gemini_result = {"score": 0, "reason": ""}
        
        if ref_thinking:
            rouge = compute_rouge_l(pred_thinking, ref_thinking)
            # Nếu có Gemini Key, nhờ Gemini chấm bài
            if gemini_model:
                gemini_result = evaluate_explanation_with_gemini(instruction, input_text, ref_thinking, pred_thinking, gemini_model)
                results["gemini_scores"].append(gemini_result["score"])
        else:
            rouge = compute_rouge_l(prediction, reference)
            
        results["rouge_scores"].append(rouge)

        # Lưu lại vài ví dụ đầu tiên để in ra báo cáo
        if i < 10:
            sample_info = {
                "instruction": instruction,
                "input": input_text,
                "ref_answer": ref_answer,
                "pred_answer": pred_answer,
                "mcq_correct": is_mcq_correct,
                "rouge_score": round(rouge, 3),
                "full_prediction": prediction
            }
            if gemini_model:
                sample_info["gemini_score"] = gemini_result["score"]
                sample_info["gemini_reason"] = gemini_result["reason"]
                
            results["sample_outputs"].append(sample_info)

    return results


def summarize_results(results: dict) -> dict:
    """Tính toán phần trăm và tóm tắt kết quả thành một dictionary đẹp mắt."""
    scores = results["rouge_scores"]
    avg_rouge = sum(scores) / len(scores) if scores else 0.0

    mcq_acc = (results["mcq_correct"] / results["mcq_total"]) if results["mcq_total"] > 0 else 0.0
    format_acc = (results["format_correct"] / results["total"]) if results["total"] > 0 else 0.0

    summary = {
        "tong_so_cau": results["total"],
        "ty_le_tuan_thu_format": round(format_acc * 100, 2),
        "do_chinh_xac_trac_nghiem": round(mcq_acc * 100, 2),
        "diem_giai_thich_ROUGE": round(avg_rouge, 4),
        "chi_tiet_ROUGE": {
            ">=0.8 (Giai thich xuat sac)": sum(1 for s in scores if s >= 0.8),
            "0.5-0.8 (Giai thich tot)":    sum(1 for s in scores if 0.5 <= s < 0.8),
            "0.3-0.5 (Giai thich tam)":    sum(1 for s in scores if 0.3 <= s < 0.5),
            "<0.3 (Giai thich te/lac de)": sum(1 for s in scores if s < 0.3),
        }
    }
    
    if results.get("gemini_scores"):
        g_scores = results["gemini_scores"]
        summary["diem_giai_thich_GEMINI_trung_binh"] = round(sum(g_scores) / len(g_scores), 2)
        summary["chi_tiet_GEMINI"] = {
            "5 diem (Xuat sac)": sum(1 for s in g_scores if s == 5),
            "4 diem (Kha)":      sum(1 for s in g_scores if s == 4),
            "3 diem (Trung binh)": sum(1 for s in g_scores if s == 3),
            "1-2 diem (Kem)":    sum(1 for s in g_scores if s <= 2),
        }
        
    return summary


# ─── 6. ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chấm điểm AI giải đề JLPT")
    parser.add_argument("--model", type=str, required=True, 
                        help="Tên model (vd: 'Qwen/Qwen2.5-3B') hoặc đường dẫn tới thư mục model")
    parser.add_argument("--max-samples", type=int, default=200, 
                        help="Chỉ chấm N câu hỏi để tiết kiệm thời gian (mặc định 200)")
    parser.add_argument("--gemini-key", type=str, default=os.environ.get("GEMINI_API_KEY", ""),
                        help="API Key của Google Gemini để làm Giám khảo (Mặc định tự động lấy từ file .env)")
    args = parser.parse_args()

    if not EVAL_DATA.exists():
        print(f"LỖI: Không tìm thấy file dữ liệu đánh giá: {EVAL_DATA}")
        sys.exit(1)

    print("="*60)
    print(" BẮT ĐẦU CHẤM ĐIỂM (EVALUATION)")
    print("="*60)

    # 1. Load data
    eval_records = load_eval_data(EVAL_DATA, max_samples=args.max_samples)
    print(f"Đã load {len(eval_records)} câu hỏi từ tập EVAL.")

    # 2. Load model
    model, tokenizer = load_model(args.model)

    # 3. Chấm bài
    raw_results = evaluate(model, tokenizer, eval_records, gemini_api_key=args.gemini_key)
    summary = summarize_results(raw_results)

    # 4. In báo cáo
    print("\n" + "="*60)
    print(" BÁO CÁO KẾT QUẢ")
    print("="*60)
    print(f"- Tổng số câu đã chấm: {summary['tong_so_cau']}")
    print(f"- Tỷ lệ tuân thủ Format <thinking>: {summary['ty_le_tuan_thu_format']}%")
    print(f"- ĐỘ CHÍNH XÁC TRẮC NGHIỆM (MCQ):   {summary['do_chinh_xac_trac_nghiem']}%")
    print(f"- ĐIỂM GIẢI THÍCH (ROUGE-L):        {summary['diem_giai_thich_ROUGE']}")
    
    print("\nPhân bổ điểm giải thích (ROUGE):")
    for k, v in summary['chi_tiet_ROUGE'].items():
        print(f"  + {k}: {v} câu")

    if "diem_giai_thich_GEMINI_trung_binh" in summary:
        print(f"\n- ĐIỂM GIÁM KHẢO GEMINI (1-5):      {summary['diem_giai_thich_GEMINI_trung_binh']} / 5.0")
        for k, v in summary['chi_tiet_GEMINI'].items():
            print(f"  + {k}: {v} câu")

    print("\n" + "="*60)
    print(" MỘT VÀI VÍ DỤ MODEL ĐÃ LÀM")
    print("="*60)
    for i, s in enumerate(raw_results["sample_outputs"], 1):
        print(f"\n--- Câu {i} ---")
        print(f"Hỏi: {s['instruction']}")
        if s['input']:
            print(f"Nội dung: {s['input']}")
        print(f"-> Đáp án đúng: {s['ref_answer']}")
        print(f"-> Model chọn:  {s['pred_answer']}  ({'✅ ĐÚNG' if s['mcq_correct'] else '❌ SAI'})")
        if "gemini_score" in s:
            print(f"-> Giám khảo Gemini chấm: {s['gemini_score']}/5 - Lý do: {s['gemini_reason']}")
        print(f"-> Bài làm của Model:\n{s['full_prediction']}")

    # 5. Lưu báo cáo ra file JSON để tham khảo sau
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORT_DIR / "eval_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu báo cáo chi tiết vào {report_file}")

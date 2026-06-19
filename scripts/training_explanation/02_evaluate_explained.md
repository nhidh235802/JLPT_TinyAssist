# Giải phẫu chi tiết: 02_evaluate.py

File này có nhiệm vụ: **Chấm điểm model (đánh giá Baseline hoặc model đã train)**. Nó lấy dữ liệu test, bắt model làm bài, ghi nhận kết quả và dùng một Giám khảo (Gemini) để chấm độ chính xác của câu giải thích.

Dưới đây là ý nghĩa của từng khối code và tham số quan trọng:

## 1. Cấu hình ban đầu
```python
model_name_or_path = "unsloth/Qwen2.5-7B-Instruct" 
# Đây là tên model trên HuggingFace. Khi bạn muốn test model đã train, 
# hãy đổi nó thành đường dẫn trỏ tới thư mục LoRA của bạn (ví dụ: "/content/drive/.../jlpt-model-lora")

test_file_path = "../../data/processed/combined_train.jsonl"
# File chứa bài thi. Đáng lẽ phải dùng tập test riêng, nhưng ở đây ta bốc ngẫu nhiên 
# 200 câu từ tập train để test nhanh.

num_eval_samples = 200
# Số lượng câu hỏi muốn test. 200 câu là con số hoàn hảo: đủ lớn để đại diện cho 7000 câu, 
# và đủ nhỏ để không mất cả ngày ngồi chờ model làm bài.
```

## 2. Hàm `load_model()`
Khối này nạp model vào VRAM của GPU.
```python
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
# Tokenizer là "từ điển" của model, giúp biến chữ viết thành các con số (tokens) để model tính toán.

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16, 
    # Dùng float16 thay vì float32 mặc định. Giúp model nhẹ đi một nửa (từ 28GB xuống 14GB) 
    # mà không làm giảm độ thông minh, vừa vặn nhét vào card T4 của Colab.
    
    device_map="auto",
    # HuggingFace sẽ tự động tính toán và nhét model vào GPU. 
    # Nếu GPU đầy, nó sẽ đẩy bớt sang RAM máy tính (nhưng chạy sẽ rất chậm).
)
```

## 3. Khối sinh văn bản (Generation)
Đây là lúc model "rặn" ra câu trả lời.
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    # CỰC KỲ QUAN TRỌNG: Chỉ cho phép model "nói nhảm" tối đa 300 chữ (tokens) cho mỗi câu.
    # Nếu không chặn lại, model chưa train có thể bị ảo giác và in ra hàng ngàn chữ không liên quan,
    # làm tốn VRAM và tốn thời gian chờ đợi vô ích.
    
    pad_token_id=tokenizer.eos_token_id,
    # Khi model làm xong bài, nó sẽ tự nhét ký tự EOS (End of Sentence) vào để ra hiệu dừng lại.
    
    do_sample=False
    # Tắt sự "sáng tạo". do_sample=False (hay còn gọi là Greedy Search) buộc model luôn chọn
    # từ có xác suất cao nhất. Trong thi cử trắc nghiệm, chúng ta cần sự chính xác và nhất quán,
    # chứ không cần model phải làm thơ hay bịa chuyện sáng tạo.
)
```

## 4. Parser và Giám khảo Gemini
Model trả lời xong, làm sao biết đúng hay sai?
1. **Lấy đáp án (Parser):** Chúng ta dùng Regex (Biểu thức chính quy) để tìm đúng nội dung nằm giữa thẻ `<answer>...</answer>`. Nếu số tìm được trùng với đáp án gốc -> Đúng (+1 điểm).
2. **Chấm giải thích (LLM as a Judge):** Nếu chỉ khoanh bừa đúng thì chưa chắc đã hiểu bài. Ta gửi phần `<thinking>` của Qwen cho Gemini (prompt `JUDGE_PROMPT`) để nhờ Gemini đóng vai giáo viên chấm điểm từ 1-5 xem tư duy của Qwen có logic không.

## Tại sao Evaluation lại chạy 16-bit nguyên bản?
Ở file train ta dùng bản nén 4-bit, nhưng ở file đánh giá này ta nạp toàn bộ trọng số 16-bit.
Lý do: Khi đánh giá Baseline (Mốc cơ sở), chúng ta muốn đo lường năng lực "trí tuệ thật sự" gốc của model mà không bị suy hao bất kỳ % nào do kỹ thuật nén. Điều này giúp ta có một thước đo chuẩn xác nhất trước khi bắt đầu can thiệp (train).

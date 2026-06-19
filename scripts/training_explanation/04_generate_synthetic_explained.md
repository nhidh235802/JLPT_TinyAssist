# Giải phẫu chi tiết: 04_generate_synthetic.py

Bản chất của file này là ứng dụng **"LLM tự sinh Data để dạy LLM" (Synthetic Data Generation)**. Qwen 7B thì ngốc (vì nó nhỏ), nhưng Gemini 2.0 Flash Lite thì cực kỳ thông minh (vì nó là model trăm tỷ tham số của Google). Ta lấy cái thông minh tạo ra sách giáo khoa xịn để dạy cho thằng ngốc.

## 1. Thiết kế Prompt (Bí quyết)
LLM rất giỏi làm bài, nhưng lại rất hay làm sai nếu bị bắt "Ra đề thi". Để ép Gemini ra đúng định dạng thi JLPT, ta phải dùng kỹ thuật **Few-shot Prompting / Strict Rule Prompting**.

```python
PROMPT_PART2 = """Bạn là giáo viên tiếng Nhật JLPT chuyên nghiệp...
...
ĐỊNH DẠNG OUTPUT (JSON THUẦN TÚY, không có markdown):
{{
  "full_sentence": "câu hoàn chỉnh tiếng Nhật",
  "sentence_with_blank": "phần đầu câu ＿＿＿★＿＿＿ phần cuối câu",
  "choices": {{...}},
  "correct_order": [...],
  "answer": "số của mảnh ở vị trí ★",
  "thinking": "Phân tích: [nghĩa câu]..."
}}"""
```
- **Ép kiểu JSON:** Thay vì bảo "hãy tạo một câu hỏi", ta ép Gemini điền vào một cái Form bằng định dạng JSON. Điều này giúp ta lấy được chính xác câu văn, 4 đáp án và lời giải thích tách biệt nhau hoàn toàn để nạp vào Code.
- **Bắt ép tư duy (`<thinking>`):** Đây là kỹ thuật **Chain-of-Thought**. Thay vì bắt Gemini rặn ra đáp án luôn, ta bắt nó viết ra giấy nháp: Tại sao câu này đúng ngữ pháp? Mảnh này đứng trước mảnh kia vì quy tắc gì? Khi nạp đoạn `<thinking>` này vào Qwen, Qwen sẽ học được "Cách suy luận logic" chứ không chỉ học thuộc lòng đáp án.

## 2. API Rate Limiting & Retry Mechanism
API công cộng (đặc biệt là bản miễn phí) rất dễ bị sập hoặc cấm do Rate Limit (Gửi request quá nhanh).

```python
# gemini-2.0-flash-lite: Miễn phí 1500 req/phút (đủ dùng thoải mái)
DELAY_SECONDS = 2  # Nghỉ 2 giây sau mỗi câu để an toàn
```

Cơ chế Retry (Thử lại) kiên cường:
```python
def call_gemini(client, prompt: str, retries: int = 3):
    for attempt in range(retries):
        try:
            return client.models.generate_content(...).text
        except Exception as e:
            if "429" in str(e): # Lỗi Quota Exceeded / Rate Limit
                wait = 35 # API bảo đợi 30s, ta đợi hẳn 35s cho chắc
            else:
                wait = (attempt + 1) * 10
            time.sleep(wait)
```
- **Lỗi 429:** Là lỗi nhạy cảm nhất. Google bảo "Mày gọi tao nhanh quá, cút ra đợi 30 giây nữa hãy quay lại". Nếu ta cố tình gọi tiếp ngay, Google sẽ ban vĩnh viễn API Key. Đoạn code này bắt được lỗi đó và ngoan ngoãn tắt máy ngủ 35 giây trước khi gọi lại.

## 3. Ghi file dạng Streaming
```python
with open(output_file, "w", encoding="utf-8") as f:
    for gp in selected:
        record = generate_part2_record(gemini_model, gp)
        if record:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```
- File `synthetic_part2.jsonl` được mở ở chế độ `w` (Write). 
- **Bí mật nằm ở dấu `\n`**: Cứ Gemini sinh xong câu nào, ta ném ngay câu đó vào ổ cứng (dòng `f.write(...)`). Điều này đảm bảo an toàn tuyệt đối 100%. Dù bạn có đang sinh câu số 1500 mà bị cúp điện, 1499 câu trước đó đã nằm gọn gàng an toàn trong ổ cứng, không mất đi đâu cả!

## Tại sao lại phải sinh Data giả? (Synthetic Data)
Trong dataset gốc lấy từ Tatoeba hay Mazii, thường nó chỉ có 1 câu tiếng Nhật và 1 câu tiếng Việt (Song ngữ). Nó không hề có bài tập "Đục lỗ" hay "Sắp xếp sao". 
Nếu ta chỉ cho Qwen đọc song ngữ, Qwen sẽ thành 1 cỗ máy dịch thuật (Translator).
Khi ta dùng Gemini tạo ra bài tập giả lập môi trường thi JLPT (Có 4 đáp án A B C D, có tư duy loại trừ), Qwen sẽ học được tư duy làm bài trắc nghiệm của con người, từ đó biến thành một "Gia sư luyện thi" đúng nghĩa!

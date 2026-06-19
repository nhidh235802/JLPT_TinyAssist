# JLPT TinyAssist (Qwen 7B Fine-Tuning)

Branch học tập và thử fine-tune mô hình **Qwen2.5-7B-Instruct** làm gia sư ôn thi JLPT (có Chain of Thought để giải thích ngữ pháp).

## Thư mục chính
* `scripts/`:
  - `clean_data/`: Dọn dẹp dữ liệu thô.
  - `02_evaluate.py`: Chạy thử và chấm điểm tự động bằng Gemini API (LLM-as-a-Judge).
  - `03_train_colab.py`: Kịch bản train chính bằng Unsloth + QLoRA trên Google Colab.
  - `04_generate_synthetic.py`: Script dùng Gemini API để sinh thêm dữ liệu trắc nghiệm JLPT giả lập.
  - `training_explanation/`: Tài liệu đọc hiểu chi tiết từng dòng code & tham số huấn luyện.
* `data/`: Chứa file dữ liệu thô (`raw`) và dữ liệu huấn luyện (`processed`).


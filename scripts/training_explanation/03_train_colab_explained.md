# Giải phẫu chi tiết: 03_train_colab.py

Đây là "Trái tim" của dự án: Lò luyện đan (Training script). File này sử dụng những công nghệ tối tân nhất hiện nay (`Unsloth` + `QLoRA`) để có thể ép một model 7 Tỷ tham số khổng lồ chạy mượt mà trên chiếc card màn hình miễn phí (Tesla T4 16GB VRAM) của Google.

Dưới đây là ý nghĩa của từng cấu hình (Hyperparameters) quan trọng nhất:

## 1. Tải Model bằng Unsloth (Khối 2)
```python
model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
load_in_4bit = True 
```
- **Tại sao dùng bản `bnb-4bit`?** Bản gốc của Qwen nặng ~15GB. Quá trình nén nó xuống 4-bit (Quantization) tiêu tốn rất nhiều RAM hệ thống và dễ làm sập Colab. Bản `bnb-4bit` đã được Unsloth nén sẵn cẩn thận thành file 5GB. Ta chỉ việc tải về, tránh được hoàn toàn nguy cơ sập RAM (OOM - Out of Memory).
- **`load_in_4bit=True` (Kỹ thuật QLoRA):** QLoRA đóng băng toàn bộ não bộ 4-bit này lại (không cho thay đổi). Thay vào đó, nó gắn thêm một bộ não phụ nhỏ xíu (LoRA) bằng chuẩn 16-bit. Ta chỉ train bộ não phụ này. Kết quả: VRAM giảm từ 100GB xuống chỉ còn ~6GB!

## 2. Cấu hình LoRA (Não phụ)
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    lora_alpha = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    use_gradient_checkpointing = "unsloth",
)
```
- **`r = 16`:** Độ lớn của bộ não phụ. `r=8` là vừa đủ học, `r=16` thông minh hơn nhưng train chậm hơn. Cao hơn nữa (32, 64) thường không mang lại nhiều tác dụng mà lại làm mô hình dễ bị "Học vẹt" (Overfitting).
- **`target_modules`:** Xác định xem bộ não phụ này sẽ được gắn vào những nơ-ron nào của bộ não chính. Gắn vào tất cả các module có chữ `proj` (Linear Layers) là công thức chuẩn mực nhất hiện nay để tối đa hóa độ thông minh.
- **`use_gradient_checkpointing = "unsloth"`:** Phép màu giảm VRAM. Khi train, hệ thống phải lưu trữ rất nhiều rác trung gian (Activations). Tính năng này ra lệnh: "Tính xong cái nào xóa ngay cái đó, lúc nào cần thì tính lại". Nó làm chậm tốc độ train đi ~20% nhưng đổi lại tiết kiệm được 60% VRAM! (Unsloth đã tối ưu hàm này để không bị chậm đi quá nhiều).

## 3. Cấu hình SFTConfig (Lò luyện)
`SFTTrainer` (Supervised Fine-Tuning) là người thầy giáo cầm thước kẻ dạy model từng dòng dữ liệu.

```python
per_device_train_batch_size = 2,
gradient_accumulation_steps = 4,
```
- **Batch Size (`2`):** Số câu hỏi model phải đọc cùng lúc trong 1 nhịp thở. Vì VRAM yếu, ta chỉ cho nó đọc 2 câu/lần.
- **Gradient Accumulation (`4`):** Đọc 2 câu thì học được ít quá, dễ bị lệch hướng. Vậy ta ra lệnh: "Mày cứ đọc 2 câu, nhưng **đừng** vội sửa não. Đọc tiếp 2 câu nữa... làm như vậy 4 lần (tổng 8 câu) rồi mới tổng hợp lại và sửa não 1 lần!". Đây là kỹ thuật mô phỏng Batch Size lớn trên Card yếu. (Batch size thực tế ở đây = 2 x 4 = 8).

```python
learning_rate = 2e-4, # (0.0002)
lr_scheduler_type = "linear",
warmup_steps = 5,
```
- **`learning_rate` (Tốc độ học):** Khả năng tiếp thu. `2e-4` là con số vàng cho QLoRA. Nếu cao quá (như 1e-2), model sẽ bị "ngáo" (phá vỡ kiến thức cũ). Nếu thấp quá (1e-6), nó học quá chậm, mãi không xong.
- **`warmup_steps = 5`:** Khởi động. Trong 5 bước đầu tiên, Tốc độ học sẽ tăng từ từ từ 0 lên `2e-4` để tránh việc model bị "sốc kiến thức" ngay ở giây phút đầu tiên.

```python
max_steps = 500, # Hoặc num_train_epochs = 1
```
- **`max_steps = 500`:** Dừng ép buộc sau 500 bước học (Dùng để test chạy thử xem có lỗi không).
- **`num_train_epochs = 1`:** Một Epoch nghĩa là model đã đọc qua **toàn bộ** 7,000 câu trong cuốn sách dữ liệu đúng 1 lần. Thường với fine-tune LLM, 1-3 Epochs là đủ. Đọc quá nhiều lần 1 cuốn sách sẽ dẫn đến "Học vẹt" (Nó nhớ luôn đáp án câu hỏi đó thay vì hiểu logic).

```python
save_steps = 100,
save_total_limit = 2,
```
- **Lưu dự phòng (Checkpointing):** Rất hay bị sập mạng khi dùng Colab miễn phí. Lệnh này bảo máy: "Cứ học được 100 bước thì cất não vào thư mục Google Drive 1 lần. Nhớ dọn dẹp, chỉ giữ lại 2 bản gần nhất để khỏi đầy ổ cứng". Nhờ vậy, nếu sập mạng ở bước 499, ta chỉ việc load lại bước 400 và chạy tiếp, không phải làm lại từ đầu!

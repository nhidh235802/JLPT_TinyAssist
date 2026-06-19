# ==============================================================================
# HƯỚNG DẪN CHẠY TRÊN GOOGLE COLAB
# 1. Bật GPU: Runtime -> Change runtime type -> T4 GPU (hoặc A100/L4 nếu có)
# 2. Cài đặt Unsloth (Chạy lệnh sau trong ô đầu tiên):
#    !pip install unsloth "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
# 3. Kết nối Google Drive để lưu Checkpoints (chống mất data khi rớt mạng):
#    Chạy lệnh: from google.colab import drive; drive.mount('/content/drive')
# 4. Upload file `combined_train.jsonl` lên Colab (vào thư mục /content hoặc Drive)
# 5. Copy toàn bộ code bên dưới vào ô tiếp theo và chạy
# ==============================================================================

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ─── 1. CẤU HÌNH THÔNG SỐ (HYPERPARAMETERS) ──────────────────────────────────
max_seq_length = 2048 # Độ dài tối đa của câu hỏi + câu trả lời
dtype = None # Tự động phát hiện (float16 cho T4, bfloat16 cho Ampere)
load_in_4bit = True # Dùng QLoRA 4-bit để cực kỳ tiết kiệm VRAM

# Tên model HuggingFace
# Lời khuyên: Dùng bản bnb-4bit sẽ tải nhanh hơn và chắc chắn không sập RAM:
# model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" 
model_name = "unsloth/Qwen2.5-7B-Instruct"

# ─── 2. TẢI MODEL & TOKENIZER SIÊU NHANH BẰNG UNSLOTH ───────────────────────
print("Đang tải model (khoảng 3-4 phút)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Thêm kỹ thuật LoRA (Chỉ train 1% tham số, giữ nguyên 99% kiến thức gốc)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Càng cao càng thông minh nhưng train lâu hơn (chuẩn là 8 hoặc 16)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth tối ưu dropout = 0
    bias = "none",    # Tối ưu hóa
    use_gradient_checkpointing = "unsloth", # Giảm VRAM mạnh
    random_state = 3407,
)

# ─── 3. CHUẨN BỊ DỮ LIỆU (DATASET) ──────────────────────────────────────────
# Định dạng Alpaca chuẩn (giống hệt trong helper.py của chúng ta)
alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Dấu chấm hết câu của AI
def format_prompts(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

print("Đang tải file data combined_train.jsonl...")
# Nhớ upload file combined_train.jsonl lên thư mục hiện tại của Colab
dataset = load_dataset("json", data_files={"train": "combined_train.jsonl"}, split="train")
dataset = dataset.map(format_prompts, batched = True,)

# ─── 4. BẮT ĐẦU TRAINING LÒ ĐAN ──────────────────────────────────────────────
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Có thể để True nếu muốn gộp câu để train nhanh hơn
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Card VRAM yếu thì để 2, mạnh (24GB) thì để 4 hoặc 8
        gradient_accumulation_steps = 4, # Mô phỏng batch size lớn
        warmup_steps = 5,
        max_steps = 500, # BẬT CÁI NÀY ĐỂ TEST NHANH (Train 500 bước). Nếu muốn train thật, TẮT dòng này đi và bật num_train_epochs = 1
        # num_train_epochs = 1, # Train toàn bộ data đúng 1 vòng
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        
        # --- CẤU HÌNH LƯU CHECKPOINTS DỰ PHÒNG ---
        # Nên trỏ output_dir vào Google Drive để Colab có sập thì vẫn còn Checkpoint
        # output_dir = "/content/drive/MyDrive/jlpt-checkpoints",
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 100,        # Cứ 100 bước (steps) thì lưu lại 1 lần
        save_total_limit = 2,    # Chỉ giữ lại 2 checkpoint gần nhất để tránh đầy ổ cứng
    ),
)

print("BẮT ĐẦU TRAINING...")
# Nếu bị ngắt giữa chừng, đổi thành trainer.train(resume_from_checkpoint=True) để chạy tiếp
trainer_stats = trainer.train()

# ─── 5. LƯU MÔ HÌNH SAU KHI TRAIN ────────────────────────────────────────────
# Lưu kết quả LoRA (Nhẹ, khoảng 100-300MB)
print("Đang lưu model LoRA...")
model.save_pretrained("jlpt-model-lora")
tokenizer.save_pretrained("jlpt-model-lora")

print("Hoàn tất! Hãy download thư mục 'jlpt-model-lora' về máy.")

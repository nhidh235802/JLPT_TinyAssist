import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ─── 1. CẤU HÌNH THÔNG SỐ (HYPERPARAMETERS) ───────
max_seq_length = 2048 # Độ dài tối đa của câu hỏi + câu trả lời
dtype = None
load_in_4bit = True

model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

# ─── 2. TẢI MODEL & TOKENIZER BẰNG UNSLOTH ────
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
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ─── 3. CHUẨN BỊ DỮ LIỆU (DATASET) ────
# Định dạng Alpaca chuẩn
alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
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
dataset = load_dataset("json", data_files={"train": "/content/drive/MyDrive/ML_DL/data/processed/combined_train.jsonl"}, split="train")
dataset = dataset.map(format_prompts, batched = True,)

# ─── 4. BẮT ĐẦU TRAINING ─────
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 500,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        # Lưu checkpoint
        output_dir = "/content/drive/MyDrive/ML_DL/outputs",
        save_strategy = "steps",
        save_steps = 100,
        save_total_limit = 2,
    ),
)

print("BẮT ĐẦU TRAINING...")
trainer_stats = trainer.train(resume_from_checkpoint=True)

# ─── 5. LƯU MÔ HÌNH SAU KHI TRAIN ────
print("Đang lưu model LoRA...")
model.save_pretrained("/content/drive/MyDrive/ML_DL/jlpt-model-lora")
tokenizer.save_pretrained("/content/drive/MyDrive/ML_DL/jlpt-model-lora")

print("Hoàn tất! Model LoRA đã được lưu vào Google Drive của bạn.")

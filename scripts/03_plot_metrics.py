import json
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Vui lòng cài đặt thư viện để vẽ biểu đồ:")
    print("pip install matplotlib")
    exit(1)

def plot_metrics(trainer_state_path: str, output_path: str = "training_metrics.png"):
    """
    Đọc file trainer_state.json (do HuggingFace Trainer sinh ra)
    và vẽ biểu đồ Loss (Sai số) & Accuracy (Độ chính xác) theo từng bước.
    """
    path = Path(trainer_state_path)
    if not path.exists():
        print(f"LỖI: Không tìm thấy file {trainer_state_path}.")
        print("Lưu ý: File này chỉ xuất hiện SAU KHI/TRONG KHI bạn chạy lệnh Train (bằng HuggingFace Trainer/Unsloth).")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    log_history = data.get("log_history", [])
    if not log_history:
        print("LỖI: File log_history rỗng. Có vẻ model chưa train được bước nào.")
        return

    # Trích xuất dữ liệu
    steps_train = []
    loss_train = []
    
    steps_eval = []
    loss_eval = []

    for log in log_history:
        step = log.get("step")
        if "loss" in log:  # Training loss
            steps_train.append(step)
            loss_train.append(log["loss"])
        elif "eval_loss" in log:  # Evaluation loss
            steps_eval.append(step)
            loss_eval.append(log["eval_loss"])

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    
    if steps_train:
        plt.plot(steps_train, loss_train, label='Training Loss', color='blue', alpha=0.7)
    if steps_eval:
        plt.plot(steps_eval, loss_eval, label='Validation Loss', color='red', marker='o')

    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    
    # Chia nhỏ trục Y để dễ nhìn hơn
    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))  # Vạch chính cách nhau 0.1
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.05)) # Vạch phụ cách nhau 0.05
    
    plt.legend()
    # Bật lưới cho cả vạch chính (0.1) và vạch phụ (0.05)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n[THÀNH CÔNG] Đã vẽ xong biểu đồ! Mời bạn mở file ảnh: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ Loss/Accuracy từ quá trình train")
    parser.add_argument("--log-file", type=str, default="trainer_state.json",
                        help="Đường dẫn tới file trainer_state.json (trong thư mục checkpoint)")
    parser.add_argument("--output", type=str, default="training_metrics.png",
                        help="Tên file ảnh xuất ra")
    args = parser.parse_args()

    print("="*60)
    print(" VẼ BIỂU ĐỒ CHỨNG MINH MODEL THÔNG MINH LÊN")
    print("="*60)
    plot_metrics(args.log_file, args.output)

import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_jlpt_website(url, output_file):
    # 1. Giả lập trình duyệt (Rất quan trọng để không bị web chặn)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    print(f"Đang truy cập: {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Báo lỗi nếu web sập (mã lỗi 404, 500...)
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải trang: {e}")
        return

    # 2. Đưa HTML thô vào nồi "súp" để bóc tách
    soup = BeautifulSoup(response.text, 'lxml')
    
    # =====================================================================
    # 3. KHU VỰC CẦN SỬA: Tìm các thẻ HTML chứa dữ liệu
    # =====================================================================
    
    # Tìm tất cả các khối (box) chứa bài học ngữ pháp
    grammar_blocks = soup.find_all('div', class_='grammar-box') 
    
    if not grammar_blocks:
        print("Cảnh báo: Không tìm thấy khối dữ liệu nào. Hãy kiểm tra lại class HTML!")
        return

    count = 0
    with open(output_file, 'a', encoding='utf-8') as f:
        for block in grammar_blocks:
            try:
                # Lấy tên cấu trúc (VD: Nằm trong thẻ h3, class 'title')
                title = block.find('h3', class_='title').text.strip()
                
                # Lấy phần giải thích và ví dụ (VD: Nằm trong thẻ div, class 'content')
                content = block.find('div', class_='content').text.strip()
                
                # Đóng gói thành định dạng JSON cho AI học
                data_point = {
                    "instruction": f"Giải thích chi tiết cấu trúc ngữ pháp JLPT N3: {title}",
                    "input": "",
                    "output": content
                }
                
                # Ghi vào file
                f.write(json.dumps(data_point, ensure_ascii=False) + '\n')
                count += 1
                
            except AttributeError:
                # Bỏ qua nếu có khối nào đó bị thiếu thẻ h3 hoặc div
                continue 
                
    print(f"Thành công! Đã cào và lưu {count} cấu trúc ngữ pháp vào {output_file}.")

# 4. Chạy thử nghiệm
if __name__ == "__main__":
    # Thay link này bằng link thật bạn muốn cào
    TARGET_URL = "https://example.com/ngu-phap-n3" 
    
    # File sẽ được lưu trong thư mục data/raw/
    OUTPUT_FILE = "../data/raw/n3_grammar_raw.jsonl"
    
    scrape_jlpt_website(TARGET_URL, OUTPUT_FILE)
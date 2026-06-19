from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def scrape_with_selenium():
    print("Đang khởi động Robot Chrome...")
    # Tự động tải và cài đặt ChromeDriver phù hợp
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    try:
        # 1. Truy cập trang web
        print("Đang truy cập Mazii...")
        driver.get("https://mazii.net/vi-VN/learning-hub/library/jlpt/grammar/n3")
        
        # 2. CHỜ DỮ LIỆU TẢI LÊN (Rất quan trọng với web động)
        # Đợi tối đa 15 giây cho đến khi danh sách ngữ pháp xuất hiện
        # BẠN CẦN ĐỔI '.class-cua-ngu-phap' THÀNH CLASS THẬT TÌM BẰNG F12
        wait = WebDriverWait(driver, 15)
        grammar_items = wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".widget-word-list .word-item")) # Ví dụ class
        )
        
        print(f"Đã tìm thấy {len(grammar_items)} cấu trúc ngữ pháp trên màn hình.")
        
        # 3. CLICK VÀO NGỮ PHÁP ĐẦU TIÊN
        print("Đang click vào ngữ pháp đầu tiên...")
        grammar_items[0].click()
        
        # 4. CHỜ POP-UP HIỆN RA VÀ LẤY CHỮ
        # Đợi thẻ div chứa nội dung pop-up xuất hiện
        popup_content = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".modal-content-class")) # Cần đổi class
        )
        
        print("--- THU HOẠCH DỮ LIỆU ---")
        print(popup_content.text)
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        
    finally:
        # Tạm dừng 5 giây để bạn nhìn thành quả trước khi tắt trình duyệt
        time.sleep(5)
        driver.quit()

if __name__ == "__main__":
    scrape_with_selenium()
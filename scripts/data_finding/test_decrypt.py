from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import base64
import json

def decrypt_mazii_v3(encrypted_string):
    key = b"sHqvxUbI3HRHqHNdluQ5thFw5e8DCglJ"
    
    iv_b64, cipher_b64 = encrypted_string.split(':')
    
    # 3. Chuyển đổi Base64 sang dạng byte thô
    iv = base64.b64decode(iv_b64)
    ciphertext = base64.b64decode(cipher_b64)
    
    # 4. Giải mã bằng thuật toán AES chế độ CBC
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(ciphertext)
    
    # 5. Gỡ bỏ lớp đệm (padding) tự động sinh ra khi mã hóa
    decrypted_bytes = unpad(decrypted_padded, AES.block_size)
    
    # 6. Ép kiểu về cấu trúc JSON
    return json.loads(decrypted_bytes.decode('utf-8'))

# Chạy thử với dữ liệu thật của API v3
data_v3 = "cgxMHsKpHJOdVeosEyI2mw==:J78od8MTzQil9/3ouw9cNkauAN2VUmdzbBxzKL+VfyAGFeLZg24EwVY2ZXK59LeAIGdRAMRkP3yHXwyGHU2fY2zJb++UJ/iCiyBAS9lJXtK+CbHE1OYYIQ8Y/GdKAouH97VoR0rX6CT4BH3U+zVWw15axHXmC7a3i782ZR7t3sQsQZFNw1Wo2vRC73sVp8249MRT7v+jMuiGlhA6FdHS0H/4siajc21LbhdRBCMzu5dEDuEfHMK5Ednjui2NNSzPF2pkEnGPlqLb6PF4EDWG5NKKgUphdlhFZDXj5rqhaoOubjhY4vkDQlGgJ5vbQs//A6qSP8RLN6XFKSA84ttfAw=="

# In kết quả với format JSON đẹp, hỗ trợ tiếng Nhật/Việt
print(json.dumps(decrypt_mazii_v3(data_v3), ensure_ascii=False, indent=2))
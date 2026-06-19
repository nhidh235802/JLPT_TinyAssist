# step3b_vocab_fill_mcq.py
# =============================================================================
# BƯỚC 3B: Tạo MCQ dạng 問題3 — Điền từ vào chỗ trống theo ngữ cảnh
#
# INPUT : data/raw/words.json
# OUTPUT: data/processed/vocab_fill_mcq.jsonl
#
# Đây là dạng bài JLPT thật (問題3):
#   "（　）に入れるのに最もよいものを、1・2・3・4から一つえらびなさい。"
#   Ví dụ:
#   たまねぎとにんじんは皮を（　）から料理に使ってください。
#   1) むいて  2) ほって  3) 破って  4) 離して
#
# Phương pháp:
# - Lấy các từ có câu ví dụ (example sentence).
# - Thay thế từ đó trong câu bằng （　）.
# - Lấy 3 từ khác cùng level JLPT và CÙNG LOẠI TỪ (Part of Speech) làm distractor.
# - Định dạng Output dưới dạng Chain-of-Thought (CoT).
# =============================================================================

import sys
import random
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from helpers import load_json, save_jsonl, make_qa

JLPT_LEVELS = {"N1", "N2", "N3", "N4", "N5"}
RANDOM_SEED = 42
LABELS = ["1", "2", "3", "4"]

def lay_tu_hien_thi(word: dict) -> str:
    kanji_list = word.get("kanji", [])
    kana_list = word.get("kana", [])
    kanji_chinh = next((k["text"] for k in kanji_list if k.get("common")), None)
    kana_chinh = next((k["text"] for k in kana_list if k.get("common")), None)
    return kanji_chinh or kana_chinh or (kanji_list[0]["text"] if kanji_list else kana_list[0]["text"])

def lay_nghia_tieng_anh(word: dict) -> str:
    senses = word.get("sense", [])
    if not senses: return ""
    glosses = [g["text"] for g in senses[0].get("gloss", []) if g.get("lang") == "eng"]
    return ", ".join(glosses[:3])

def xay_dung_index_pos_level(all_words: list) -> dict:
    """Index: (pos, level) -> list_of_words"""
    idx = defaultdict(list)
    for w in all_words:
        lv = w.get("jlpt_waller")
        if lv not in JLPT_LEVELS:
            continue
        senses = w.get("sense", [])
        if not senses:
            continue
        # Chuyển list pos thành tuple để dùng làm dict key (nếu từ có nhiều loại từ, ta lấy loại từ chính đầu tiên)
        pos = tuple(senses[0].get("partOfSpeech", []))
        if pos:
            idx[(pos, lv)].append(w)
    return idx

def tao_fill_mcq(data: dict) -> list[dict]:
    results = []
    all_words = data.get("words", [])
    jlpt_words = [w for w in all_words if w.get("jlpt_waller") in JLPT_LEVELS]

    idx_pos_level = xay_dung_index_pos_level(all_words)
    random.seed(RANDOM_SEED)

    for word in jlpt_words:
        level = word.get("jlpt_waller")
        senses = word.get("sense", [])
        if not senses: continue

        tu_hien_thi = lay_tu_hien_thi(word)
        if not tu_hien_thi: continue
        
        nghia_dung = lay_nghia_tieng_anh(word)
        pos = tuple(senses[0].get("partOfSpeech", []))

        # Tìm câu ví dụ
        cau_vi_du_jp = ""
        cau_vi_du_en = ""
        tu_trong_cau = ""
        for sense in senses:
            for ex in sense.get("examples", []):
                jp = ex.get("japanese", "")
                if tu_hien_thi in jp:
                    cau_vi_du_jp = jp
                    cau_vi_du_en = ex.get("english", "")
                    tu_trong_cau = tu_hien_thi
                    break
                else:
                    # Thử tìm các biến thể kana/kanji trong câu
                    for k in word.get("kanji", []) + word.get("kana", []):
                        if k["text"] in jp:
                            cau_vi_du_jp = jp
                            cau_vi_du_en = ex.get("english", "")
                            tu_trong_cau = k["text"]
                            break
                if cau_vi_du_jp: break
            if cau_vi_du_jp: break
        
        if not cau_vi_du_jp or not tu_trong_cau:
            continue
            
        # Tạo câu có chỗ trống
        cau_hoi_jp = cau_vi_du_jp.replace(tu_trong_cau, "（　　）", 1)

        # Tìm distractors cùng loại từ và cùng JLPT level
        pool = idx_pos_level.get((pos, level), [])
        distractors = []
        for w in pool:
            tu_d = lay_tu_hien_thi(w)
            if tu_d and tu_d != tu_hien_thi and tu_d != tu_trong_cau:
                distractors.append({
                    "word": tu_d,
                    "meaning": lay_nghia_tieng_anh(w)
                })
        
        if len(distractors) < 3:
            continue
            
        random.shuffle(distractors)
        sai_options = distractors[:3]
        
        # Xáo trộn đáp án
        options = [
            {"label": "", "word": tu_hien_thi, "meaning": nghia_dung, "is_correct": True},
            {"label": "", "word": sai_options[0]["word"], "meaning": sai_options[0]["meaning"], "is_correct": False},
            {"label": "", "word": sai_options[1]["word"], "meaning": sai_options[1]["meaning"], "is_correct": False},
            {"label": "", "word": sai_options[2]["word"], "meaning": sai_options[2]["meaning"], "is_correct": False},
        ]
        random.shuffle(options)
        
        correct_label = ""
        for i, opt in enumerate(options):
            opt["label"] = LABELS[i]
            if opt["is_correct"]:
                correct_label = LABELS[i]
        
        # Input
        input_text = f"Sentence: {cau_hoi_jp}\nOptions:\n"
        for opt in options:
            input_text += f"{opt['label']}) {opt['word']}\n"
        input_text = input_text.strip()
        
        # Output CoT
        # <thinking>
        # 1. Context Analysis: The sentence means "{english}". We need a word that fits this context.
        # 2. Option Analysis:
        #    - 1) Word (Meaning): Correct/Incorrect because...
        # ...
        # </thinking>
        # <answer> X </answer>
        
        cot = "<thinking>\n"
        cot += f"1. Context Analysis: The sentence translates to '{cau_vi_du_en}'. We need a vocabulary word that semantically fits into the blank '（　　）'.\n"
        cot += "2. Option Analysis:\n"
        for opt in options:
            status = "Correct" if opt["is_correct"] else "Incorrect"
            reason = "It perfectly matches the intended meaning of the sentence." if opt["is_correct"] else f"The meaning does not fit the context of the sentence."
            cot += f"   - {opt['label']}) {opt['word']} (meaning: {opt['meaning']}): {status}. {reason}\n"
        cot += f"</thinking>\n<answer> {correct_label} </answer>"

        instruction = (
            "問題3: （　　）に入れるのに最もよいものを、1・2・3・4から一つえらびなさい。\n"
            "Solve this JLPT vocabulary question step-by-step. Provide your reasoning in a <thinking> block, and output the final number in an <answer> block."
        )

        results.append(make_qa(instruction, input_text, cot))

    print(f"  Ket qua: {len(results):,} vocab fill MCQ examples")
    return results

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 3B: Tao MCQ dien tu (Mon 3 - Fill-in-the-blank)")
    print("=" * 55)
    raw_data = load_json("words.json")
    records = tao_fill_mcq(raw_data)
    save_jsonl(records, "vocab_fill_mcq.jsonl")
    
    if records:
        print("\n--- Vi du ---")
        print(records[0]["output"])

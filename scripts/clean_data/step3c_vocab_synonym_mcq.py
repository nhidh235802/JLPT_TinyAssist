# step3c_vocab_synonym_mcq.py
# =============================================================================
# BƯỚC 3C: Tạo MCQ dạng 問題4 — Tìm từ đồng nghĩa (Synonyms)
#
# INPUT : data/raw/words.json
# OUTPUT: data/processed/vocab_synonym_mcq.jsonl
#
# Đây là dạng bài JLPT thật (問題4):
#   "___に意味が最も近いものを、1・2・3・4から一つえらびなさい。"
#   Ví dụ:
#   あの人は【短気】だ。
#   1) すぐ怒る  2) すぐ謝る  3) すぐ驚く  4) すぐ喜ぶ
#
# Phương pháp:
# - Tìm các cặp từ có chung nghĩa tiếng Anh (trường gloss).
# - 1 từ làm câu hỏi, 1 từ làm đáp án đúng.
# - Chọn 3 từ khác không chung nghĩa làm distractor.
# - Định dạng CoT.
# =============================================================================

import sys
import random
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

def lay_tat_ca_nghia_tieng_anh(word: dict) -> set[str]:
    nghia = set()
    for sense in word.get("sense", []):
        for g in sense.get("gloss", []):
            if g.get("lang") == "eng":
                nghia.add(g.get("text").lower())
    return nghia

def lay_nghia_rut_gon(word: dict) -> str:
    senses = word.get("sense", [])
    if not senses: return ""
    glosses = [g["text"] for g in senses[0].get("gloss", []) if g.get("lang") == "eng"]
    return ", ".join(glosses[:3])

def tao_synonym_mcq(data: dict) -> list[dict]:
    results = []
    all_words = data.get("words", [])
    
    # Chỉ dùng từ có jlpt level
    jlpt_words = [w for w in all_words if w.get("jlpt_waller") in JLPT_LEVELS]
    
    # Xây dựng index theo nghĩa tiếng Anh
    gloss_index = defaultdict(list)
    level_index = defaultdict(list)
    
    for w in jlpt_words:
        lv = w.get("jlpt_waller")
        level_index[lv].append(w)
        nghia = lay_tat_ca_nghia_tieng_anh(w)
        for g in nghia:
            gloss_index[g].append(w)
            
    random.seed(RANDOM_SEED)
    
    seen_pairs = set()

    for word in jlpt_words:
        tu_hien_thi = lay_tu_hien_thi(word)
        if not tu_hien_thi: continue
        
        nghia_set = lay_tat_ca_nghia_tieng_anh(word)
        level = word.get("jlpt_waller")
        
        # Tìm từ đồng nghĩa
        synonyms = []
        for g in nghia_set:
            for w_syn in gloss_index.get(g, []):
                tu_syn = lay_tu_hien_thi(w_syn)
                if tu_syn and tu_syn != tu_hien_thi:
                    synonyms.append(w_syn)
                    
        if not synonyms:
            continue
            
        # Lấy từ đồng nghĩa đầu tiên làm đáp án
        w_correct = random.choice(synonyms)
        tu_correct = lay_tu_hien_thi(w_correct)
        
        pair_key = tuple(sorted([tu_hien_thi, tu_correct]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        
        nghia_correct_set = lay_tat_ca_nghia_tieng_anh(w_correct)
        
        # Lấy câu ví dụ của từ gốc để tạo câu hỏi (nếu có)
        cau_vi_du_jp = ""
        tu_trong_cau = tu_hien_thi
        for sense in word.get("sense", []):
            for ex in sense.get("examples", []):
                jp = ex.get("japanese", "")
                if tu_hien_thi in jp:
                    cau_vi_du_jp = jp
                    break
            if cau_vi_du_jp: break
            
        # Tìm distractors (không đồng nghĩa)
        pool = level_index.get(level, [])
        distractors = []
        for w_dist in pool:
            tu_dist = lay_tu_hien_thi(w_dist)
            if tu_dist and tu_dist != tu_hien_thi and tu_dist != tu_correct:
                nghia_dist = lay_tat_ca_nghia_tieng_anh(w_dist)
                # Đảm bảo không trùng nghĩa
                if not nghia_set.intersection(nghia_dist):
                    distractors.append(w_dist)
        
        if len(distractors) < 3:
            continue
            
        random.shuffle(distractors)
        sai_options = distractors[:3]
        
        options = [
            {"label": "", "word": tu_correct, "meaning": lay_nghia_rut_gon(w_correct), "is_correct": True},
            {"label": "", "word": lay_tu_hien_thi(sai_options[0]), "meaning": lay_nghia_rut_gon(sai_options[0]), "is_correct": False},
            {"label": "", "word": lay_tu_hien_thi(sai_options[1]), "meaning": lay_nghia_rut_gon(sai_options[1]), "is_correct": False},
            {"label": "", "word": lay_tu_hien_thi(sai_options[2]), "meaning": lay_nghia_rut_gon(sai_options[2]), "is_correct": False},
        ]
        random.shuffle(options)
        
        correct_label = ""
        for i, opt in enumerate(options):
            opt["label"] = LABELS[i]
            if opt["is_correct"]:
                correct_label = LABELS[i]
                
        # Nếu có câu ví dụ, dùng format câu. Nếu không, chỉ hỏi từ gốc.
        if cau_vi_du_jp:
            cau_hoi_jp = cau_vi_du_jp.replace(tu_trong_cau, f"【{tu_trong_cau}】", 1)
            input_text = f"Sentence: {cau_hoi_jp}\nOptions:\n"
        else:
            input_text = f"Word: 【{tu_hien_thi}】\nOptions:\n"
            
        for opt in options:
            input_text += f"{opt['label']}) {opt['word']}\n"
        input_text = input_text.strip()
        
        # CoT Output
        cot = "<thinking>\n"
        cot += f"1. Target Analysis: The underlined word is 【{tu_hien_thi}】, which means '{lay_nghia_rut_gon(word)}'. We need to find the option with the closest meaning.\n"
        cot += "2. Option Analysis:\n"
        for opt in options:
            status = "Correct" if opt["is_correct"] else "Incorrect"
            reason = f"It means '{opt['meaning']}', which is synonymous with the target word." if opt["is_correct"] else f"It means '{opt['meaning']}', which is entirely different."
            cot += f"   - {opt['label']}) {opt['word']}: {status}. {reason}\n"
        cot += f"</thinking>\n<answer> {correct_label} </answer>"

        instruction = (
            "問題4: 【 】の言葉に意味が最も近いものを、1・2・3・4から一つえらびなさい。\n"
            "Solve this JLPT vocabulary question step-by-step. Provide your reasoning in a <thinking> block, and output the final number in an <answer> block."
        )

        results.append(make_qa(instruction, input_text, cot))

    print(f"  Ket qua: {len(results):,} vocab synonym MCQ examples")
    return results

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 3C: Tao MCQ dong nghia (Mon 4 - Synonyms)")
    print("=" * 55)
    raw_data = load_json("words.json")
    records = tao_synonym_mcq(raw_data)
    save_jsonl(records, "vocab_synonym_mcq.jsonl")
    
    if records:
        print("\n--- Vi du ---")
        print(records[0]["output"])

# step3d_vocab_usage_mcq.py
# =============================================================================
# BƯỚC 3D: Tạo MCQ dạng 問題5 — Cách sử dụng từ (Usage)
#
# INPUT : data/raw/words.json
# OUTPUT: data/processed/vocab_usage_mcq.jsonl
#
# Đây là dạng bài JLPT thật (問題5):
#   "つぎのことばの使い方として最もよいものを、1・2・3・4から一つえらびなさい。"
#   Ví dụ:
#   整理
#   1) 机の引き出しを整理して、いらない物を捨てました。(Đúng)
#   2) 家の廊下が汚れていたので、ぞうきんで整理しました。(Sai - phải là 掃除)
#   ...
#
# Phương pháp:
# - Đáp án đúng: Lấy câu ví dụ gốc của từ đó.
# - Đáp án sai: Lấy câu ví dụ của TỪ KHÁC (cùng loại từ, cùng level), sau đó thay thế
#   từ khác đó bằng từ đang xét → tạo ra câu sai ngữ cảnh.
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

def lay_nghia_tieng_anh(word: dict) -> str:
    senses = word.get("sense", [])
    if not senses: return ""
    glosses = [g["text"] for g in senses[0].get("gloss", []) if g.get("lang") == "eng"]
    return ", ".join(glosses[:3])

def xay_dung_index_pos_level_sentences(all_words: list) -> dict:
    """Index: (pos, level) -> list_of_sentences_info"""
    idx = defaultdict(list)
    for w in all_words:
        lv = w.get("jlpt_waller")
        if lv not in JLPT_LEVELS:
            continue
        senses = w.get("sense", [])
        if not senses:
            continue
        pos = tuple(senses[0].get("partOfSpeech", []))
        tu_hien_thi = lay_tu_hien_thi(w)
        if not pos or not tu_hien_thi:
            continue
            
        for sense in senses:
            for ex in sense.get("examples", []):
                jp = ex.get("japanese", "")
                en = ex.get("english", "")
                if tu_hien_thi in jp:
                    idx[(pos, lv)].append({
                        "word": tu_hien_thi,
                        "meaning": lay_nghia_tieng_anh(w),
                        "sentence_jp": jp,
                        "sentence_en": en
                    })
    return idx

def tao_usage_mcq(data: dict) -> list[dict]:
    results = []
    all_words = data.get("words", [])
    jlpt_words = [w for w in all_words if w.get("jlpt_waller") in JLPT_LEVELS]
    
    idx_sentences = xay_dung_index_pos_level_sentences(all_words)
    random.seed(RANDOM_SEED)

    for word in jlpt_words:
        level = word.get("jlpt_waller")
        senses = word.get("sense", [])
        if not senses: continue

        tu_hien_thi = lay_tu_hien_thi(word)
        if not tu_hien_thi: continue
        
        pos = tuple(senses[0].get("partOfSpeech", []))
        nghia_dung = lay_nghia_tieng_anh(word)
        
        # Tìm câu ví dụ đúng
        cau_dung = None
        for sense in senses:
            for ex in sense.get("examples", []):
                jp = ex.get("japanese", "")
                if tu_hien_thi in jp:
                    cau_dung = {
                        "word": tu_hien_thi,
                        "meaning": nghia_dung,
                        "sentence_jp": jp,
                        "sentence_en": ex.get("english", "")
                    }
                    break
            if cau_dung: break
            
        if not cau_dung:
            continue
            
        # Tìm 3 câu sai bằng cách lấy câu của từ khác và thay thế
        pool = idx_sentences.get((pos, level), [])
        distractors = []
        for d in pool:
            if d["word"] != tu_hien_thi:
                # Thay thế từ gốc trong câu đó bằng từ hiện tại
                cau_sai_jp = d["sentence_jp"].replace(d["word"], tu_hien_thi, 1)
                distractors.append({
                    "original_word": d["word"],
                    "original_meaning": d["meaning"],
                    "sentence_jp": cau_sai_jp
                })
                
        if len(distractors) < 3:
            continue
            
        random.shuffle(distractors)
        sai_options = distractors[:3]
        
        options = [
            {"label": "", "sentence": cau_dung["sentence_jp"], "is_correct": True, "detail": cau_dung},
            {"label": "", "sentence": sai_options[0]["sentence_jp"], "is_correct": False, "detail": sai_options[0]},
            {"label": "", "sentence": sai_options[1]["sentence_jp"], "is_correct": False, "detail": sai_options[1]},
            {"label": "", "sentence": sai_options[2]["sentence_jp"], "is_correct": False, "detail": sai_options[2]},
        ]
        random.shuffle(options)
        
        correct_label = ""
        for i, opt in enumerate(options):
            opt["label"] = LABELS[i]
            if opt["is_correct"]:
                correct_label = LABELS[i]
                
        input_text = f"Word: 【{tu_hien_thi}】\nOptions:\n"
        for opt in options:
            input_text += f"{opt['label']}) {opt['sentence']}\n"
        input_text = input_text.strip()
        
        # CoT Output
        cot = "<thinking>\n"
        cot += f"1. Target Word Analysis: The word is 【{tu_hien_thi}】, which means '{nghia_dung}'. We need to find the sentence where this word is used correctly.\n"
        cot += "2. Option Analysis:\n"
        for opt in options:
            status = "Correct" if opt["is_correct"] else "Incorrect"
            if opt["is_correct"]:
                reason = f"The word is used properly in this context. The sentence translates to '{opt['detail']['sentence_en']}'."
            else:
                orig_w = opt["detail"]["original_word"]
                orig_m = opt["detail"]["original_meaning"]
                reason = f"The context actually requires a word like '{orig_w}' ({orig_m}). Using '{tu_hien_thi}' here is unnatural or grammatically incorrect."
            cot += f"   - {opt['label']}): {status}. {reason}\n"
        cot += f"</thinking>\n<answer> {correct_label} </answer>"

        instruction = (
            "問題5: つぎのことばの使い方として最もよいものを、1・2・3・4から一つえらびなさい。\n"
            "Solve this JLPT vocabulary question step-by-step. Provide your reasoning in a <thinking> block, and output the final number in an <answer> block."
        )

        results.append(make_qa(instruction, input_text, cot))

    print(f"  Ket qua: {len(results):,} vocab usage MCQ examples")
    return results

if __name__ == "__main__":
    print("=" * 55)
    print("BUOC 3D: Tao MCQ cach dung tu (Mon 5 - Usage)")
    print("=" * 55)
    raw_data = load_json("words.json")
    records = tao_usage_mcq(raw_data)
    save_jsonl(records, "vocab_usage_mcq.jsonl")
    
    if records:
        print("\n--- Vi du ---")
        print(records[0]["output"])

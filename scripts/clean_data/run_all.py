# run_all.py
# =============================================================================
# CHẠY TẤT CẢ CÁC BƯỚC TRONG 1 LỆNH
#
# Thay vì chạy từng file:
#   python step1_grammar.py
#   python step2_kanji.py
#   python step3_words.py
#   python step4_combine.py
#   python step5_check.py
#
# Chỉ cần chạy:
#   python scripts/clean_data/run_all.py
#
# Sơ đồ pipeline:
#
#   grammar.json ──┐
#   kanji.json ────┼──► [step1-3] tao Q&A ──► [step4] gop + chia ──► [step5] kiem tra
#   words.json ────┘
#
# =============================================================================

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

def chay_buoc(ten: str, ham_thuc_thi):
    """Chạy 1 bước, đo thời gian, báo lỗi nếu có."""
    print(f"\n{'='*55}")
    print(f"  {ten}")
    print(f"{'='*55}")
    bat_dau = time.time()
    try:
        ham_thuc_thi()
        thoi_gian = time.time() - bat_dau
        print(f"\n  [XONG] Mat {thoi_gian:.1f} giay")
    except Exception as e:
        print(f"\n  [LOI] {ten} that bai: {e}")
        raise


# Import các hàm xử lý chính từ từng bước
from helpers      import load_json, save_jsonl
from step1_grammar import xu_ly_ngu_phap
from step1b_grammar_mcq import tao_mcq_grammar
from step2_kanji   import xu_ly_kanji
from step2b_kanji_reading_mcq import tao_reading_mcq
from step2c_kanji_writing_mcq import tao_writing_mcq
from step3_words   import xu_ly_tu_vung
from step3b_vocab_fill_mcq import tao_fill_mcq
from step3c_vocab_synonym_mcq import tao_synonym_mcq
from step3d_vocab_usage_mcq import tao_usage_mcq
from step4_combine import gop_va_chia, luu_thong_ke, doc_jsonl as doc_processed
from step5_check   import kiem_tra_file

def buoc_1():
    data = load_json("grammar.json")
    save_jsonl(xu_ly_ngu_phap(data), "grammar_qa.jsonl")
    save_jsonl(tao_mcq_grammar(data), "grammar_mcq.jsonl")

def buoc_2():
    data = load_json("kanji.json")
    save_jsonl(xu_ly_kanji(data), "kanji_qa.jsonl")

def buoc_2_vocab():
    data = load_json("words.json")
    save_jsonl(tao_reading_mcq(data), "kanji_reading_mcq.jsonl")
    save_jsonl(tao_writing_mcq(data), "kanji_writing_mcq.jsonl")

def buoc_3():
    data = load_json("words.json")
    save_jsonl(xu_ly_tu_vung(data), "words_qa.jsonl")
    save_jsonl(tao_fill_mcq(data), "vocab_fill_mcq.jsonl")
    save_jsonl(tao_synonym_mcq(data), "vocab_synonym_mcq.jsonl")
    save_jsonl(tao_usage_mcq(data), "vocab_usage_mcq.jsonl")

def buoc_4():
    g   = doc_processed("grammar_qa.jsonl")
    g_m = doc_processed("grammar_mcq.jsonl")
    k   = doc_processed("kanji_qa.jsonl")
    w   = doc_processed("words_qa.jsonl")
    r_m = doc_processed("kanji_reading_mcq.jsonl")
    w_m = doc_processed("kanji_writing_mcq.jsonl")
    f_m = doc_processed("vocab_fill_mcq.jsonl")
    s_m = doc_processed("vocab_synonym_mcq.jsonl")
    u_m = doc_processed("vocab_usage_mcq.jsonl")

    train, eval_ = gop_va_chia(g, g_m, k, w, r_m, w_m, f_m, s_m, u_m)
    save_jsonl(train, "combined_train.jsonl")
    save_jsonl(eval_,  "combined_eval.jsonl")
    luu_thong_ke(g, g_m, k, w, r_m, w_m, f_m, s_m, u_m, train, eval_)

def buoc_5():
    kiem_tra_file("combined_train.jsonl")
    kiem_tra_file("combined_eval.jsonl")

if __name__ == "__main__":
    print("\n" + "#" * 55)
    print("#  JLPT DATA PIPELINE - CHAY TAT CA CAC BUOC  #")
    print("#" * 55)

    tat_ca_bat_dau = time.time()

    chay_buoc("BUOC 1: Xu ly grammar", buoc_1)
    chay_buoc("BUOC 2: Xu ly kanji",   buoc_2)
    chay_buoc("BUOC 2b/2c: Xu ly kanji MCQ (tu words.json)", buoc_2_vocab)
    chay_buoc("BUOC 3: Xu ly words & vocab MCQ",   buoc_3)
    chay_buoc("BUOC 4: Gop va chia train/eval", buoc_4)
    chay_buoc("BUOC 5: Kiem tra data",      buoc_5)

    tong_thoi_gian = time.time() - tat_ca_bat_dau

    print("\n" + "#" * 55)
    print(f"#  HOAN THANH! Tong thoi gian: {tong_thoi_gian:.0f} giay  #")
    print("#" * 55)
    print("\nCac file da tao:")
    print("  data/processed/*.jsonl")
    print("  data/processed/combined_train.jsonl  <- dung de TRAIN")
    print("  data/processed/combined_eval.jsonl   <- dung de DANH GIA")
    print("  data/processed/stats.json            <- thong ke tong quat")
    print("\nBuoc tiep theo: upload combined_train.jsonl len Google Colab de train!")

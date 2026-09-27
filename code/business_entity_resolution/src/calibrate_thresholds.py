import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import find_dataset_dir, load_tsv, load_ground_truth
from preprocessing import normalize_name, normalize_address, normalize_country
from blocking import MultiKeyBlocker
from features import FeatureExtractor
from model import EntityMatchingModel
from evaluate import compute_macro_f05

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=15000, help="Validation sample size")
    parser.add_argument("--model-path", type=str, default="models/matching_model.pkl")
    parser.add_argument("--out-path", type=str, default="models/calibrated_thresholds.json")
    args = parser.parse_args()
    
    data_dir = find_dataset_dir(None)
    train_dir = os.path.join(data_dir, "train")
    
    print("Loading Ground Truth...")
    gt_map = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    print(f"Loading {args.sample_size:,} validation entities from train_source1.tsv...")
    df_s1 = load_tsv(os.path.join(train_dir, "train_source1.tsv"), nrows=args.sample_size)
    df_s1["country"] = df_s1["country"].apply(normalize_country)
    df_s1["norm_name"] = df_s1["business_name"].apply(normalize_name)
    s1_addr = df_s1["business_address"].apply(normalize_address)
    df_s1["norm_addr"] = [r[0] for r in s1_addr]
    df_s1["postal_code"] = [r[1] for r in s1_addr]
    df_s1["house_num"] = [r[2] for r in s1_addr]
    
    s1_dict = dict(zip(
        df_s1["entity_id"].values,
        zip(
            df_s1["norm_name"].values,
            df_s1["norm_addr"].values,
            df_s1["country"].values,
            df_s1["postal_code"].values,
            df_s1["house_num"].values
        )
    ))
    eval_ids = list(s1_dict.keys())
    
    print("Loading target records (Source 2 & 3)...")
    df_s2 = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
    df_s3 = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    
    df_targets["country"] = df_targets["country"].apply(normalize_country)
    df_targets["norm_name"] = df_targets["business_name"].apply(normalize_name)
    t_addr = df_targets["business_address"].apply(normalize_address)
    df_targets["norm_addr"] = [r[0] for r in t_addr]
    df_targets["postal_code"] = [r[1] for r in t_addr]
    df_targets["house_num"] = [r[2] for r in t_addr]
    
    target_dict = dict(zip(
        df_targets["entity_id"].values,
        zip(
            df_targets["norm_name"].values,
            df_targets["norm_addr"].values,
            df_targets["country"].values,
            df_targets["postal_code"].values,
            df_targets["house_num"].values
        )
    ))
    
    print("Indexing target records in blocker...")
    blocker = MultiKeyBlocker(max_candidates_per_entity=35)
    blocker.index_target_records(df_targets)
    del df_targets
    
    print(f"Loading trained model: {args.model_path}...")
    model = EntityMatchingModel.load(args.model_path)
    
    print("Scoring candidates for validation set...")
    val_scores_map = {}
    for s1_id in eval_ids:
        s1_data = s1_dict[s1_id]
        cands = blocker.get_candidates_for_s1(s1_data[2], s1_data[0], s1_data[3], s1_data[4], s1_data[1])
        if not cands:
            val_scores_map[s1_id] = []
            continue
            
        cand_feats, valid_cands = [], []
        for cid in cands:
            if cid in target_dict:
                t_data = target_dict[cid]
                cand_feats.append(FeatureExtractor.extract_pair_features(
                    s1_data[0], s1_data[1], s1_data[2], s1_data[3], s1_data[4],
                    t_data[0], t_data[1], t_data[2], t_data[3], t_data[4],
                    target_id=cid
                ))
                valid_cands.append(cid)
                
        if cand_feats:
            probs = model.predict_proba(np.array(cand_feats))
            pairs = sorted(zip(valid_cands, probs), key=lambda x: x[1], reverse=True)
            val_scores_map[s1_id] = pairs
        else:
            val_scores_map[s1_id] = []
            
    print("\nRunning Grid Search over Gate, Sibling Floor, Margin, and MaxPerSource...")
    best_gate = 0.50
    best_sib = 0.30
    best_margin = 0.30
    best_max = 3
    best_score = -1.0
    
    sub_gt = {k: gt_map.get(k, set()) for k in eval_ids}
    
    print(f"{'Gate':<8} | {'Sibling':<8} | {'Margin':<8} | {'MaxSrc':<6} | {'Macro F0.5':<10}")
    print("-" * 52)
    
    for gate in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for sib in [0.15, 0.20, 0.25, 0.30, 0.35]:
            if sib > gate:
                continue
            for margin in [0.20, 0.25, 0.30, 0.35, 0.40]:
                for max_src in [3, 4]:
                    preds = {}
                    for s1_id in eval_ids:
                        pairs = val_scores_map[s1_id]
                        if not pairs:
                            preds[s1_id] = set()
                            continue
                        max_p = pairs[0][1]
                        if max_p >= gate:
                            cutoff = max(sib, max_p - margin)
                            passing = [cid for cid, p in pairs if p >= cutoff]
                            
                            selected = []
                            s2_cnt, s3_cnt = 0, 0
                            for cid in passing:
                                if cid.startswith("S2-"):
                                    if s2_cnt < max_src:
                                        selected.append(cid)
                                        s2_cnt += 1
                                elif cid.startswith("S3-"):
                                    if s3_cnt < max_src:
                                        selected.append(cid)
                                        s3_cnt += 1
                                else:
                                    selected.append(cid)
                            preds[s1_id] = set(selected)
                        else:
                            preds[s1_id] = set()
                            
                    score = compute_macro_f05(preds, sub_gt)
                    if score > best_score:
                        best_score = score
                        best_gate = gate
                        best_sib = sib
                        best_margin = margin
                        best_max = max_src
                        print(f"{gate:<8.2f} | {sib:<8.2f} | {margin:<8.2f} | {max_src:<6} | {score:<10.4f} *")
                        
    print("-" * 52)
    print(f"Optimal Parameters: Gate={best_gate:.2f}, SiblingFloor={best_sib:.2f}, Margin={best_margin:.2f}, MaxSrc={best_max}")
    print(f"Peak Validation Macro F0.5: {best_score:.4f}")
    
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({
            "gate_threshold": best_gate,
            "sibling_threshold": best_sib,
            "best_threshold": best_gate,
            "best_margin": best_margin,
            "max_per_source": best_max,
            "val_f05": best_score
        }, f, indent=2)
    print(f"Saved optimal parameters to: {args.out_path}")

if __name__ == "__main__":
    main()

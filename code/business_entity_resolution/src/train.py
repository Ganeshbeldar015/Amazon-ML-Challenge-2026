import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import find_dataset_dir, load_tsv, load_ground_truth
from preprocessing import normalize_name, normalize_address, normalize_country
from blocking import MultiKeyBlocker
from features import FeatureExtractor
from model import EntityMatchingModel
from evaluate import compute_macro_f05

def parse_args():
    parser = argparse.ArgumentParser(description="State-of-the-Art Anti-Overfitting Entity Matching Training")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--model-out", type=str, default="models/matching_model.pkl", help="Model output path")
    parser.add_argument("--threshold-out", type=str, default="models/best_threshold.json", help="Threshold output path")
    parser.add_argument("--sample-size", type=int, default=80000, help="Number of S1 train records (default: 80,000)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    data_dir = find_dataset_dir(args.data_dir)
    train_dir = os.path.join(data_dir, "train")
    print(f"Dataset root: {train_dir}")
    
    # 1. Load Ground Truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print("Loading ground truth...")
    gt_map = load_ground_truth(gt_path)
    
    # 2. Load Source 1
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    nrows = args.sample_size if args.sample_size > 0 else None
    print(f"Loading Source 1 (sample_size={args.sample_size})...")
    df_s1 = load_tsv(s1_path, nrows=nrows)
    
    print("Preprocessing Source 1...")
    df_s1["country"] = df_s1["country"].apply(normalize_country)
    df_s1["norm_name"] = df_s1["business_name"].apply(normalize_name)
    addr_res = df_s1["business_address"].apply(normalize_address)
    df_s1["norm_addr"] = [r[0] for r in addr_res]
    df_s1["postal_code"] = [r[1] for r in addr_res]
    df_s1["house_num"] = [r[2] for r in addr_res]
    
    # 3. Train/Val Split
    s1_ids = df_s1["entity_id"].tolist()
    random.shuffle(s1_ids)
    split_idx = int(len(s1_ids) * (1.0 - args.val_ratio))
    train_s1_ids = set(s1_ids[:split_idx])
    val_s1_ids = set(s1_ids[split_idx:])
    print(f"Train S1: {len(train_s1_ids):,} | Val S1: {len(val_s1_ids):,}")
    
    # 4. Load Source 2 & Source 3 targets
    print("Loading Source 2 and Source 3...")
    df_s2 = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
    df_s3 = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    
    print("Preprocessing Source 2 & 3...")
    df_targets["country"] = df_targets["country"].apply(normalize_country)
    df_targets["norm_name"] = df_targets["business_name"].apply(normalize_name)
    t_addr_res = df_targets["business_address"].apply(normalize_address)
    df_targets["norm_addr"] = [r[0] for r in t_addr_res]
    df_targets["postal_code"] = [r[1] for r in t_addr_res]
    df_targets["house_num"] = [r[2] for r in t_addr_res]
    
    print("Indexing target records into High-Recall Blocker...")
    blocker = MultiKeyBlocker(max_candidates_per_entity=35)
    blocker.index_target_records(df_targets)
    
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
    del df_targets
    
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
    del df_s1
    
    # 5. Build Training Set with Balanced Hard-Negative Mining
    print("Mining Positive Pairs and Realistic Multi-Aspect Negatives...")
    X_train, y_train = [], []
    MAX_TRAIN_PAIRS = 3000000  # Optimal GBDT convergence cap (~540MB RAM in float32)
    
    for s1_id in train_s1_ids:
        if len(y_train) >= MAX_TRAIN_PAIRS:
            print(f"Reached optimal training sample ceiling ({len(y_train):,} pairs) for maximum GBDT convergence and memory safety.")
            break
        s1_data = s1_dict[s1_id]
        true_matches = gt_map.get(s1_id, set())
        
        # Add True Positives
        for tid in true_matches:
            if tid in target_dict:
                t_data = target_dict[tid]
                feats = FeatureExtractor.extract_pair_features(
                    s1_data[0], s1_data[1], s1_data[2], s1_data[3], s1_data[4],
                    t_data[0], t_data[1], t_data[2], t_data[3], t_data[4],
                    target_id=tid
                )
                X_train.append(feats)
                y_train.append(1)
                
        # Mine negatives from blocker candidates
        cands = blocker.get_candidates_for_s1(s1_data[2], s1_data[0], s1_data[3], s1_data[4], s1_data[1])
        neg_cands = [c for c in cands if c not in true_matches and c in target_dict]
        
        if not neg_cands:
            continue
            
        # 1. Hard negatives by name similarity (discriminating near-name collisions)
        hard_name_negs = []
        same_postal_negs = []
        random_negs = []
        
        for cid in neg_cands:
            t_data = target_dict[cid]
            name_sim = fuzz.token_sort_ratio(s1_data[0], t_data[0]) / 100.0
            if name_sim >= 0.40:
                hard_name_negs.append((cid, name_sim))
            if s1_data[3] and t_data[3] and s1_data[3] == t_data[3]:
                same_postal_negs.append(cid)
            else:
                random_negs.append(cid)
                
        hard_name_negs.sort(key=lambda x: x[1], reverse=True)
        
        selected_negs = set()
        # Top 4 hard name negatives
        for cid, _ in hard_name_negs[:4]:
            selected_negs.add(cid)
            
        # Up to 2 same-postal collisions (shopping mall / multi-tenant building)
        for cid in same_postal_negs[:2]:
            selected_negs.add(cid)
            
        # Up to 2 random negatives (blocker background noise)
        if random_negs and len(selected_negs) < 6:
            for cid in random.sample(random_negs, min(len(random_negs), 2)):
                selected_negs.add(cid)
                
        for cid in selected_negs:
            t_data = target_dict[cid]
            feats = FeatureExtractor.extract_pair_features(
                s1_data[0], s1_data[1], s1_data[2], s1_data[3], s1_data[4],
                t_data[0], t_data[1], t_data[2], t_data[3], t_data[4],
                target_id=cid
            )
            X_train.append(feats)
            y_train.append(0)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int8)
    print(f"Constructed {len(y_train):,} training pairs (Pos: {np.sum(y_train==1):,}, Neg: {np.sum(y_train==0):,})")
    
    # 6. Fit LightGBM Model with Regularization
    print("Training High-Capacity Regularized LightGBM classifier (350 trees, depth 8, 63 leaves)...")
    model = EntityMatchingModel(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=8,
        num_leaves=63,
        min_child_samples=40,
        reg_alpha=0.05,
        reg_lambda=0.5
    )
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    model.save(args.model_out)
    print(f"Model saved to: {args.model_out}")
    
    # 7. Comprehensive Calibration Grid Search on Validation Split
    print("\nRunning Validation Macro-F0.5 Threshold & Margin Search...")
    val_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
    val_scores_map = {}
    
    val_eval_ids = list(val_s1_ids)[:15000]
    
    for s1_id in val_eval_ids:
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
            
    best_gate = 0.60
    best_sibling = 0.35
    best_margin = 0.25
    best_max_per_src = 3
    best_f05 = -1.0
    
    sub_gt = {k: val_gt[k] for k in val_eval_ids}
    
    print("-" * 65)
    print(f"{'GateThresh':<12} | {'SiblingThresh':<14} | {'Margin':<10} | {'MaxSrc':<8} | {'Macro F0.5':<12}")
    print("-" * 65)
    
    for gate_t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        gate_t = round(float(gate_t), 2)
        for sib_t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            sib_t = round(float(sib_t), 2)
            if sib_t > gate_t:
                continue
            for margin in [0.15, 0.20, 0.25, 0.30, 0.35]:
                margin = round(float(margin), 2)
                for max_per_src in [2, 3, 4]:
                    preds = {}
                    for s1_id in val_eval_ids:
                        pairs = val_scores_map[s1_id]
                        if not pairs:
                            preds[s1_id] = set()
                            continue
                        max_p = pairs[0][1]
                        if max_p >= gate_t:
                            cutoff = max(sib_t, max_p - margin)
                            passing = [cid for cid, p in pairs if p >= cutoff]
                            
                            selected = []
                            s2_cnt, s3_cnt = 0, 0
                            for cid in passing:
                                if cid.startswith("S2-"):
                                    if s2_cnt < max_per_src:
                                        selected.append(cid)
                                        s2_cnt += 1
                                elif cid.startswith("S3-"):
                                    if s3_cnt < max_per_src:
                                        selected.append(cid)
                                        s3_cnt += 1
                                else:
                                    selected.append(cid)
                            preds[s1_id] = set(selected)
                        else:
                            preds[s1_id] = set()
                            
                    score = compute_macro_f05(preds, sub_gt)
                    if score > best_f05:
                        best_f05 = score
                        best_gate = gate_t
                        best_sibling = sib_t
                        best_margin = margin
                        best_max_per_src = max_per_src
                        print(f"{gate_t:<12.2f} | {sib_t:<14.2f} | {margin:<10.2f} | {max_per_src:<8} | {score:<12.4f} *")
                        
    print("-" * 65)
    print(f"Optimal Configuration: Gate={best_gate:.2f}, SiblingFloor={best_sibling:.2f}, Margin={best_margin:.2f}, MaxPerSource={best_max_per_src}")
    print(f"Best Validation Macro F0.5: {best_f05:.4f}")
    
    with open(args.threshold_out, "w", encoding="utf-8") as f:
        json.dump({
            "gate_threshold": best_gate,
            "sibling_threshold": best_sibling,
            "best_threshold": best_gate,
            "best_margin": best_margin,
            "max_per_source": best_max_per_src,
            "val_f05": best_f05
        }, f, indent=2)
        
    print(f"Optimal calibration saved to: {args.threshold_out}")
    print("Training and Calibration Complete!")

if __name__ == "__main__":
    main()

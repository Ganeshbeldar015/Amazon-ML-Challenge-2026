import os
import sys
import argparse
import random
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import find_dataset_dir, load_tsv, load_ground_truth
from preprocessing import normalize_name, normalize_address, normalize_country
from blocking import MultiKeyBlocker

def parse_args():
    parser = argparse.ArgumentParser(description="Blocking Recall Audit & Country Normalization Diagnostic")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--sample-size", type=int, default=30000, help="S1 sample size for audit (default: 30,000)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2)")
    parser.add_argument("--max-candidates", type=int, default=35, help="Max candidates per entity (default: 35)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def check_country_distributions(train_dir: str, test_dir: str):
    print("\n" + "=" * 70)
    print("STEP 9 DIAGNOSTIC: COUNTRY STRING NORMALIZATION AUDIT")
    print("=" * 70)
    
    datasets = {
        "Train S1": os.path.join(train_dir, "train_source1.tsv"),
        "Train S2": os.path.join(train_dir, "train_source2.tsv"),
        "Train S3": os.path.join(train_dir, "train_source3.tsv"),
        "Test S1": os.path.join(test_dir, "test_source1.tsv"),
        "Test S2": os.path.join(test_dir, "test_source2.tsv"),
        "Test S3": os.path.join(test_dir, "test_source3.tsv"),
    }
    
    country_sets = {}
    for name, path in datasets.items():
        if os.path.isfile(path):
            df = load_tsv(path, usecols=["country"], nrows=50000)
            norm_countries = set(df["country"].apply(normalize_country).unique())
            country_sets[name] = norm_countries
            print(f"  {name:<10}: {sorted(norm_countries)}")
        else:
            print(f"  {name:<10}: File not found ({path})")
            
    # Check mismatches
    print("\n  Cross-Source Alignment Checks:")
    if "Train S1" in country_sets and "Train S2" in country_sets and "Train S3" in country_sets:
        s1_train = country_sets["Train S1"]
        target_train = country_sets["Train S2"] | country_sets["Train S3"]
        mismatch_train = s1_train - target_train
        if mismatch_train:
            print(f"  [WARNING] Train S1 has countries not in Targets: {mismatch_train}")
        else:
            print("  [OK] Train S1 countries are 100% covered in Train Targets.")
            
    if "Test S1" in country_sets and "Test S2" in country_sets and "Test S3" in country_sets:
        s1_test = country_sets["Test S1"]
        target_test = country_sets["Test S2"] | country_sets["Test S3"]
        mismatch_test = s1_test - target_test
        if mismatch_test:
            print(f"  [ALERT] Test S1 has countries not in Test Targets: {mismatch_test}")
        else:
            print("  [OK] Test S1 countries are 100% covered in Test Targets (including France).")
    print("=" * 70)

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    data_dir = find_dataset_dir(args.data_dir)
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    # 1. Run Country Diagnostic
    check_country_distributions(train_dir, test_dir)
    
    # 2. Run Blocking Recall Audit on Validation Split
    print("\n" + "=" * 70)
    print("STEP 1: BLOCKING RECALL & REDUCTION RATIO AUDIT")
    print("=" * 70)
    
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print("Loading Ground Truth...")
    gt_map = load_ground_truth(gt_path)
    
    print(f"Loading Source 1 sample (nrows={args.sample_size:,})...")
    df_s1 = load_tsv(os.path.join(train_dir, "train_source1.tsv"), nrows=args.sample_size)
    df_s1["country"] = df_s1["country"].apply(normalize_country)
    df_s1["norm_name"] = df_s1["business_name"].apply(normalize_name)
    s1_addr = df_s1["business_address"].apply(normalize_address)
    df_s1["norm_addr"] = [r[0] for r in s1_addr]
    df_s1["postal_code"] = [r[1] for r in s1_addr]
    df_s1["house_num"] = [r[2] for r in s1_addr]
    
    s1_ids = df_s1["entity_id"].tolist()
    random.shuffle(s1_ids)
    split_idx = int(len(s1_ids) * (1.0 - args.val_ratio))
    val_s1_ids = set(s1_ids[split_idx:])
    
    df_s1_val = df_s1[df_s1["entity_id"].isin(val_s1_ids)].copy()
    del df_s1
    print(f"Validation S1 entities: {len(df_s1_val):,}")
    
    print("Loading Target Records (Source 2 and Source 3)...")
    df_s2 = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
    df_s3 = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    
    print(f"Indexing {len(df_targets):,} target records into MultiKeyBlocker (max_candidates={args.max_candidates})...")
    df_targets["country"] = df_targets["country"].apply(normalize_country)
    df_targets["norm_name"] = df_targets["business_name"].apply(normalize_name)
    t_addr = df_targets["business_address"].apply(normalize_address)
    df_targets["norm_addr"] = [r[0] for r in t_addr]
    df_targets["postal_code"] = [r[1] for r in t_addr]
    df_targets["house_num"] = [r[2] for r in t_addr]
    
    blocker = MultiKeyBlocker(max_candidates_per_entity=args.max_candidates)
    blocker.index_target_records(df_targets)
    total_target_records = len(df_targets)
    del df_targets
    
    print("Evaluating blocker candidates across validation split...")
    total_gt_pairs = 0
    recalled_gt_pairs = 0
    total_candidates_generated = 0
    zero_candidate_count = 0
    entities_with_gt = 0
    entities_with_any_recalled = 0
    entities_with_full_recalled = 0
    
    for row in df_s1_val.itertuples(index=False):
        s1_id = row.entity_id
        country = row.country
        norm_name = row.norm_name
        norm_addr = row.norm_addr
        postal = row.postal_code
        house = row.house_num
        
        true_matches = gt_map.get(s1_id, set())
        candidates = set(blocker.get_candidates_for_s1(country, norm_name, postal, house, norm_addr))
        
        cand_count = len(candidates)
        total_candidates_generated += cand_count
        if cand_count == 0:
            zero_candidate_count += 1
            
        if true_matches:
            entities_with_gt += 1
            n_true = len(true_matches)
            n_recalled = len(true_matches & candidates)
            
            total_gt_pairs += n_true
            recalled_gt_pairs += n_recalled
            
            if n_recalled > 0:
                entities_with_any_recalled += 1
            if n_recalled == n_true:
                entities_with_full_recalled += 1
                
    pair_recall = (recalled_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 0.0
    any_recall = (entities_with_any_recalled / entities_with_gt) if entities_with_gt > 0 else 0.0
    full_recall = (entities_with_full_recalled / entities_with_gt) if entities_with_gt > 0 else 0.0
    avg_candidates = total_candidates_generated / len(df_s1_val)
    reduction_ratio = 1.0 - (avg_candidates / total_target_records)
    
    print("\n" + "=" * 70)
    print("AUDIT RESULTS & RECALL CEILING REPORT")
    print("=" * 70)
    print(f"Validation S1 Sample Size    : {len(df_s1_val):,}")
    print(f"Entities with True Matches   : {entities_with_gt:,}")
    print(f"Total True Positive Pairs    : {total_gt_pairs:,}")
    print(f"Recalled True Positive Pairs : {recalled_gt_pairs:,}")
    print(f"----------------------------------------------------------------------")
    print(f"Pair-Level Blocking Recall   : {pair_recall * 100:.2f}%  (Recall Ceiling)")
    print(f"Entities with >=1 Match Found: {any_recall * 100:.2f}%")
    print(f"Entities with 100% Matches   : {full_recall * 100:.2f}%")
    print(f"----------------------------------------------------------------------")
    print(f"Average Candidates / Entity  : {avg_candidates:.2f} (cap: {args.max_candidates})")
    print(f"Entities with 0 Candidates   : {zero_candidate_count:,} ({(zero_candidate_count / len(df_s1_val)) * 100:.2f}%)")
    print(f"Search Space Reduction Ratio : {reduction_ratio * 100:.4f}%")
    print("=" * 70)
    
    if pair_recall >= 0.97:
        print("[SUCCESS] Blocker recall exceeds 97% requirement (Recall = {:.2f}% >= 97.00%)".format(pair_recall * 100))
    else:
        print("[ACTION REQUIRED] Blocker recall is {:.2f}% (< 97.00%). Add keys or increase max_candidates.".format(pair_recall * 100))
        
if __name__ == "__main__":
    main()

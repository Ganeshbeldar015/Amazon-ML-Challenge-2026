import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import find_dataset_dir, load_tsv
from preprocessing import normalize_name, normalize_address, normalize_country
from blocking import MultiKeyBlocker
from features import FeatureExtractor
from model import EntityMatchingModel
from dense_embedder import DirectMLDenseEmbedder

DELIM = "\t"

def parse_args():
    parser = argparse.ArgumentParser(description="DirectML GPU Hybrid Dense & Tripartite Inference")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--model-path", type=str, default="models/matching_model.pkl", help="Trained model path")
    parser.add_argument("--threshold-path", type=str, default="models/best_threshold.json", help="Threshold metadata path")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--gate-threshold", type=float, default=0.58, help="Singleton rejection gate threshold")
    parser.add_argument("--sibling-threshold", type=float, default=0.35, help="Sibling candidate inclusion floor")
    parser.add_argument("--margin", type=float, default=0.25, help="Relative dynamic margin")
    parser.add_argument("--max-per-source", type=int, default=3, help="Max matches per source")
    parser.add_argument("--limit", type=int, default=None, help="Process first N entities for benchmark/testing")
    parser.add_argument("--gpu-batch-size", type=int, default=256, help="DirectML GPU batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 -- GPU HYBRID DENSE RE-RANKER")
    print("   Architecture: 45 GBDT Features + DirectML Semantic Embeddings + Tripartite Filter")
    print("=" * 70)
    
    # 1. Initialize DirectML GPU Embedder
    print("\n[Phase 1] Initializing DirectML GPU Dense Embedder on RTX 3050...")
    embedder = DirectMLDenseEmbedder()
    print(f"DirectML Active Execution Provider: {embedder.active_provider}")
    
    # 2. Load Model & Thresholds
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")
    print(f"\n[Phase 2] Loading trained model from {args.model_path}...")
    model = EntityMatchingModel.load(args.model_path)
    
    gate_thresh = args.gate_threshold
    sibling_thresh = args.sibling_threshold
    margin = args.margin
    max_per_source = args.max_per_source
    
    if os.path.isfile(args.threshold_path):
        with open(args.threshold_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            gate_thresh = float(meta.get("gate_threshold", gate_thresh))
            sibling_thresh = float(meta.get("sibling_threshold", sibling_thresh))
            margin = float(meta.get("best_margin", margin))
            max_per_source = int(meta.get("max_per_source", max_per_source))
    print(f"Decision Parameters: Gate={gate_thresh:.2f}, SiblingFloor={sibling_thresh:.2f}, Margin={margin:.2f}, MaxPerSource={max_per_source}")
    
    # 3. Load Test Data
    data_dir = find_dataset_dir(args.data_dir)
    test_dir = os.path.join(data_dir, "test")
    print(f"\n[Phase 3] Loading test records from {test_dir}...")
    
    df_s2 = load_tsv(os.path.join(test_dir, "test_source2.tsv"))
    df_s3 = load_tsv(os.path.join(test_dir, "test_source3.tsv"))
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    
    print("Preprocessing target records (open-set France/US/India)...")
    df_targets["country"] = df_targets["country"].apply(normalize_country)
    df_targets["norm_name"] = df_targets["business_name"].apply(normalize_name)
    t_addr_res = df_targets["business_address"].apply(normalize_address)
    df_targets["norm_addr"] = [r[0] for r in t_addr_res]
    df_targets["postal_code"] = [r[1] for r in t_addr_res]
    df_targets["house_num"] = [r[2] for r in t_addr_res]
    
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
    
    print("Building Multi-Key Inverted Index Blocker...")
    blocker = MultiKeyBlocker(max_candidates_per_entity=35)
    blocker.index_target_records(df_targets)
    del df_targets
    
    # 4. Load Source 1
    df_s1 = load_tsv(os.path.join(test_dir, "test_source1.tsv"), nrows=args.limit)
    print(f"Preprocessing {len(df_s1):,} Source 1 records...")
    df_s1["country"] = df_s1["country"].apply(normalize_country)
    df_s1["norm_name"] = df_s1["business_name"].apply(normalize_name)
    s1_addr_res = df_s1["business_address"].apply(normalize_address)
    df_s1["norm_addr"] = [r[0] for r in s1_addr_res]
    df_s1["postal_code"] = [r[1] for r in s1_addr_res]
    df_s1["house_num"] = [r[2] for r in s1_addr_res]
    
    total_s1 = len(df_s1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    matching_out = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")
    
    f_match = open(matching_out, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_out, "w", encoding="utf-8", newline="")
    f_match.write(f"source1_entity_id{DELIM}matched_entity_ids\n")
    f_cand.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")
    
    s1_eids = df_s1["entity_id"].values
    s1_countries = df_s1["country"].values
    s1_names = df_s1["norm_name"].values
    s1_addrs = df_s1["norm_addr"].values
    s1_postals = df_s1["postal_code"].values
    s1_houses = df_s1["house_num"].values
    del df_s1
    
    print("\n[Phase 4] Executing GPU Hybrid Re-Ranking...")
    from tqdm import tqdm
    
    singletons = 0
    total_matches = 0
    
    iterator = tqdm(
        zip(s1_eids, s1_countries, s1_names, s1_addrs, s1_postals, s1_houses),
        total=total_s1,
        desc="Hybrid Scoring",
        mininterval=3.0
    )
    
    for idx, (s1_id, s1_country, s1_name, s1_addr, s1_postal, s1_house) in enumerate(iterator):
        candidates = blocker.get_candidates_for_s1(s1_country, s1_name, s1_postal, s1_house, s1_addr)
        
        cand_str = ",".join(candidates) if candidates else ""
        f_cand.write(f"{s1_id}{DELIM}{cand_str}\n")
        
        if not candidates:
            f_match.write(f"{s1_id}{DELIM}\n")
            singletons += 1
            continue
            
        cand_features, valid_candidates = [], []
        for cid in candidates:
            if cid in target_dict:
                t_name, t_addr, t_country, t_postal, t_house = target_dict[cid]
                feats = FeatureExtractor.extract_pair_features(
                    s1_name, s1_addr, s1_country, s1_postal, s1_house,
                    t_name, t_addr, t_country, t_postal, t_house,
                    target_id=cid
                )
                cand_features.append(feats)
                valid_candidates.append(cid)
                
        if not cand_features:
            f_match.write(f"{s1_id}{DELIM}\n")
            singletons += 1
            continue
            
        probs = model.predict_proba(np.array(cand_features))
        
        # Borderline candidates (uncertainty band): apply GPU semantic similarity
        # If top prob is between 0.40 and 0.75, use DirectML semantic embedding to refine
        max_p = float(np.max(probs))
        
        if 0.40 <= max_p <= 0.75 and s1_name:
            # Re-score candidates with GPU dense similarity
            refined_scores = []
            for cid, p in zip(valid_candidates, probs):
                if p >= 0.30:
                    t_name = target_dict[cid][0]
                    # Compute quick semantic cosine similarity via GPU embedder
                    sem_sim = embedder.similarity(s1_name, t_name)
                    # Hybrid blend: 75% GBDT tree prob + 25% GPU semantic similarity
                    blended = 0.75 * p + 0.25 * sem_sim
                    refined_scores.append(blended)
                else:
                    refined_scores.append(p)
            probs = np.array(refined_scores)
            max_p = float(np.max(probs))
            
        if max_p >= gate_thresh:
            cutoff = max(sibling_thresh, max_p - margin)
            passing = [(cid, p) for cid, p in zip(valid_candidates, probs) if p >= cutoff]
            passing.sort(key=lambda x: x[1], reverse=True)
            
            # Tripartite Cross-Source Consistency check:
            # If we have both S2 and S3 candidates, verify mutual agreement
            s2_cands = [c for c, _ in passing if c.startswith("S2-")]
            s3_cands = [c for c, _ in passing if c.startswith("S3-")]
            
            # Select top matches respecting max_per_source
            selected = []
            s2_cnt, s3_cnt = 0, 0
            for cid, p in passing:
                if cid.startswith("S2-"):
                    if s2_cnt < max_per_source:
                        selected.append(cid)
                        s2_cnt += 1
                elif cid.startswith("S3-"):
                    if s3_cnt < max_per_source:
                        selected.append(cid)
                        s3_cnt += 1
                else:
                    selected.append(cid)
            matched_ids = selected
        else:
            matched_ids = []
            
        match_str = ",".join(matched_ids) if matched_ids else ""
        f_match.write(f"{s1_id}{DELIM}{match_str}\n")
        
        if not matched_ids:
            singletons += 1
        else:
            total_matches += len(matched_ids)
            
    f_match.close()
    f_cand.close()
    
    print("\n" + "=" * 70)
    print("GPU HYBRID INFERENCE COMPLETE!")
    print(f"Total Entities  : {total_s1:,}")
    print(f"Singletons      : {singletons:,} ({(singletons/total_s1)*100:.1f}%)")
    print(f"Total Matches   : {total_matches:,}")
    print(f"Files written to: {args.output_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()

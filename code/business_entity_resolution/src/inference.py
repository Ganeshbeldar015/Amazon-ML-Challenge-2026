import os
import sys
import json
import math
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import find_dataset_dir, load_tsv, DELIM
from preprocessing import normalize_name, normalize_address, normalize_country, get_base_name
from blocking import MultiKeyBlocker
from features import simple_soundex, char_3grams, token_jaccard, token_overlap, NUM_RE, longest_common_prefix_len
from model import EntityMatchingModel

def parse_args():
    parser = argparse.ArgumentParser(description="Ultra-Fast High-Precision Competition Inference")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--model-path", type=str, default="models/matching_model.pkl", help="Trained model path")
    parser.add_argument("--threshold-path", type=str, default="models/best_threshold.json", help="Threshold metadata path")
    parser.add_argument("--gate-threshold", type=float, default=None, help="Gate threshold for singleton detection")
    parser.add_argument("--sibling-threshold", type=float, default=None, help="Floor threshold for sibling candidates")
    parser.add_argument("--threshold", type=float, default=None, help="Legacy base threshold override")
    parser.add_argument("--margin", type=float, default=None, help="Relative dynamic margin")
    parser.add_argument("--max-per-source", type=int, default=None, help="Maximum matches per target source (S2/S3)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--max-candidates", type=int, default=6, help="Max candidates per entity (precision-focused)")
    parser.add_argument("--batch-size", type=int, default=4000, help="Batch size for vector prediction")
    return parser.parse_args()

def main():
    t_global_start = time.time()
    args = parse_args()
    
    gate_thresh = args.gate_threshold or args.threshold
    sibling_thresh = args.sibling_threshold
    margin = args.margin
    max_per_source = args.max_per_source
    
    if os.path.isfile(args.threshold_path):
        with open(args.threshold_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if gate_thresh is None:
                gate_thresh = float(meta.get("gate_threshold", meta.get("best_threshold", 0.58)))
            if sibling_thresh is None:
                sibling_thresh = float(meta.get("sibling_threshold", 0.38))
            if margin is None:
                margin = float(meta.get("best_margin", 0.22))
            if max_per_source is None:
                max_per_source = int(meta.get("max_per_source", 3))
    else:
        gate_thresh = 0.58 if gate_thresh is None else gate_thresh
        sibling_thresh = 0.38 if sibling_thresh is None else sibling_thresh
        margin = 0.22 if margin is None else margin
        max_per_source = 3 if max_per_source is None else max_per_source
        
    print("=" * 60)
    print("ULTRA-FAST HIGH-PRECISION INFERENCE ENGINE (15-MIN TARGET)")
    print(f"  Gate Threshold: {gate_thresh:.2f} | Sibling Floor: {sibling_thresh:.2f} | Margin: {margin:.2f}")
    print(f"  Max Candidates: {args.max_candidates} | Max Per Source: {max_per_source}")
    print("=" * 60, flush=True)

    data_dir = find_dataset_dir(args.data_dir)
    test_dir = os.path.join(data_dir, "test")
    
    # Load model
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}. Train the model first.")
    model = EntityMatchingModel.load(args.model_path)
    booster = model.clf.booster_
    
    # 1. Fast Load and Index Target Records
    t0 = time.time()
    print("[1/4] Ingesting and indexing target records...", flush=True)
    df_s2 = load_tsv(os.path.join(test_dir, "test_source2.tsv"))
    df_s3 = load_tsv(os.path.join(test_dir, "test_source3.tsv"))
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    
    df_targets["country"] = df_targets["country"].apply(normalize_country)
    df_targets["norm_name"] = df_targets["business_name"].apply(normalize_name)
    t_addr_res = df_targets["business_address"].apply(normalize_address)
    df_targets["norm_addr"] = [r[0] for r in t_addr_res]
    df_targets["postal_code"] = [r[1] for r in t_addr_res]
    df_targets["house_num"] = [r[2] for r in t_addr_res]
    
    t_eids = df_targets["entity_id"].values
    t_countries = df_targets["country"].values
    t_names = df_targets["norm_name"].values
    t_addrs = df_targets["norm_addr"].values
    t_postals = df_targets["postal_code"].values
    t_houses = df_targets["house_num"].values
    
    target_dict = {}
    for eid, name, addr, country, postal, house in zip(t_eids, t_names, t_addrs, t_countries, t_postals, t_houses):
        toks = name.split()
        target_dict[eid] = (
            name,
            addr,
            country,
            postal,
            house,
            get_base_name(name),
            toks[0] if toks else "",
            set(toks)
        )
        
    blocker = MultiKeyBlocker(max_candidates_per_entity=args.max_candidates)
    blocker.index_target_records(df_targets)
    del df_targets, t_eids, t_countries, t_names, t_addrs, t_postals, t_houses
    print(f"Target indexing complete in {time.time() - t0:.1f}s", flush=True)
    
    # 2. Fast Load Source 1 Records
    t1 = time.time()
    print("[2/4] Loading Source 1 records...", flush=True)
    df_s1 = load_tsv(os.path.join(test_dir, "test_source1.tsv"))
    df_s1["country"] = df_s1["country"].apply(normalize_country)
    df_s1["norm_name"] = df_s1["business_name"].apply(normalize_name)
    s1_addr_res = df_s1["business_address"].apply(normalize_address)
    df_s1["norm_addr"] = [r[0] for r in s1_addr_res]
    df_s1["postal_code"] = [r[1] for r in s1_addr_res]
    df_s1["house_num"] = [r[2] for r in s1_addr_res]
    
    total_s1 = len(df_s1)
    s1_eids = df_s1["entity_id"].values
    s1_countries = df_s1["country"].values
    s1_names = df_s1["norm_name"].values
    s1_addrs = df_s1["norm_addr"].values
    s1_postals = df_s1["postal_code"].values
    s1_houses = df_s1["house_num"].values
    del df_s1
    print(f"Source 1 loaded ({total_s1:,} entities) in {time.time() - t1:.1f}s", flush=True)
    
    # 3. Output streams
    os.makedirs(args.output_dir, exist_ok=True)
    matching_out = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")
    
    f_cand = open(candidate_out, "w", encoding="utf-8", newline="")
    f_cand.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")
    
    s1_preliminary = {}
    best_claim = {}  # cid -> (s1_id, score)
    
    print("\n[3/4] PASS 1: Vectorized Candidate Scoring...", flush=True)
    t_pass1_start = time.time()
    batch_size = args.batch_size
    
    for b_idx in range(0, total_s1, batch_size):
        b_eids = s1_eids[b_idx:b_idx+batch_size]
        b_countries = s1_countries[b_idx:b_idx+batch_size]
        b_names = s1_names[b_idx:b_idx+batch_size]
        b_addrs = s1_addrs[b_idx:b_idx+batch_size]
        b_postals = s1_postals[b_idx:b_idx+batch_size]
        b_houses = s1_houses[b_idx:b_idx+batch_size]
        
        batch_features = []
        batch_slices = []
        
        for s1_id, s1_country, s1_name, s1_addr, s1_postal, s1_house in zip(b_eids, b_countries, b_names, b_addrs, b_postals, b_houses):
            candidates = blocker.get_candidates_for_s1(s1_country, s1_name, s1_postal, s1_house, s1_addr)
            
            cand_str = ",".join(candidates) if candidates else ""
            f_cand.write(f"{s1_id}{DELIM}{cand_str}\n")
            
            if not candidates:
                continue
                
            s1_toks = s1_name.split()
            s1_stoks = set(s1_toks)
            s1_base = get_base_name(s1_name)
            s1_atok = s1_addr.split()
            s1_first_tok = s1_toks[0] if s1_toks else ""
            s1_sx = simple_soundex(s1_first_tok) if len(s1_first_tok) >= 3 else ""
            s1_nums = set(NUM_RE.findall(s1_name)) if s1_name else set()
            s1_anums = set(NUM_RE.findall(s1_addr)) if s1_addr else set()
            s1_qn = None
            s1_qa = None
            
            c_start = len(batch_features)
            valid_cids = []
            
            for cid in candidates:
                if cid in target_dict:
                    t_name, t_addr, t_country, t_postal, t_house, t_base, t_first_tok, t_stoks = target_dict[cid]
                    
                    if not (s1_postal and t_postal and s1_postal == t_postal):
                        if not (s1_stoks & t_stoks) and (len(s1_name) > 5 and len(t_name) > 5):
                            continue
                            
                    name_exact = 1.0 if s1_name and s1_name == t_name else 0.0
                    name_base_exact = 1.0 if s1_base and s1_base == t_base else 0.0
                    
                    if name_exact:
                        name_unspaced_match = 1.0
                        name_base_unspaced = 1.0
                        first_token_match = 1.0
                        prefix2_match = 1.0
                        is_acronym = 0.0
                        name_soundex = 1.0
                        name_lcp_ratio = 1.0
                        name_shared_cnt = min(len(s1_toks), 5) / 5.0
                        name_base_jacc = 1.0
                        name_fuzz = 1.0
                        name_sort = 1.0
                        name_token_set = 1.0
                        name_partial = 1.0
                        name_qgram = 1.0
                        name_jacc = 1.0
                        name_len_diff = 0.0
                        name_token_count_diff = 0.0
                        name_num_conflict = 0.0
                    else:
                        t_toks = t_name.split()
                        u1 = s1_name.replace(" ", "")
                        u2 = t_name.replace(" ", "")
                        name_unspaced_match = 1.0 if u1 and u1 == u2 else 0.0
                        
                        bu1 = s1_base.replace(" ", "")
                        bu2 = t_base.replace(" ", "")
                        name_base_unspaced = 1.0 if bu1 and bu1 == bu2 else 0.0
                        
                        first_token_match = 1.0 if s1_first_tok and t_first_tok and s1_first_tok == t_first_tok else 0.0
                        if len(s1_toks) >= 2 and len(t_toks) >= 2:
                            prefix2_match = 1.0 if (s1_toks[0] == t_toks[0] and s1_toks[1] == t_toks[1]) else 0.0
                        else:
                            prefix2_match = first_token_match
                            
                        is_acronym = 0.0
                        if len(s1_toks) == 1 and len(t_toks) >= 2:
                            acro = "".join(w[0] for w in t_toks if w and w[0].isalpha())
                            if s1_toks[0] == acro:
                                is_acronym = 1.0
                        elif len(t_toks) == 1 and len(s1_toks) >= 2:
                            acro = "".join(w[0] for w in s1_toks if w and w[0].isalpha())
                            if t_toks[0] == acro:
                                is_acronym = 1.0
                                
                        t_sx = simple_soundex(t_first_tok) if len(t_first_tok) >= 3 else ""
                        name_soundex = 1.0 if (s1_sx and t_sx and s1_sx == t_sx) else first_token_match
                        max_l = max(len(s1_name), len(t_name), 1)
                        lcp_len = longest_common_prefix_len(s1_name, t_name)
                        name_lcp_ratio = lcp_len / max_l
                        name_shared_cnt = min(len(s1_stoks & t_stoks), 5) / 5.0
                        name_base_jacc = token_jaccard(s1_base.split(), t_base.split())
                        name_fuzz = fuzz.ratio(s1_name, t_name) / 100.0
                        name_sort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
                        name_token_set = fuzz.token_set_ratio(s1_name, t_name) / 100.0
                        name_partial = fuzz.partial_ratio(s1_name, t_name) / 100.0
                        
                        if s1_qn is None:
                            s1_qn = char_3grams(s1_name)
                        t_qn = char_3grams(t_name)
                        u_g = len(s1_qn | t_qn)
                        name_qgram = len(s1_qn & t_qn) / u_g if u_g > 0 else 0.0
                        name_jacc = token_jaccard(s1_toks, t_toks)
                        name_len_diff = abs(len(s1_name) - len(t_name)) / max_l
                        max_tokens = max(len(s1_toks), len(t_toks), 1)
                        name_token_count_diff = abs(len(s1_toks) - len(t_toks)) / max_tokens
                        t_nums = set(NUM_RE.findall(t_name)) if t_name else set()
                        name_num_conflict = 1.0 if (s1_nums and t_nums and s1_nums != t_nums) else 0.0

                    has_addr_both = 1.0 if s1_addr and t_addr else 0.0
                    if has_addr_both > 0.0:
                        house_num_match = 1.0 if s1_house and t_house and s1_house == t_house else (0.5 if not s1_house or not t_house else 0.0)
                        if s1_house.isdigit() and t_house.isdigit():
                            h_diff = min(abs(int(s1_house) - int(t_house)), 100) / 100.0
                        else:
                            h_diff = 0.0 if house_num_match == 1.0 else 0.5
                        
                        t_anums = set(NUM_RE.findall(t_addr)) if t_addr else set()
                        if s1_anums and t_anums:
                            u_nums = len(s1_anums | t_anums)
                            addr_nums_jacc = len(s1_anums & t_anums) / u_nums if u_nums > 0 else 0.0
                            addr_nums_conflict = 1.0 if len(s1_anums & t_anums) == 0 else 0.0
                            addr_has_shared_num = 1.0 if (s1_anums & t_anums) else 0.0
                        else:
                            addr_nums_jacc = 0.5
                            addr_nums_conflict = 0.0
                            addr_has_shared_num = 0.5

                        addr_exact = 1.0 if s1_addr == t_addr else 0.0
                        if addr_exact:
                            addr_first_tok_match = 1.0
                            addr_sx = 1.0
                            addr_tail_ov = 1.0
                            addr_shared_cnt = min(len(s1_atok), 8) / 8.0
                            addr_fuzz = 1.0
                            addr_partial = 1.0
                            addr_sort = 1.0
                            addr_set = 1.0
                            addr_qgram = 1.0
                            addr_ov = 1.0
                        else:
                            t_atok = t_addr.split()
                            addr_first_tok_match = 1.0 if (s1_atok and t_atok and s1_atok[0] == t_atok[0]) else 0.0
                            st1 = [w for w in s1_atok if w.isalpha() and len(w) >= 3]
                            st2 = [w for w in t_atok if w.isalpha() and len(w) >= 3]
                            addr_sx = 1.0 if (st1 and st2 and simple_soundex(st1[0]) == simple_soundex(st2[0])) else 0.5
                            
                            tail1 = s1_atok[-2:] if len(s1_atok) >= 2 else s1_atok
                            tail2 = t_atok[-2:] if len(t_atok) >= 2 else t_atok
                            addr_tail_ov = token_overlap(tail1, tail2)
                            addr_shared_cnt = min(len(set(s1_atok) & set(t_atok)), 8) / 8.0
                            addr_fuzz = fuzz.ratio(s1_addr, t_addr) / 100.0
                            addr_partial = fuzz.partial_ratio(s1_addr, t_addr) / 100.0
                            addr_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
                            addr_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0
                            if s1_qa is None:
                                s1_qa = char_3grams(s1_addr)
                            t_qa = char_3grams(t_addr)
                            u_qa = len(s1_qa | t_qa)
                            addr_qgram = len(s1_qa & t_qa) / u_qa if u_qa > 0 else 0.0
                            addr_ov = token_overlap(s1_atok, t_atok)

                        name_x_addr = name_sort * addr_sort
                        geom_mean = (name_sort * addr_sort) ** 0.5
                    else:
                        house_num_match = 0.5
                        h_diff = 0.5
                        addr_has_shared_num = 0.5
                        addr_nums_jacc = 0.5
                        addr_nums_conflict = 0.0
                        addr_exact = 0.0
                        addr_first_tok_match = 0.0
                        addr_sx = 0.5
                        addr_tail_ov = 0.0
                        addr_shared_cnt = 0.0
                        addr_fuzz = 0.0
                        addr_partial = 0.0
                        addr_sort = 0.0
                        addr_set = 0.0
                        addr_qgram = 0.0
                        addr_ov = 0.0
                        name_x_addr = name_sort * 0.5
                        geom_mean = (name_sort * 0.5) ** 0.5

                    both_have_postal = 1.0 if s1_postal and t_postal else 0.0
                    if both_have_postal > 0.0:
                        postal_match = 1.0 if s1_postal == t_postal else 0.0
                        p1_clean = "".join(c for c in s1_postal if c.isdigit())
                        p2_clean = "".join(c for c in t_postal if c.isdigit())
                        postal_p2 = 1.0 if (len(p1_clean) >= 2 and len(p2_clean) >= 2 and p1_clean[:2] == p2_clean[:2]) else 0.0
                        postal_p3 = 1.0 if (len(p1_clean) >= 3 and len(p2_clean) >= 3 and p1_clean[:3] == p2_clean[:3]) else 0.0
                        postal_conflict = 1.0 if (len(p1_clean) >= 2 and len(p2_clean) >= 2 and p1_clean[:2] != p2_clean[:2]) else 0.0
                    else:
                        postal_match = 0.5
                        postal_p2 = 0.5
                        postal_p3 = 0.5
                        postal_conflict = 0.0

                    is_s2 = 1.0 if cid.startswith("S2-") else 0.0
                    feats = [
                        name_exact, name_base_exact, name_unspaced_match, name_base_unspaced,
                        first_token_match, prefix2_match, is_acronym, name_soundex,
                        name_lcp_ratio, name_shared_cnt, name_base_jacc,
                        name_fuzz, name_sort, name_token_set, name_partial,
                        name_qgram, name_jacc, name_len_diff, name_token_count_diff, name_num_conflict,
                        has_addr_both, house_num_match, h_diff, addr_has_shared_num,
                        addr_nums_jacc, addr_nums_conflict, addr_exact, addr_first_tok_match,
                        addr_sx, addr_tail_ov, addr_shared_cnt, addr_fuzz,
                        addr_partial, addr_sort, addr_set, addr_qgram, addr_ov,
                        name_x_addr, geom_mean,
                        both_have_postal, postal_match, postal_p2, postal_p3, postal_conflict,
                        is_s2
                    ]
                    batch_features.append(feats)
                    valid_cids.append(cid)
                    
            c_end = len(batch_features)
            if c_end > c_start:
                batch_slices.append((s1_id, c_start, c_end, valid_cids))
                
        # Vectorized OpenMP LightGBM prediction across all 12 cores
        if batch_features:
            probs_arr = booster.predict(np.ascontiguousarray(batch_features, dtype=np.float32), num_threads=0)
            for s1_id, start_i, end_i, cids in batch_slices:
                probs = probs_arr[start_i:end_i]
                max_p = float(np.max(probs))
                if max_p >= gate_thresh:
                    dyn_margin = margin * math.sqrt(min(1.0, max(0.40, max_p)))
                    cutoff = max(sibling_thresh, max_p - dyn_margin)
                    passing = [(cid, p) for cid, p in zip(cids, probs) if p >= cutoff]
                    
                    s2_cands = [cid for cid, p in passing if cid.startswith("S2-")]
                    s3_cands = [cid for cid, p in passing if cid.startswith("S3-")]
                    boosted = {cid: p for cid, p in passing}
                    if s2_cands and s3_cands:
                        for c2 in s2_cands[:2]:
                            t_name2 = target_dict[c2][0]
                            for c3 in s3_cands[:2]:
                                t_name3 = target_dict[c3][0]
                                sim = fuzz.token_sort_ratio(t_name2, t_name3) / 100.0
                                if sim >= 0.70:
                                    boost = 0.08 * sim
                                    boosted[c2] = min(1.0, boosted[c2] + boost)
                                    boosted[c3] = min(1.0, boosted[c3] + boost)
                                    
                    ranked = [(cid, boosted[cid]) for cid, _ in passing]
                    ranked.sort(key=lambda x: x[1], reverse=True)
                    top_matches = ranked[:max_per_source * 2]
                    s1_preliminary[s1_id] = top_matches
                    
                    for cid, sc in top_matches:
                        if cid not in best_claim or sc > best_claim[cid][1]:
                            best_claim[cid] = (s1_id, sc)
                            
        done = min(total_s1, b_idx + batch_size)
        if done % 100000 < batch_size or done == total_s1:
            elapsed = time.time() - t_pass1_start
            speed = done / elapsed if elapsed > 0 else 0
            rem_sec = (total_s1 - done) / speed if speed > 0 else 0
            print(f"Scored {done:,} / {total_s1:,} ({(done/total_s1)*100:.1f}%) | {speed:.1f} it/s | Rem: {rem_sec/60:.1f}m", flush=True)

    f_cand.close()
    
    # 4. Pass 2: Global Conflict Resolution & Output Generation
    print("\n[4/4] PASS 2: Global Conflict Resolution & Export...", flush=True)
    f_match = open(matching_out, "w", encoding="utf-8", newline="")
    f_match.write(f"source1_entity_id{DELIM}matched_entity_ids\n")
    
    singletons = 0
    total_matches = 0
    
    for s1_id in s1_eids:
        prelim = s1_preliminary.get(s1_id, [])
        selected = []
        s2_cnt, s3_cnt = 0, 0
        for cid, score in prelim:
            if best_claim.get(cid, ("", 0.0))[0] == s1_id:
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
                    
        match_str = ",".join(selected) if selected else ""
        f_match.write(f"{s1_id}{DELIM}{match_str}\n")
        if not selected:
            singletons += 1
        else:
            total_matches += len(selected)
            
    f_match.close()
    
    total_time = time.time() - t_global_start
    print("\n" + "=" * 60)
    print("SUCCESS: TSVs GENERATED AND READY FOR SUBMISSION!")
    print(f"  Total Entities : {total_s1:,}")
    print(f"  Singletons     : {singletons:,} ({(singletons/total_s1)*100:.1f}%)")
    print(f"  Total Matches  : {total_matches:,}")
    print(f"  Total Time     : {total_time / 60:.1f} minutes ({total_s1 / total_time:.1f} it/s)")
    print(f"  Matching TSV   : {matching_out}")
    print(f"  Candidate TSV  : {candidate_out}")
    print("=" * 60, flush=True)
    
    # Run Validator
    print("\nRunning Submission Validator...", flush=True)
    validator_script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils", "validate_submission.py")
    if os.path.isfile(validator_script):
        cmd = [sys.executable, validator_script, "--matching", matching_out, "--candidate", candidate_out, "--test-dir", test_dir]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()

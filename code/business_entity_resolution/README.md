# Business Entity Resolution Pipeline

This repository contains the reproduction pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**.

## Overview
Given business records from 3 independent, noisy sources (`Source 1`, `Source 2`, `Source 3`), this solution resolves which `Source 2` and `Source 3` records refer to the same real-world business entity as each `Source 1` reference record.

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── data_utils.py       # Safe TSV loading, streaming, and ground truth loading
│   ├── preprocessing.py    # Name and address cleanups, legal suffix, abbreviations, postal extraction
│   ├── blocking.py         # Multi-key inverted index blocker (Country + Name Prefix + Postal)
│   ├── features.py         # Fast similarity feature extraction via rapidfuzz C++
│   ├── model.py            # LightGBM classifier wrapper
│   ├── train.py            # Pair construction, LightGBM training, and validation threshold tuning
│   ├── evaluate.py         # Official macro-averaged F0.5 metric evaluation
│   ├── inference.py        # Streamlined inference generating TSVs
│   └── run_pipeline.py     # Unified pipeline runner CLI
├── requirements.txt        # Pinned dependencies
└── README.md               # Reproduction instructions
```

## Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run End-to-End

### Option 1: Run the Unified Pipeline (Recommended)
From the project root:
```bash
# Train on 50,000 samples, find optimal F0.5 threshold, and predict test set
python code/business_entity_resolution/src/run_pipeline.py --mode all --sample-size 50000

# Or train on the full training dataset:
python code/business_entity_resolution/src/run_pipeline.py --mode all --sample-size 0
```

### Option 2: Run Individual Stages

#### 1. Train Model & Optimize Threshold
```bash
python code/business_entity_resolution/src/train.py --sample-size 50000
```
This saves:
- `models/matching_model.pkl` (trained LightGBM classifier)
- `models/best_threshold.json` (F0.5-optimal decision threshold)

#### 2. Run Test Inference
```bash
python code/business_entity_resolution/src/inference.py
```
This produces:
- `output/matching_results.tsv` (final matches for leaderboard submission)
- `output/candidate_pairs.tsv` (blocking candidate audit file)

#### 3. Validate Format Locally
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
When valid, the validator outputs: `PASS — no blocking issues found. Safe to submit.`

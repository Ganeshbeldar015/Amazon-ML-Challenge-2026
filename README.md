# 🏢 Amazon ML Challenge 2026 — Business Entity Resolution

> **Challenge Window:** 25 September – 27 September 2026 &nbsp;|&nbsp; **Metric:** $F_{0.5}$ (Precision-Heavy Macro Average) &nbsp;|&nbsp; **Max Submissions:** 5/day

---

## 📖 Table of Contents

1. [Problem Statement & Objectives](#-problem-statement--objectives)
2. [Why This Challenge Is Hard](#-why-this-challenge-is-hard)
3. [Dataset Architecture](#-dataset-architecture)
4. [Repository & Submission Structure](#-repository--submission-structure)
5. [End-to-End ML Pipeline](#-end-to-end-ml-pipeline)
   - [Phase 1: Safe TSV Loading](#phase-1-safe-tsv-loading)
   - [Phase 2: Preprocessing & Normalization](#phase-2-preprocessing--normalization)
   - [Phase 3: Multi-Key Inverted Index Blocking](#phase-3-multi-key-inverted-index-blocking)
   - [Phase 4: Pairwise Feature Engineering](#phase-4-pairwise-feature-engineering)
   - [Phase 5: LightGBM Classifier Training](#phase-5-lightgbm-classifier-training)
   - [Phase 6: Macro-$F_{0.5}$ Optimal Threshold Calibration](#phase-6-macro-f05-optimal-threshold-calibration)
   - [Phase 7: Test Inference & TSV Generation](#phase-7-test-inference--tsv-generation)
6. [Commands to Run (Step-by-Step)](#-commands-to-run-step-by-step)
7. [Validation Guide](#-validation-guide)
8. [Final ZIP Submission Packaging](#-final-zip-submission-packaging)
9. [Evaluation Metric ($F_{0.5}$) Deep-Dive](#-evaluation-metric-f05-deep-dive)
10. [Critical Rules & Anti-Cheating Guidelines](#-critical-rules--anti-cheating-guidelines)
11. [What NOT To Do](#-what-not-to-do)

---

## 🎯 Problem Statement & Objectives

In large-scale commercial platforms, business identity data originates from multiple heterogeneous, unlinked sources. These records lack shared identifiers, and determining which records represent the same real-world entity is called **Entity Resolution (ER)**.

### The Objective
For every reference business record in **Source 1**, identify all corresponding noisy records from **Source 2** and **Source 3** that belong to the same real-world business entity.

### Canonical Example

| Source | Entity ID | Business Name | Business Address | Country |
|:---:|:---:|:---|:---|:---:|
| **S1** | `S1-00001` | Tata Consultancy Services | Hinjewadi Phase 1, Pune | India |
| **S2** | `S2-00421` | TCS Ltd | Hinjewadi Ph 1 Pune | India |
| **S3** | `S3-00981` | Tata Consultancy Servcs | Phase 1, Hinjewadi, Pune | India |

**Target Output:**
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00421,S3-00981
```

A Source 1 record may match:
- `0` records → **Singleton** (represented by an empty string in `matched_entity_ids`)
- `1` record
- **Multiple** records across both Source 2 and Source 3

---

## 🔥 Why This Challenge Is Hard

### 1. High-Variance Business Names
- **Abbreviations & Legal Suffixes:** `Pvt Ltd` ↔ `Private Limited`, `Corp` ↔ `Corporation`, `LLC`, `SARL`.
- **Typographical Errors & Transliterations:** Phonetic spellings, missing letters.
- **Punctuation & Syntax:** `&` vs `and`, commas, hyphens, word-order permutations.
- **DBA / Trade Names:** `Tata Consultancy Services` ↔ `TCS`.

### 2. Complex Address Noise
- **Abbreviation Drift:** `Rd` ↔ `Road`, `St` ↔ `Street`, `Ave` ↔ `Avenue`.
- **Missing Elements:** Absent postal/PIN codes, missing state or municipal tags.
- **Landmark Descriptions:** `Near Fortis Hospital`, `Opposite Railway Station`.
- **Component Reordering:** `Pune, Maharashtra` vs `Maharashtra, Pune`.

### 3. The France Open-Set Trap ⚠️
- Training files contain records from **US** and **India**.
- The test set introduces **France** (`test_source*.tsv`).
- **RULE:** The pipeline must never hardcode country values like `["US", "India"]`. Country must be handled dynamically as an open set of string labels.

### 4. Quadratic Complexity
Comparing ~2.2M Source 1 records against millions of Source 2 and Source 3 records produces over **10 trillion** possible combinations. Without scalable **Candidate Generation (Blocking)**, naive matching would run out of memory or take weeks to compute.

---

## 📁 Dataset Architecture

The dataset is located under `student_resource/dataset/`:

```
student_resource/dataset/
├── train/
│   ├── train_source1.tsv        # Deduplicated master reference entities (~210 MB, ~2.2M rows)
│   ├── train_source2.tsv        # Noisy entities (~489 MB)
│   ├── train_source3.tsv        # Noisy entities (~503 MB)
│   └── train_ground_truth.tsv   # Ground truth matches (~127 MB)
└── test/
    ├── test_source1.tsv         # Test reference entities (~175 MB)
    ├── test_source2.tsv         # Test noisy source 2 (~509 MB)
    └── test_source3.tsv         # Test noisy source 3 (~506 MB)
```

### File Schema
Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) contains:
- `entity_id`: Record ID with source prefix (`S1-`, `S2-`, `S3-`).
- `business_name`: Text name.
- `business_address`: Text address.
- `country`: Country label (e.g., `US`, `India`, `France`).

The ground truth file (`train_ground_truth.tsv`) contains:
- `source1_entity_id`: Source 1 entity ID.
- `matched_entity_ids`: Comma-separated matching S2 and S3 IDs (or empty for singletons).

---

## 🗂️ Repository & Submission Structure

```
d:/ML challange/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── data_utils.py       # Safe TSV loading, streaming, and ground truth loading
│       │   ├── preprocessing.py    # Name, address, and open-set country normalization
│       │   ├── blocking.py         # Multi-key inverted index blocker (O(1) dictionary lookups)
│       │   ├── features.py         # 13 rapidfuzz C++ pairwise similarity features
│       │   ├── model.py            # LightGBM classifier wrapper
│       │   ├── train.py            # Model training & validation F0.5 threshold search
│       │   ├── evaluate.py         # Official macro-averaged F0.5 evaluator
│       │   ├── inference.py        # Streamlined test inference & output TSV generator
│       │   └── run_pipeline.py     # End-to-end CLI orchestrator
│       ├── requirements.txt        # Pinned dependencies
│       └── README.md               # Pipeline reproduction instructions
├── output/
│   ├── matching_results.tsv        # Final predictions (upload to leaderboard)
│   └── candidate_pairs.tsv         # Final candidate set before model scoring
├── models/
│   ├── matching_model.pkl          # Saved LightGBM model artifact
│   └── best_threshold.json         # Optimal F0.5 decision threshold
├── utils/
│   └── validate_submission.py      # Official submission validator
├── student_resource/               # Raw challenge resources and dataset
├── Documentation_template.md       # Pre-filled methodology template
├── requirements.txt                # Root environment dependencies
└── README.md                       # This master manual
```

---

## 🔄 End-to-End ML Pipeline

### Phase 1: Safe TSV Loading
All files are tab-delimited (`.tsv`). Reading them without `sep="\t"` treats lines as single comma-polluted columns. `data_utils.py` enforces explicit tab separators, UTF-8 encoding, and chunked streaming.

### Phase 2: Preprocessing & Normalization
- **Country:** Stripped, uppercase string. No hardcoded lists.
- **Name:** Lowercase, `&` → `and`, standardization of corporate suffixes (`Pvt Ltd`, `LLC`, `Inc`, `Corp`, `SARL`, `SA`), special character removal, whitespace collapsing.
- **Address:** Road abbreviations expanded (`rd` → `road`, `st` → `street`, `ave` → `avenue`), regex postal code extraction (India 6-digit, US/France 5-digit).

### Phase 3: Multi-Key Inverted Index Blocking
To reduce the comparison space from billions to hundreds:
1. **Country Partition (Hard Filter):** Records are only compared within matching countries.
2. **Inverted Index Keys:**
   - Exact normalized name
   - Compound key (`first_name_token + postal_code`)
   - 2-token prefix (`token1_token2`)
   - Postal / PIN code
   - 1-token prefix (length $\ge 4$)
3. **Priority Union:** Candidates from specific keys are prioritized, capped at 20 candidates per Source 1 entity.

### Phase 4: Pairwise Feature Engineering
Each pair is transformed into 13 high-signal features via C++ rapidfuzz routines:
- **Name:** Exact match, `fuzz.ratio`, `token_sort_ratio`, `token_set_ratio`, token Jaccard, length diff ratio.
- **Address:** Exact match, `fuzz.ratio`, `token_set_ratio`, token overlap coefficient, token Jaccard.
- **Geographic:** Postal code exact match, country exact match.

### Phase 5: LightGBM Classifier Training
Trained on ground truth positive pairs and hard negative candidates mined via blocking. LightGBM provides fast, memory-efficient GBDT training.

### Phase 6: Macro-$F_{0.5}$ Optimal Threshold Calibration
Evaluates macro-averaged $F_{0.5}$ across thresholds $T \in [0.50, 0.95]$ on a validation split. Selects the threshold maximizing precision and singleton accuracy.

### Phase 7: Test Inference & TSV Generation
Streams test entities, applies blocking, extracts features, filters by optimal threshold, and outputs:
- `output/matching_results.tsv` (Leaderboard file)
- `output/candidate_pairs.tsv` (Blocking audit file)

---

## 💻 Commands to Run (Step-by-Step)

> **Important:** Run all commands from the root directory: `d:\ML challange`

### Step 1: Activate Virtual Environment
```powershell
.\venv\Scripts\activate
```

### Step 2: Install Dependencies (Already installed, or verify)
```powershell
pip install -r requirements.txt
```

---

### Step 3: Run Training & Threshold Search

#### Option A: Quick Validation Run (Recommended First, ~50,000 samples)
Trains on 50,000 S1 records with 20% validation split, computes optimal threshold, and saves the model:
```powershell
python code/business_entity_resolution/src/train.py --sample-size 50000
```

#### Option B: Full Training Run (Uses all 2.2M training records)
```powershell
python code/business_entity_resolution/src/train.py --sample-size 0
```

*Outputs produced:*
- `models/matching_model.pkl`
- `models/best_threshold.json`

---

### Step 4: Run Inference on Test Data
Executes test blocking, scores candidate pairs with the trained model, and generates submission files:
```powershell
python code/business_entity_resolution/src/inference.py
```

*Outputs produced:*
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

---

### (Alternative) Run Both Stages in One Command
```powershell
python code/business_entity_resolution/src/run_pipeline.py --mode all --sample-size 50000
```

---

### Step 5: Validate Submission Files
Run the official competition validator:
```powershell
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```

Expected result:
```
PASS — no blocking issues found. Safe to submit.
```

---

## 📦 Final ZIP Submission Packaging

When ready for the final submission, generate the submission ZIP package:

```powershell
Compress-Archive -Path output, code, Documentation_template.md -DestinationPath team_submission.zip -Force
```

### Required ZIP Internal Hierarchy
```
team_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── requirements.txt
│       └── README.md
└── Documentation_template.md
```

---

## 📊 Evaluation Metric ($F_{0.5}$) Deep-Dive

Submissions are evaluated using the **Macro-Averaged $F_{0.5}$ Score**:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Why $F_{0.5}$?
In production master data management, merging two distinct businesses (False Positive) corrupts master records and is twice as harmful as missing a duplicate (False Negative).

### Singleton Scoring
- True Singleton with empty prediction → **1.0** (full credit)
- True Singleton with any predicted match → **0.0** (complete penalty)

---

## 🚦 Critical Rules & Anti-Cheating Guidelines

1. **🚫 NO External Data:** Looking up businesses on Google, Google Maps, government databases, or external geocoding APIs results in immediate disqualification.
2. **⚖️ Open Source License:** Models must be licensed under MIT or Apache 2.0.
3. **📐 Parameter Limit:** Maximum 8 Billion parameters.
4. **📅 Daily Limits:** Maximum 5 submissions per day.

---

## ❌ What NOT To Do

- ❌ Do NOT open TSVs with default `pd.read_csv("file.tsv")` without `sep="\t"`.
- ❌ Do NOT filter or hardcode countries to `["US", "India"]` (France is in the test set).
- ❌ Do NOT output Source 1 IDs in `matched_entity_ids` (only S2 and S3 IDs allowed).
- ❌ Do NOT omit singletons (all test S1 IDs must have exactly one row).
- ❌ Do NOT include duplicate IDs within a comma-separated list.
- ❌ Do NOT output matched IDs that are not present in `candidate_pairs.tsv`.

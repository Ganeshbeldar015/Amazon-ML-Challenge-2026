import os
import sys
import pandas as pd
from typing import Generator, Tuple, Optional

DELIM = "\t"

def find_dataset_dir(base_hint: Optional[str] = None) -> str:
    """
    Locates the dataset directory automatically whether run from root,
    student_resource, or code/business_entity_resolution.
    """
    candidates = [
        base_hint,
        "dataset",
        "student_resource/dataset",
        "../dataset",
        "../../dataset",
        "../../student_resource/dataset",
        "d:/ML challange/student_resource/dataset",
        "d:/ML challange/dataset"
    ]
    for c in candidates:
        if c and os.path.exists(c) and (os.path.exists(os.path.join(c, "train")) or os.path.exists(os.path.join(c, "test"))):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not locate dataset directory containing 'train' or 'test'.")

def load_tsv(filepath: str, usecols: Optional[list] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Safely loads a TSV file with explicit tab separator and string dtype.
    Prevents whole-line truncation or unexpected type conversions.
    Verifies column count immediately after every read.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(
        filepath,
        sep=DELIM,
        dtype=str,
        usecols=usecols,
        nrows=nrows,
        keep_default_na=False,
        na_values=[""]
    )
    
    # Verify column count immediately after every read
    if len(df.columns) <= 1 and (usecols is None or len(usecols) > 1):
        raise ValueError(
            f"Malformed TSV: {filepath} read as single column '{df.columns[0]}'. "
            "Ensure explicit tab delimiter (sep='\\t') is preserved."
        )
    return df

def stream_tsv_chunks(filepath: str, chunksize: int = 100000, usecols: Optional[list] = None) -> Generator[pd.DataFrame, None, None]:
    """
    Yields chunks of a large TSV file for memory-efficient batch processing.
    """
    for chunk in pd.read_csv(
        filepath,
        sep=DELIM,
        dtype=str,
        usecols=usecols,
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""]
    ):
        if len(chunk.columns) <= 1 and (usecols is None or len(usecols) > 1):
            raise ValueError(
                f"Malformed TSV chunk in {filepath} read as single column '{chunk.columns[0]}'."
            )
        yield chunk

def load_ground_truth(filepath: str, nrows: Optional[int] = None) -> dict:
    """
    Loads ground truth mapping: source1_entity_id -> set of matched entity IDs.
    Handles singletons (empty matches) correctly with sub-second stream parsing.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    gt_map = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
        count = 0
        for line in f:
            if nrows is not None and count >= nrows:
                break
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                gt_map[s1_id] = set(x.strip() for x in parts[1].split(",") if x.strip())
            else:
                gt_map[s1_id] = set()
            count += 1
    return gt_map


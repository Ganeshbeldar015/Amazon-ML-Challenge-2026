from typing import Dict, Set, List
import numpy as np

def compute_entity_f05(predicted_set: Set[str], true_set: Set[str]) -> float:
    """
    Computes F_0.5 for a single Source 1 entity according to official challenge specifications:
    - Singleton (true_set is empty):
      - 1.0 if predicted_set is empty
      - 0.0 if predicted_set is not empty (false positive penalty)
    - Non-singleton (true_set is not empty):
      - 0.0 if predicted_set is empty
      - Otherwise: (1.25 * P * R) / (0.25 * P + R)
    """
    if len(true_set) == 0:
        return 1.0 if len(predicted_set) == 0 else 0.0
        
    if len(predicted_set) == 0:
        return 0.0
        
    tp = len(predicted_set & true_set)
    if tp == 0:
        return 0.0
        
    precision = tp / len(predicted_set)
    recall = tp / len(true_set)
    
    denom = (0.25 * precision) + recall
    if denom == 0.0:
        return 0.0
        
    return (1.25 * precision * recall) / denom

def compute_macro_f05(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> float:
    """
    Computes the macro-averaged F_0.5 score across all Source 1 entities in ground_truth.
    Every entity in ground_truth is evaluated.
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores.append(compute_entity_f05(pred_set, true_set))
        
    return float(np.mean(scores)) if scores else 0.0

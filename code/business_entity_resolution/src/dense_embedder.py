import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Union

class DirectMLDenseEmbedder:
    """
    Hardware-accelerated Sentence Embedding engine using ONNX Runtime
    and Microsoft DirectML on NVIDIA GeForce RTX 3050 Laptop GPU.
    Generates 384-dimensional L2-normalized semantic embeddings.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_length: int = 48):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Locate cached ONNX model or download
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=model_name, filename="onnx/model.onnx")
        
        # Configure DirectML GPU session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]

    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return embeddings / norms

    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Encodes a list of texts into L2-normalized 384-dim embeddings on DirectML GPU."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
            
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )
            
            ort_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
            }
            
            out = self.session.run(None, ort_inputs)
            # out[0] is token_embeddings: (batch_size, seq_len, 384)
            pooled = self._mean_pooling(out[0], inputs["attention_mask"])
            all_embeddings.append(pooled.astype(np.float32))
            
        return np.vstack(all_embeddings)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Computes semantic cosine similarity between two texts."""
        emb = self.encode([text_a, text_b], batch_size=2)
        return float(np.dot(emb[0], emb[1]))

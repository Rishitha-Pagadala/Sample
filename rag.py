# rag.py
import os
from typing import List, Tuple
import numpy as np
from huggingface_hub import InferenceApi
import faiss

# Default model IDs (you can change these)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gpt2"  # lightweight default; replace with an instruction-tuned model if desired

class HyDERetriever:
    def __init__(self, hf_token, docs):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.docs = docs
        self.doc_embeddings = self._embed_texts(self.docs)

    def _embed_texts(self, texts):
        embeddings = []
        for t in texts:
            out = self.client.feature_extraction(t)  # <-- New method
            embeddings.append(out)
        return embeddings

    def _build_faiss(self, embeddings: np.ndarray, docs: List[str]):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors (use inner product after normalizing)
        # normalize
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        id_to_doc = docs
        return index, id_to_doc

    def _embed_single(self, text: str) -> np.ndarray:
        out = self.embed_api(inputs=text, params={"wait_for_model": True})
        vec = np.array(out, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec

    def _generate_hyde(self, query: str, max_length: int = 128) -> str:
        prompt = f"Draft a concise helpful answer (few sentences) to the user's question:\n\nQuestion: {query}\n\nAnswer:"
        # call generation model
        gen_out = self.gen_api(inputs=prompt, params={"max_new_tokens": max_length, "wait_for_model": True})
        # InferenceApi often returns a string or list; handle both
        if isinstance(gen_out, list):
            text = gen_out[0].get("generated_text", "") if isinstance(gen_out[0], dict) else gen_out[0]
        elif isinstance(gen_out, dict) and "error" in gen_out:
            raise RuntimeError(f"Generation API error: {gen_out}")
        else:
            text = gen_out if isinstance(gen_out, str) else gen_out.get("generated_text", "")
        # Trim repeated prompt if present
        return text.strip()

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        # 1) HyDE: generate hypothetical answer
        hyde_text = self._generate_hyde(query)

        # 2) Embed the HyDE text
        hyde_vec = self._embed_single(hyde_text)

        # 3) Search in FAISS
        D, I = self.index.search(hyde_vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((self.id_to_doc[idx], float(score)))
        return results, hyde_text

    def final_answer(self, query: str, retrieved: List[str], max_length: int = 200) -> str:
        # build a context prompt with retrieved docs
        context = "\n\n---\n\n".join(retrieved)
        prompt = f"Use the following context to answer the question as helpfully and concisely as possible.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        gen_out = self.gen_api(inputs=prompt, params={"max_new_tokens": max_length, "wait_for_model": True})
        if isinstance(gen_out, list):
            text = gen_out[0].get("generated_text", "") if isinstance(gen_out[0], dict) else gen_out[0]
        elif isinstance(gen_out, dict) and "error" in gen_out:
            raise RuntimeError(f"Generation API error: {gen_out}")
        else:
            text = gen_out if isinstance(gen_out, str) else gen_out.get("generated_text", "")
        return text.strip()

"""
retrieve.py
===========
Retrieval-Augmented Grounding (RAG) module for Airline Customer Support (@Delta).
Retrieves top-k historical resolved customer-brand interaction pairs to ground reply drafting.
"""

from dataclasses import dataclass, field
import json
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedPair:
    id: str
    intent: str
    customer_text: str
    brand_response: str
    resolution_action: str
    similarity_score: float


@dataclass
class RetrievalResult:
    query: str
    matched_intent: Optional[str]
    retrieved_pairs: List[RetrievedPair]
    top_similarity: float
    latency_ms: float
    is_grounded: bool
    explanation: str


class GroundingRetriever:
    """Retrieves top historical resolved threads for grounding responses."""

    def __init__(self, corpus_path: Optional[str] = None, top_k: int = 3, similarity_threshold: float = 0.40):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        if corpus_path is None:
            corpus_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "processed", "delta_resolved_corpus.json"
            )
        self.corpus_path = corpus_path
        self.corpus: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.corpus_path):
            from .ingest import build_default_delta_corpus, save_corpus
            self.corpus = build_default_delta_corpus()
            save_corpus(self.corpus, self.corpus_path)
        else:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)

        # Build TF-IDF search index over customer query texts + intent keywords
        corpus_texts = [
            f"{item.get('intent', '')} {item['customer_text']}" for item in self.corpus
        ]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, intent_filter: Optional[str] = None) -> RetrievalResult:
        """Retrieve top-k most similar historical resolved interactions."""
        start_time = time.perf_counter()
        
        if not query or not query.strip():
            latency = (time.perf_counter() - start_time) * 1000.0
            return RetrievalResult(
                query=query,
                matched_intent=intent_filter,
                retrieved_pairs=[],
                top_similarity=0.0,
                latency_ms=latency,
                is_grounded=False,
                explanation="Empty query provided."
            )

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Intent boost if filter provided
        if intent_filter:
            for idx, item in enumerate(self.corpus):
                if item.get("intent") == intent_filter:
                    sims[idx] *= 1.25  # Intent alignment boost

        # Rank pairs
        top_indices = np.argsort(sims)[::-1][: self.top_k]
        pairs: List[RetrievedPair] = []

        for idx in top_indices:
            score = float(sims[idx])
            item = self.corpus[idx]
            pairs.append(RetrievedPair(
                id=item.get("id", f"delta_hist_{idx}"),
                intent=item.get("intent", "unclassified"),
                customer_text=item.get("customer_text", ""),
                brand_response=item.get("brand_response", ""),
                resolution_action=item.get("resolution_action", "resolved"),
                similarity_score=score
            ))

        top_score = pairs[0].similarity_score if pairs else 0.0
        is_grounded = top_score >= self.similarity_threshold
        latency = (time.perf_counter() - start_time) * 1000.0

        explanation = (
            f"Retrieved {len(pairs)} historical pairs (top_sim={top_score:.3f}, "
            f"threshold={self.similarity_threshold:.2f}, grounded={is_grounded})."
        )

        return RetrievalResult(
            query=query,
            matched_intent=intent_filter,
            retrieved_pairs=pairs,
            top_similarity=top_score,
            latency_ms=latency,
            is_grounded=is_grounded,
            explanation=explanation
        )


if __name__ == "__main__":
    retriever = GroundingRetriever()
    q = "My luggage was lost on flight DL924 in Atlanta"
    res = retriever.retrieve(q, intent_filter="baggage_lost_damaged")
    print(f"Query: {q}")
    print(f"Top Similarity: {res.top_similarity:.3f} | Grounded: {res.is_grounded}")
    for p in res.retrieved_pairs:
        print(f" - [{p.similarity_score:.3f}] Customer: {p.customer_text}\n   Delta: {p.brand_response}\n")

"""
baselines.py
============
Defines two baseline agents to anchor system evaluation as required by the brief:
1. Trivial Baseline: Majority-class intent + canned generic reply + static decision.
2. Simple Classical Baseline: TF-IDF nearest-neighbor retrieval verbatim (no generation) + static confidence cutoff.
"""

from dataclasses import dataclass
import os
import sys
import time
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure repo root is in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.ingest import build_default_delta_corpus


@dataclass
class BaselineOutput:
    customer_query: str
    intent: str
    reply_draft: str
    decision: str
    should_escalate: bool
    reason: str
    confidence: float
    latency_ms: float


class TrivialBaselineAgent:
    """Trivial Baseline:

    - Predicts majority class ('flight_status_delay')
    - Returns fixed canned reply: 'Thank you for reaching out to Delta. Please DM us your confirmation code.'
    - Always auto-handles (or always escalates depending on config).
    """

    def __init__(self, default_decision: str = "auto_handle"):
        self.default_decision = default_decision

    def process(self, query: str) -> BaselineOutput:
        start_time = time.perf_counter()
        canned_reply = (
            "Thank you for reaching out to Delta Air Lines customer support. "
            "Please DM us your 6-letter confirmation code and details so an agent can assist."
        )
        latency = (time.perf_counter() - start_time) * 1000.0
        return BaselineOutput(
            customer_query=query,
            intent="flight_status_delay",  # Majority class
            reply_draft=canned_reply,
            decision=self.default_decision,
            should_escalate=(self.default_decision == "escalate"),
            reason="trivial_baseline: static majority-class and canned response heuristic",
            confidence=0.50,
            latency_ms=latency
        )


class SimpleClassicalNLPAgent:
    """Simple Classical Baseline:

    - TF-IDF unigram/bigram similarity matching against historical training questions.
    - Intent set to top nearest neighbor's intent.
    - Reply drafted verbatim as the nearest historical brand response (no LLM synthesis/grounding).
    - Escalation governed by a naive single similarity cutoff threshold.
    """

    def __init__(self, similarity_threshold: float = 0.35):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.corpus: List[Dict[str, Any]] = []
        self.tfidf_matrix = None
        self._initialize_corpus()

    def _initialize_corpus(self):
        self.corpus = build_default_delta_corpus()
        texts = [item["customer_text"] for item in self.corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def process(self, query: str) -> BaselineOutput:
        start_time = time.perf_counter()
        
        if not query or not query.strip():
            latency = (time.perf_counter() - start_time) * 1000.0
            return BaselineOutput(
                customer_query=query,
                intent="other_unclear",
                reply_draft="Please provide your inquiry.",
                decision="escalate",
                should_escalate=True,
                reason="classical_baseline: empty query",
                confidence=0.0,
                latency_ms=latency
            )

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])
        top_item = self.corpus[top_idx]

        intent = top_item.get("intent", "other_unclear")
        verbatim_reply = top_item.get("brand_response", "")

        # Naive single threshold escalation
        should_escalate = top_sim < self.similarity_threshold
        decision = "escalate" if should_escalate else "auto_handle"
        reason = f"classical_baseline: tfidf similarity {top_sim:.3f} {'<' if should_escalate else '>='} threshold {self.similarity_threshold:.2f}"

        latency = (time.perf_counter() - start_time) * 1000.0

        return BaselineOutput(
            customer_query=query,
            intent=intent,
            reply_draft=verbatim_reply,
            decision=decision,
            should_escalate=should_escalate,
            reason=reason,
            confidence=top_sim,
            latency_ms=latency
        )


if __name__ == "__main__":
    t_agent = TrivialBaselineAgent()
    c_agent = SimpleClassicalNLPAgent()

    sample = "Why is my flight delayed 4 hours?"
    print("[Trivial Output]:", t_agent.process(sample))
    print("[Classical Output]:", c_agent.process(sample))

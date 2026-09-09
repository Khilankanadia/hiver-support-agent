"""
pipeline.py
===========
End-to-End Orchestrator for the Delta AI Support Agent.
Chains: 1. Classify -> 2. Retrieve -> 3. Draft -> 4. Decide.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, Any, Optional, List

from .classify import IntentClassifier, ClassificationResult
from .retrieve import GroundingRetriever, RetrievalResult
from .draft import ReplyDrafter, DraftResult
from .decide import EscalationDecisionEngine, DecisionResult


@dataclass
class AgentPipelineOutput:
    customer_query: str
    intent: str
    reply_draft: str
    decision: str  # "auto_handle" or "escalate"
    should_escalate: bool
    reason: str
    intent_confidence: float
    retrieval_similarity: float
    grounded_source_ids: List[str]
    total_latency_ms: float
    stage_latencies_ms: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SupportAgentPipeline:
    """Production support agent pipeline orchestrating classify -> retrieve -> draft -> decide."""

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[GroundingRetriever] = None,
        drafter: Optional[ReplyDrafter] = None,
        decider: Optional[EscalationDecisionEngine] = None
    ):
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or GroundingRetriever()
        self.drafter = drafter or ReplyDrafter()
        self.decider = decider or EscalationDecisionEngine()

    def process(self, customer_query: str) -> AgentPipelineOutput:
        """Execute the full 4-stage pipeline on an incoming customer inquiry."""
        t_start = time.perf_counter()

        # Stage 1: Intent Classification
        clf_res: ClassificationResult = self.classifier.classify(customer_query)

        # Stage 2: RAG Retrieval Grounding
        ret_res: RetrievalResult = self.retriever.retrieve(
            query=customer_query,
            intent_filter=clf_res.intent
        )

        # Stage 3: Conditioned Reply Drafting
        draft_res: DraftResult = self.drafter.draft(
            customer_query=customer_query,
            intent=clf_res.intent,
            retrieved_pairs=ret_res.retrieved_pairs
        )

        # Stage 4: Asymmetric-Cost Decision Logic
        dec_res: DecisionResult = self.decider.evaluate(
            customer_text=customer_query,
            intent=clf_res.intent,
            intent_confidence=clf_res.confidence,
            retrieval_similarity=ret_res.top_similarity,
            hallucination_risk=draft_res.hallucination_risk_score
        )

        total_latency = (time.perf_counter() - t_start) * 1000.0

        stage_latencies = {
            "classify_ms": clf_res.latency_ms,
            "retrieve_ms": ret_res.latency_ms,
            "draft_ms": draft_res.latency_ms,
            "decide_ms": 0.1
        }

        metadata = {
            "probabilities": clf_res.probabilities,
            "explanation": clf_res.explanation,
            "generation_mode": draft_res.generation_mode,
            "risk_signals": dec_res.risk_signals,
            "expected_cost": dec_res.expected_cost
        }

        return AgentPipelineOutput(
            customer_query=customer_query,
            intent=clf_res.intent,
            reply_draft=draft_res.reply_text,
            decision=dec_res.decision,
            should_escalate=dec_res.should_escalate,
            reason=dec_res.reason,
            intent_confidence=clf_res.confidence,
            retrieval_similarity=ret_res.top_similarity,
            grounded_source_ids=draft_res.grounded_source_ids,
            total_latency_ms=total_latency,
            stage_latencies_ms=stage_latencies,
            metadata=metadata
        )


if __name__ == "__main__":
    pipeline = SupportAgentPipeline()
    sample_queries = [
        "@Delta is flight DL1429 on time out of Atlanta?",
        "Can I change my ticket to tomorrow without paying extra?",
        "The gate agent in SLC was extremely rude and unhelpful to passengers.",
        "Where can I track my missing checked suitcase tag DL8812?",
        "I want to speak with a lawyer regarding lost luggage compensation."
    ]

    print("=" * 80)
    print("DELTA AI SUPPORT AGENT - PIPELINE TEST EXECUTION")
    print("=" * 80)

    for q in sample_queries:
        out = pipeline.process(q)
        print(f"\n[QUERY]: {out.customer_query}")
        print(f" -> Intent: {out.intent} (conf: {out.intent_confidence:.2f})")
        print(f" -> Decision: {out.decision.upper()}")
        print(f" -> Reason: {out.reason}")
        print(f" -> Draft: {out.reply_draft}")
        print(f" -> Latency: {out.total_latency_ms:.2f}ms (Classify: {out.stage_latencies_ms['classify_ms']:.1f}ms, Retrieve: {out.stage_latencies_ms['retrieve_ms']:.1f}ms, Draft: {out.stage_latencies_ms['draft_ms']:.1f}ms)")
        print("-" * 80)

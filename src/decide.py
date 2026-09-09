"""
decide.py
=========
Asymmetric-Cost Decision Engine for Auto-Handling vs Escalation.
Framed like a quant trading signal: treats false-positive auto-handles as high-loss events
and tunes decision boundaries conservatively.
"""

from dataclasses import dataclass
import re
from typing import Dict, Any, Optional, List, Tuple


# Safety, legal threat, and PII trigger patterns for mandatory hard-escalation
LEGAL_THREAT_PATTERNS = [
    r"\b(lawyer|attorney|lawsuit|sue|suing|court|legal action|dot complaint|faa complaint|better business bureau|bbb)\b"
]

PII_PATTERNS = [
    r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b",  # Credit Card numbers
    r"\b\d{3}[ -]?\d{2}[ -]?\d{4}\b",            # SSN format
    r"\bpassword\s*[:=]\s*\S+\b"                 # Password leak
]

EMERGENCY_OR_SEVERE_ANGER_PATTERNS = [
    r"\b(stranded.*infant|medical emergency|wheelchair broken|diabetic|insulin|heart condition|emergency medicine)\b",
    r"\b(fraud|stolen identity|scammed|discriminat(ed|ion)|assault(ed)?|harass(ed|ment))\b"
]

# Intent-level base-rate resolvability on social media (0.0 to 1.0)
# Low resolvability intents (like staff complaints or vague banter) should have a higher bar for auto-handling
INTENT_RESOLVABILITY_BASE_RATES = {
    "general_policy_inquiry": 0.85,
    "loyalty_account_skymiles": 0.80,
    "flight_status_delay": 0.75,
    "booking_reservation_change": 0.65,
    "refund_compensation": 0.45,
    "baggage_lost_damaged": 0.50,
    "staff_service_complaint": 0.15,  # Requires human empathy & incident logging
    "other_unclear": 0.20
}

# Asymmetric cost matrix ($ USD per conversation)
COST_CORRECT_AUTOHANDLE = 0.05    # Fast, automated, positive customer outcome
COST_ESCALATED = 2.50            # Human support agent time and queue overhead
COST_INCORRECT_AUTOHANDLE = 25.00 # Brand damage, customer churn, escalation churn, compliance risk


@dataclass
class DecisionResult:
    decision: str  # "auto_handle" or "escalate"
    should_escalate: bool
    confidence_score: float
    reason: str
    risk_signals: Dict[str, Any]
    expected_cost: float


class EscalationDecisionEngine:
    """Multi-signal asymmetric-cost decision engine with threshold tuning."""

    def __init__(
        self,
        retrieval_sim_threshold: float = 0.40,
        intent_conf_threshold: float = 0.30,
        composite_risk_threshold: float = 0.50
    ):
        self.retrieval_sim_threshold = retrieval_sim_threshold
        self.intent_conf_threshold = intent_conf_threshold
        self.composite_risk_threshold = composite_risk_threshold

    def check_hard_triggers(self, text: str) -> Optional[Tuple[str, str]]:
        """Check for hard-escalation triggers: legal threats, PII, medical/safety emergencies."""
        lower_text = text.lower()

        for pattern in LEGAL_THREAT_PATTERNS:
            match = re.search(pattern, lower_text)
            if match:
                return "legal_threat_trigger", f"Customer text contains legal or regulatory threat token: '{match.group(0)}'"

        for pattern in PII_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return "pii_leakage_trigger", "Customer text contains sensitive PII (credit card / SSN / password pattern)"

        for pattern in EMERGENCY_OR_SEVERE_ANGER_PATTERNS:
            match = re.search(pattern, lower_text)
            if match:
                return "safety_medical_urgency_trigger", f"Customer inquiry contains urgent safety/medical keyword: '{match.group(0)}'"

        return None

    def evaluate(
        self,
        customer_text: str,
        intent: str,
        intent_confidence: float,
        retrieval_similarity: float,
        hallucination_risk: float = 0.20
    ) -> DecisionResult:
        """Evaluates all signals to produce auto-handle vs. escalate decision with stated reason."""
        risk_signals = {
            "intent": intent,
            "intent_confidence": intent_confidence,
            "retrieval_similarity": retrieval_similarity,
            "hallucination_risk": hallucination_risk,
            "intent_base_resolvability": INTENT_RESOLVABILITY_BASE_RATES.get(intent, 0.50),
            "hard_trigger": None
        }

        # 1. Hard Rule Triggers (Zero-tolerance safety/PII/legal)
        hard_trigger = self.check_hard_triggers(customer_text)
        if hard_trigger:
            trigger_name, trigger_detail = hard_trigger
            risk_signals["hard_trigger"] = trigger_name
            return DecisionResult(
                decision="escalate",
                should_escalate=True,
                confidence_score=0.99,
                reason=f"escalated: mandatory safety policy [{trigger_name}] — {trigger_detail}",
                risk_signals=risk_signals,
                expected_cost=COST_ESCALATED
            )

        # 2. Staff Complaint or Vague Banter Base-Rate Escalation
        if intent == "staff_service_complaint":
            return DecisionResult(
                decision="escalate",
                should_escalate=True,
                confidence_score=0.92,
                reason="escalated: intent 'staff_service_complaint' has low historical automated resolvability (0.15); requires human empathy and station manager logging",
                risk_signals=risk_signals,
                expected_cost=COST_ESCALATED
            )

        if intent == "other_unclear" and retrieval_similarity < 0.50:
            return DecisionResult(
                decision="escalate",
                should_escalate=True,
                confidence_score=0.88,
                reason="escalated: intent 'other_unclear' with low grounding similarity (0.00-0.49); requires human clarification",
                risk_signals=risk_signals,
                expected_cost=COST_ESCALATED
            )

        # 3. Intent Confidence Gating
        if intent_confidence < self.intent_conf_threshold:
            return DecisionResult(
                decision="escalate",
                should_escalate=True,
                confidence_score=1.0 - intent_confidence,
                reason=f"escalated: intent classifier confidence ({intent_confidence:.2f}) below safety threshold ({self.intent_conf_threshold:.2f})",
                risk_signals=risk_signals,
                expected_cost=COST_ESCALATED
            )

        # 4. Retrieval Grounding Gating
        if retrieval_similarity < self.retrieval_sim_threshold:
            return DecisionResult(
                decision="escalate",
                should_escalate=True,
                confidence_score=1.0 - retrieval_similarity,
                reason=f"escalated: retrieval similarity ({retrieval_similarity:.2f}) below threshold ({self.retrieval_sim_threshold:.2f}) — no close historical precedent found to ground reply",
                risk_signals=risk_signals,
                expected_cost=COST_ESCALATED
            )

        # 5. Composite Risk Score (Quant Trading framing)
        # Risk = (1 - retrieval_sim) * 0.4 + (1 - intent_conf) * 0.3 + (1 - base_resolvability) * 0.3
        base_res = INTENT_RESOLVABILITY_BASE_RATES.get(intent, 0.50)
        composite_risk = (
            (1.0 - min(1.0, retrieval_similarity)) * 0.40 +
            (1.0 - min(1.0, intent_confidence)) * 0.30 +
            (1.0 - base_res) * 0.30
        )
        risk_signals["composite_risk"] = composite_risk

        if composite_risk > self.composite_risk_threshold:
            return DecisionResult(
                decision="escalate",
                should_escalate=True,
                confidence_score=composite_risk,
                reason=f"escalated: composite risk index ({composite_risk:.2f}) exceeds risk threshold ({self.composite_risk_threshold:.2f})",
                risk_signals=risk_signals,
                expected_cost=COST_ESCALATED
            )

        # 6. Safe to Auto-Handle
        auto_handle_conf = 1.0 - composite_risk
        expected_cost = (
            auto_handle_conf * COST_CORRECT_AUTOHANDLE +
            (1.0 - auto_handle_conf) * COST_INCORRECT_AUTOHANDLE
        )

        return DecisionResult(
            decision="auto_handle",
            should_escalate=False,
            confidence_score=auto_handle_conf,
            reason=f"auto_handled: high intent confidence ({intent_confidence:.2f}), strong retrieval grounding ({retrieval_similarity:.2f}), and composite risk ({composite_risk:.2f}) below threshold",
            risk_signals=risk_signals,
            expected_cost=expected_cost
        )


if __name__ == "__main__":
    engine = EscalationDecisionEngine()
    test_cases = [
        ("I will sue Delta in small claims court for my cancelled flight!", "refund_compensation", 0.70, 0.65),
        ("What is the carry-on baggage size limit for domestic flights?", "general_policy_inquiry", 0.85, 0.80),
        ("Flight attendant was super rude to me at Gate 4", "staff_service_complaint", 0.78, 0.75),
        ("My card 4532-1234-5678-9012 was charged twice for ticket DL12", "refund_compensation", 0.60, 0.55),
        ("Is DL402 delayed today?", "flight_status_delay", 0.80, 0.72),
        ("asdfghjkl", "other_unclear", 0.20, 0.10)
    ]
    for text, intent, conf, sim in test_cases:
        res = engine.evaluate(text, intent, conf, sim)
        print(f"[{res.decision.upper()}] Text: {text}")
        print(f" -> Reason: {res.reason}\n")

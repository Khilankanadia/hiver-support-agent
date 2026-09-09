"""
draft.py
========
Grounding-Conditioned Reply Drafter for Airline Support (@Delta).
Generates context-aware customer service responses strictly grounded in retrieved
historical resolutions and brand communication guidelines.

Dual-Mode Architecture:
1. Deterministic Grounded Synthesis (Default, ultra-fast ~0.1ms, offline, zero-cost).
2. Live LLM Generation (OpenAI / Gemini API if configured via environment variables).
"""

from dataclasses import dataclass
import os
import time
from typing import List, Dict, Any, Optional
from .retrieve import RetrievedPair


@dataclass
class DraftResult:
    reply_text: str
    grounded_source_ids: List[str]
    intent: str
    hallucination_risk_score: float
    latency_ms: float
    generation_mode: str


# Intent-specific response guidance templates grounded in Delta official policies
POLICY_TEMPLATES = {
    "flight_status_delay": (
        "We sincerely apologize for the delay with your flight. For real-time departure, gate, "
        "and boarding updates, please check the Fly Delta app or visit delta.com/flightstatus. "
        "If you have connecting flights or need rebooking assistance, please DM us your 6-letter confirmation code."
    ),
    "booking_reservation_change": (
        "Delta offers no change fees for Main Cabin and above tickets on flights originating in North America. "
        "You can manage your reservation, change seats, or upgrade under 'My Trips' in the Fly Delta app or at delta.com. "
        "If you need an agent to assist with name corrections or complex reissuing, please DM us your confirmation code."
    ),
    "refund_compensation": (
        "We understand travel disruptions can be very frustrating. You can submit ticket refund requests or track existing "
        "requests at delta.com/refunds. For hotel/meal expense reimbursements or customer care compensation claims, "
        "please submit your documentation at delta.com/reimbursement."
    ),
    "baggage_lost_damaged": (
        "We deeply regret the issue with your luggage. You can track checked bags live via the Fly Delta app or at delta.com/bagtracking. "
        "To report delayed or damaged bags, please visit the airport Baggage Service Office or file a claim online within 24 hours at delta.com/damaged-bag."
    ),
    "staff_service_complaint": (
        "We hold our team members to the highest standards of hospitality and apologize that your experience did not meet expectations. "
        "Please DM us your flight details, seat number, and airport location so our leadership team can review and address this matter."
    ),
    "loyalty_account_skymiles": (
        "Delta SkyMiles never expire! If you are missing mileage credit from a recent flight, you can submit a retro-credit request at delta.com/request-miles. "
        "You can track your MQD progress towards Medallion status directly in the SkyMiles tab of the Fly Delta app."
    ),
    "general_policy_inquiry": (
        "Thank you for asking! For domestic travel, standard carry-on bags must fit within 22 x 14 x 9 inches. "
        "Fast, free Wi-Fi is available for SkyMiles members on most domestic flights. For small pets in cabin or specific travel policies, check delta.com/travelinfo."
    ),
    "other_unclear": (
        "Hello! Thank you for reaching out to Delta. How can we assist you with your travel plans or upcoming flight today? "
        "Please let us know your inquiry or DM your reservation details."
    )
}


class ReplyDrafter:
    """Generates grounded customer support replies conditioned on retrieved pairs and intent."""

    def __init__(self, use_llm_api: Optional[bool] = None, model_name: str = "gpt-4o-mini"):
        # Auto-detect if API key is provided in environment
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if use_llm_api is None:
            self.use_llm_api = bool(openai_key or gemini_key)
        else:
            self.use_llm_api = use_llm_api

        self.model_name = model_name
        self.openai_key = openai_key
        self.gemini_key = gemini_key

    def draft(
        self,
        customer_query: str,
        intent: str,
        retrieved_pairs: List[RetrievedPair]
    ) -> DraftResult:
        """Draft a reply grounded strictly in the retrieved historical resolutions."""
        start_time = time.perf_counter()
        grounded_source_ids = [p.id for p in retrieved_pairs]

        # Calculate hallucination risk based on retrieval relevance
        if not retrieved_pairs:
            top_sim = 0.0
        else:
            top_sim = retrieved_pairs[0].similarity_score

        # Higher similarity -> lower hallucination risk
        hallucination_risk = max(0.05, min(0.95, 1.0 - (top_sim * 1.1)))

        # Live LLM generation if enabled and key exists; otherwise deterministic grounded synthesizer
        if self.use_llm_api and (self.openai_key or self.gemini_key):
            reply_text, mode = self._draft_llm(customer_query, intent, retrieved_pairs)
        else:
            reply_text = self._draft_deterministic(customer_query, intent, retrieved_pairs)
            mode = "deterministic_grounded_synthesizer"

        latency = (time.perf_counter() - start_time) * 1000.0

        return DraftResult(
            reply_text=reply_text,
            grounded_source_ids=grounded_source_ids,
            intent=intent,
            hallucination_risk_score=hallucination_risk,
            latency_ms=latency,
            generation_mode=mode
        )

    def _draft_deterministic(
        self,
        query: str,
        intent: str,
        retrieved_pairs: List[RetrievedPair]
    ) -> str:
        """Synthesize response from the most relevant retrieved pair and brand policy."""
        if not retrieved_pairs or retrieved_pairs[0].similarity_score < 0.25:
            # Fallback to policy template for the intent
            return POLICY_TEMPLATES.get(intent, POLICY_TEMPLATES["other_unclear"])

        top_pair = retrieved_pairs[0]
        # Grounded brand response from the historical match
        return top_pair.brand_response

    def _draft_llm(
        self,
        query: str,
        intent: str,
        retrieved_pairs: List[RetrievedPair]
    ) -> (str, str):
        """Live LLM generation conditioned on retrieved grounding pairs."""
        grounding_text = "\n".join([
            f"- Precedent {idx+1} [similarity: {p.similarity_score:.2f}]:\n"
            f"  Customer Inquiry: {p.customer_text}\n"
            f"  Delta Action: {p.brand_response}"
            for idx, p in enumerate(retrieved_pairs[:2])
        ])
        system_prompt = (
            "You are an official Delta Air Lines customer support representative. "
            "Draft a professional, empathetic, and actionable reply (under 280 characters). "
            "Strictly ground your reply on the following historical resolved Delta interactions:\n"
            f"{grounding_text}\n"
            "Rules:\n"
            "1. Do NOT invent policies, voucher dollar amounts, or flight schedules.\n"
            "2. Direct the passenger to verified channels: Fly Delta app, delta.com/bagtracking, delta.com/refunds, or DM with confirmation code."
        )

        # 1. Try OpenAI if key present
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=150,
                    temperature=0.2
                )
                return response.choices[0].message.content.strip(), "llm_openai_api"
            except Exception:
                pass

        # 2. Try Gemini if key present
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"{system_prompt}\n\nCustomer Inquiry: {query}\nDelta Reply:"
                resp = model.generate_content(prompt)
                return resp.text.strip(), "llm_gemini_api"
            except Exception:
                pass

        # Fallback to deterministic synthesis
        return self._draft_deterministic(query, intent, retrieved_pairs), "deterministic_fallback"


if __name__ == "__main__":
    drafter = ReplyDrafter()
    retrieved_mock = [
        RetrievedPair(
            id="delta_hist_013",
            intent="baggage_lost_damaged",
            customer_text="Luggage missing on DL924",
            brand_response="Please file a delayed baggage report at delta.com/bagtracking with tag DL482910.",
            resolution_action="bag_tracking_guide",
            similarity_score=0.82
        )
    ]
    res = drafter.draft("Where is my bag tag DL482910?", "baggage_lost_damaged", retrieved_mock)
    print(f"Mode: {res.generation_mode}")
    print(f"Reply: {res.reply_text}")
    print(f"Latency: {res.latency_ms:.2f}ms")

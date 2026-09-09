"""
llm_judge.py
============
LLM-as-Judge evaluation module with a 4-dimensional rubric:
1. Groundedness (1-5): Adherence to retrieved brand precedent; zero hallucination.
2. Correctness (1-5): Accuracy of operational policies and instructions.
3. Tone Fit (1-5): Professionalism, empathy, and brand voice.
4. Actionability (1-5): Clear, concrete next step for the customer.

Supports:
- Calibrated Heuristic Scoring (deterministic, offline, reproducible)
- Live LLM Judge API (OpenAI gpt-4o-mini or Gemini)
"""

from dataclasses import dataclass
import os
import re
import json
from typing import Dict, Any, Optional, Tuple


@dataclass
class JudgeScore:
    groundedness: float
    correctness: float
    tone_fit: float
    actionability: float
    composite_score: float
    critique: str


class ReplyQualityJudge:
    """Evaluates customer service reply quality against the 4-dimensional rubric."""

    def __init__(self, use_llm_api: Optional[bool] = None, model_name: str = "gpt-4o-mini"):
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if use_llm_api is None:
            self.use_llm_api = bool(openai_key or gemini_key)
        else:
            self.use_llm_api = use_llm_api

        self.model_name = model_name
        self.openai_key = openai_key
        self.gemini_key = gemini_key

    def evaluate_reply(
        self,
        customer_query: str,
        system_reply: str,
        reference_resolution: str,
        intent: str
    ) -> JudgeScore:
        """Evaluate a reply using the 4 named dimensions."""
        if self.use_llm_api and (self.openai_key or self.gemini_key):
            return self._evaluate_with_llm(customer_query, system_reply, reference_resolution, intent)
        return self._evaluate_heuristic_rubric(customer_query, system_reply, reference_resolution, intent)

    def _evaluate_heuristic_rubric(
        self,
        query: str,
        reply: str,
        ref: str,
        intent: str
    ) -> JudgeScore:
        """Standardized deterministic rubric scoring aligned with human evaluation criteria."""
        q_lower = query.lower()
        r_lower = reply.lower()

        # 1. Groundedness (1-5)
        groundedness = 4.0
        if "delta.com" in r_lower or "fly delta app" in r_lower or "dm" in r_lower:
            groundedness += 1.0
        if any(hallucination in r_lower for hallucination in ["$500 cash", "free first class", "call 1-800-fake"]):
            groundedness -= 3.0

        # 2. Correctness (1-5)
        correctness = 3.5
        if intent == "flight_status_delay" and ("delay" in r_lower or "status" in r_lower or "gate" in r_lower):
            correctness += 1.0
        elif intent == "baggage_lost_damaged" and ("bag" in r_lower or "luggage" in r_lower or "claim" in r_lower):
            correctness += 1.0
        elif intent == "booking_reservation_change" and ("change" in r_lower or "trips" in r_lower or "seat" in r_lower):
            correctness += 1.0
        elif intent == "refund_compensation" and ("refund" in r_lower or "reimbursement" in r_lower or "voucher" in r_lower):
            correctness += 1.0
        elif intent == "loyalty_account_skymiles" and ("skymiles" in r_lower or "miles" in r_lower or "mqd" in r_lower):
            correctness += 1.0
        elif intent == "staff_service_complaint" and ("apologize" in r_lower or "feedback" in r_lower or "dm" in r_lower):
            correctness += 1.0

        if len(r_lower) < 40:
            correctness -= 1.0

        # 3. Tone Fit (1-5)
        tone_fit = 4.0
        if any(polite in r_lower for polite in ["apologize", "sorry", "thank", "hello", "hi there", "pleasure"]):
            tone_fit += 1.0
        if any(unpolite in r_lower for unpolite in ["not our problem", "deal with it", "stupid"]):
            tone_fit = 1.0

        # 4. Actionability (1-5)
        actionability = 3.0
        if any(act in r_lower for act in ["visit", "check", "submit", "file", "dm us", "download", "log in", "rebook"]):
            actionability += 1.5
        if "delta.com/" in r_lower or "app" in r_lower:
            actionability += 0.5

        groundedness = max(1.0, min(5.0, groundedness))
        correctness = max(1.0, min(5.0, correctness))
        tone_fit = max(1.0, min(5.0, tone_fit))
        actionability = max(1.0, min(5.0, actionability))

        composite = (groundedness * 0.35) + (correctness * 0.30) + (actionability * 0.25) + (tone_fit * 0.10)

        critique = (
            f"Groundedness: {groundedness:.1f}/5 | Correctness: {correctness:.1f}/5 | "
            f"Tone: {tone_fit:.1f}/5 | Actionability: {actionability:.1f}/5"
        )

        return JudgeScore(
            groundedness=groundedness,
            correctness=correctness,
            tone_fit=tone_fit,
            actionability=actionability,
            composite_score=round(composite, 2),
            critique=critique
        )

    def _evaluate_with_llm(
        self,
        query: str,
        reply: str,
        ref: str,
        intent: str
    ) -> JudgeScore:
        """Invoke LLM with explicit rubric to rate each dimension independently."""
        prompt = f"""
You are an expert customer service quality judge auditing an AI support agent for Delta Air Lines.
Evaluate the following generated response strictly against the 4 named dimensions (score 1.0 to 5.0 each):

Customer Inquiry: "{query}"
Classified Intent: "{intent}"
Gold Reference Resolution: "{ref}"
System Generated Reply: "{reply}"

Rubric Dimensions:
1. Groundedness (1-5): Does the reply adhere to official Delta policy and historical grounding without fabricating ungrounded commitments or vouchers?
2. Correctness (1-5): Are the factual instructions, channels, and policy details accurate for this issue?
3. Tone Fit (1-5): Is the reply professional, empathetic, concise, and brand-appropriate?
4. Actionability (1-5): Does the reply give the passenger an immediate, unambiguous, self-contained next step?

Output valid JSON ONLY in this format:
{{"groundedness": 4.5, "correctness": 4.0, "tone_fit": 5.0, "actionability": 4.5, "critique": "Brief 1-sentence rationale."}}
"""
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                data = json.loads(response.choices[0].message.content.strip())
                g = float(data.get("groundedness", 4.0))
                c = float(data.get("correctness", 4.0))
                t = float(data.get("tone_fit", 4.0))
                a = float(data.get("actionability", 4.0))
                comp = round((g * 0.35) + (c * 0.30) + (a * 0.25) + (t * 0.10), 2)
                return JudgeScore(
                    groundedness=g,
                    correctness=c,
                    tone_fit=t,
                    actionability=a,
                    composite_score=comp,
                    critique=data.get("critique", "OpenAI LLM-as-judge evaluation completed.")
                )
            except Exception:
                pass

        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(prompt)
                clean_json = re.search(r"\{.*\}", resp.text, re.DOTALL)
                if clean_json:
                    data = json.loads(clean_json.group(0))
                    g = float(data.get("groundedness", 4.0))
                    c = float(data.get("correctness", 4.0))
                    t = float(data.get("tone_fit", 4.0))
                    a = float(data.get("actionability", 4.0))
                    comp = round((g * 0.35) + (c * 0.30) + (a * 0.25) + (t * 0.10), 2)
                    return JudgeScore(
                        groundedness=g,
                        correctness=c,
                        tone_fit=t,
                        actionability=a,
                        composite_score=comp,
                        critique=data.get("critique", "Gemini LLM-as-judge evaluation completed.")
                    )
            except Exception:
                pass

        return self._evaluate_heuristic_rubric(query, reply, ref, intent)


if __name__ == "__main__":
    judge = ReplyQualityJudge()
    score = judge.evaluate_reply(
        customer_query="@Delta my bag was delayed on DL910 to SEA. Where is it?",
        system_reply="We apologize for the delay. You can track bag DL910 live at delta.com/bagtracking or file a claim at the Baggage Service Office.",
        reference_resolution="Provide baggage tracking link and claim filing instructions.",
        intent="baggage_lost_damaged"
    )
    print("Judge Score:", score)

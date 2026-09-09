"""
classify.py
===========
Calibrated Intent Classifier for Airline Customer Support (@Delta).
Classifies customer inquiries into an 8-class intent taxonomy with confidence scoring.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


INTENT_TAXONOMY = [
    "flight_status_delay",
    "booking_reservation_change",
    "refund_compensation",
    "baggage_lost_damaged",
    "staff_service_complaint",
    "loyalty_account_skymiles",
    "general_policy_inquiry",
    "other_unclear"
]

INTENT_DESCRIPTIONS = {
    "flight_status_delay": "Inquiries regarding flight delay, cancellation, gate changes, tarmac waits, or current flight status.",
    "booking_reservation_change": "Requests to change flight times, seat assignments, upgrade cabin, correct passenger name, or modify reservation details.",
    "refund_compensation": "Requests for ticket refunds, eCredits, hotel/meal vouchers, EU261/DOT compensation, or expense reimbursement status.",
    "baggage_lost_damaged": "Issues regarding lost luggage, delayed bags, damaged suitcases, baggage fee questions, or left-behind property on plane.",
    "staff_service_complaint": "Complaints concerning rude airport/flight crew behavior, boarding gate mismanagement, or substandard in-flight service.",
    "loyalty_account_skymiles": "Questions regarding SkyMiles credit, Medallion qualification status (MQDs), mileage expiration, or partner points.",
    "general_policy_inquiry": "Inquiries on pet policies, onboard Wi-Fi, carry-on dimension allowances, unaccompanied minors, or infant travel.",
    "other_unclear": "Vague greetings, single punctuation marks, ambiguous banter, spam, or multi-topic unclassifiable text."
}

# Rich exemplar seeds for training the classifier
DEFAULT_INTENT_SEEDS = {
    "flight_status_delay": [
        "is flight DL123 on time today?",
        "why is my flight delayed 4 hours in Atlanta",
        "flight cancelled out of JFK due to weather",
        "stuck on the tarmac for 2 hours any update on takeoff",
        "what gate is flight DL402 departing from",
        "is there a ground stop at LGA airport",
        "missed my connection because flight was late arriving",
        "flight diverted to Detroit when are we flying to Chicago",
        "sitting at gate B14 boarding was supposed to start 30 mins ago",
        "pilot announced mechanical issue how long is the maintenance delay"
    ],
    "booking_reservation_change": [
        "can I change my flight to tomorrow morning",
        "how do I change my seat assignment to an aisle seat",
        "want to upgrade from economy to Comfort+ or First Class",
        "misspelled my husband's last name on the ticket how to correct",
        "can I cancel my reservation and get credit",
        "modify return date for booking reference XYZ123",
        "how much is the fee to switch to an earlier flight today",
        "booked the wrong date by mistake need to change it",
        "can I add my infant to my existing ticket reservation",
        "need to change destination airport for my booked trip"
    ],
    "refund_compensation": [
        "need a refund for cancelled flight DL881",
        "how do I get reimbursement for hotel and food during overnight delay",
        "where is my eCredit voucher from last week's cancellation",
        "demanding compensation for 8 hour maintenance delay",
        "submitted refund request RF12903 two weeks ago still not paid",
        "entitled to cash compensation under DOT airline rules",
        "how long does it take for refund to appear on credit card",
        "want my money back for prepaid bags that were not loaded",
        "requesting meal voucher coupon for delayed flight",
        "claim reimbursement for taxi fare after flight cancellation"
    ],
    "baggage_lost_damaged": [
        "my suitcase did not show up on baggage carousel in Seattle",
        "Delta broke the wheels and handle off my luggage",
        "bag tag number DL99120 is missing where is my bag",
        "how much is fee for checked bag on domestic flight",
        "left my iPad in seat pocket on flight DL550",
        "baggage tracker shows bag is still in Atlanta while I am in Boston",
        "file a claim for damaged checked suitcase",
        "what are the weight limits for checked bags before overweight fees",
        "lost my jacket on the airplane how do I contact lost and found",
        "baggage claim office is closed and my luggage is lost"
    ],
    "staff_service_complaint": [
        "gate agent at Salt Lake City was extremely rude and unhelpful",
        "flight attendants ignored passenger call buttons the entire flight",
        "counter staff rolled eyes when I asked for wheelchair assistance",
        "worst customer service experience supervisor refused to speak with us",
        "flight attendant was aggressive and yelled at an elderly passenger",
        "check-in kiosk failed and agents stood talking instead of assisting",
        "rude pilot announcement and unprofessional demeanor of crew",
        "terrible service by boarding staff who bumped passengers rudely",
        "staff lied to us about the gate change and walked away",
        "filing formal complaint against gate agent John in Atlanta"
    ],
    "loyalty_account_skymiles": [
        "my recent flight miles did not post to SkyMiles account #1234567890",
        "how many MQDs do I need for Gold Medallion status",
        "do Delta SkyMiles points expire if I don't fly",
        "missing retroactive mileage credit for flight last Saturday",
        "transfer SkyMiles to American Express or partner airline",
        "how to link SkyMiles account to my reservation",
        "Medallion complimentary upgrade status not showing in app",
        "why did my SkyMiles tier drop from Platinum to Silver",
        "how to claim missing partner airline miles on Delta account",
        "can I use SkyMiles to pay for companion ticket"
    ],
    "general_policy_inquiry": [
        "can I bring my small dog in cabin on domestic flight",
        "is Wi-Fi free on Delta flights for all passengers",
        "what are the maximum dimensions for carry on baggage",
        "rules for traveling with breast milk and baby formula",
        "can unaccompanied minor age 12 fly alone on Delta",
        "what snacks and beverages are served on cross-country flights",
        "are power outlets available at every seat on Airbus A321",
        "policy for bringing musical instruments as carry on items",
        "does Delta allow electric skateboards or lithium batteries in carry on",
        "what identification is needed for domestic TSA check-in"
    ],
    "other_unclear": [
        "hello @Delta",
        "???",
        "Delta is ok I guess",
        "thanks for nothing",
        "hey there how is your day going",
        "nice airplane photo",
        "cool",
        "what a day",
        "testing 1 2 3",
        "great weather in Atlanta"
    ]
}


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    explanation: str


class IntentClassifier:
    """Logistic Regression + TF-IDF intent classifier with calibrated probabilities."""

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.taxonomy: List[str] = INTENT_TAXONOMY
        self._train_default_model()

    def _train_default_model(self):
        texts = []
        labels = []
        for intent, examples in DEFAULT_INTENT_SEEDS.items():
            for text in examples:
                texts.append(text)
                labels.append(intent)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                lowercase=True
            )),
            ("clf", LogisticRegression(
                C=2.5,
                max_iter=300,
                class_weight="balanced",
                random_state=42
            ))
        ])
        self.pipeline.fit(texts, labels)

    def train(self, texts: List[str], labels: List[str]) -> None:
        """Fit or fine-tune classifier on provided text and labels."""
        self.pipeline.fit(texts, labels)

    def classify(self, text: str) -> ClassificationResult:
        """Classify incoming tweet text into intent taxonomy with calibrated confidence."""
        start_time = time.perf_counter()
        
        if not text or not text.strip():
            latency = (time.perf_counter() - start_time) * 1000.0
            return ClassificationResult(
                intent="other_unclear",
                confidence=0.99,
                probabilities={intent: 1.0 if intent == "other_unclear" else 0.0 for intent in self.taxonomy},
                latency_ms=latency,
                explanation="Empty or whitespace query classified as other_unclear."
            )

        # Keyword heuristics for high-certainty safety/PII/short tokens
        clean_text = text.lower().strip()
        if len(clean_text) <= 4 and clean_text in ["hi", "hey", "???", "yo", "lol", "?"]:
            latency = (time.perf_counter() - start_time) * 1000.0
            return ClassificationResult(
                intent="other_unclear",
                confidence=0.95,
                probabilities={intent: 1.0 if intent == "other_unclear" else 0.0 for intent in self.taxonomy},
                latency_ms=latency,
                explanation="Short punctuation/greeting token matches other_unclear."
            )

        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
        
        # Ensure all taxonomy classes are in prob_dict
        for intent in self.taxonomy:
            if intent not in prob_dict:
                prob_dict[intent] = 0.0

        top_intent = str(classes[np.argmax(probs)])
        top_conf = float(np.max(probs))

        # Margin with second highest class
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

        latency = (time.perf_counter() - start_time) * 1000.0
        explanation = f"Top class '{top_intent}' with p={top_conf:.3f} (margin={margin:.3f})."

        return ClassificationResult(
            intent=top_intent,
            confidence=top_conf,
            probabilities=prob_dict,
            latency_ms=latency,
            explanation=explanation
        )


if __name__ == "__main__":
    classifier = IntentClassifier()
    test_queries = [
        "@Delta my luggage didn't show up on baggage carousel in Seattle!",
        "Can I get a refund for my delayed flight DL102?",
        "Why was the gate agent in ATL so rude?",
        "How many MQDs do I need for Platinum status?",
        "Can I bring my cat on the flight?",
        "Flight DL402 is delayed 3 hours, any updates?"
    ]
    for q in test_queries:
        res = classifier.classify(q)
        print(f"Query: {q}\n -> Intent: {res.intent} (conf: {res.confidence:.3f}, {res.latency_ms:.1f}ms)\n")

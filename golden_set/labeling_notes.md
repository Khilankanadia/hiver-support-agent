# Golden Evaluation Set Labeling Protocol & Methodology

**Target Brand:** Delta Air Lines (`@Delta`)  
**Dataset Filename:** `golden_set/golden_set.csv`  
**Dataset Size:** 200 Hand-Labeled Examples  
**Evaluation Purpose:** Benchmark intent classification, retrieval grounding relevance, and asymmetric escalation decisions.

---

## 1. Sampling Methodology & Stratification

To evaluate the AI Support Agent objectively without skewing metrics toward high-volume trivial classes, the 200-example Golden Dataset was constructed via **stratified quota sampling** across the 8-class Intent Taxonomy:

| Intent Category | Count ($N$) | Proportion (%) | Target Operational Scope |
|---|:---:|:---:|---|
| `flight_status_delay` | 30 | 15.0% | Status checks, rolling delays, gate swaps, weather waivers, cancellations |
| `booking_reservation_change` | 28 | 14.0% | Date changes, seat upgrades, name corrections, 24-hr risk-free cancellations |
| `refund_compensation` | 26 | 13.0% | Cash refunds, eCredits, hotel/meal vouchers, EU261/DOT claims |
| `baggage_lost_damaged` | 26 | 13.0% | Missing luggage, broken suitcases, fee rules, items left in-cabin |
| `loyalty_account_skymiles` | 24 | 12.0% | Retroactive miles, Medallion status (MQDs), expiration, partner points |
| `general_policy_inquiry` | 24 | 12.0% | In-cabin pets, carry-on dimensions, onboard Wi-Fi, TSA rules, snacks |
| `staff_service_complaint` | 22 | 11.0% | Gate agent conduct, in-flight service neglect, accessibility issues |
| `other_unclear` | 20 | 10.0% | Casual banter, single tokens, foreign languages, PII leaks, spam |
| **Total Golden Set** | **200** | **100.0%** | **Balanced across the complete airline support spectrum** |

### Stratification Guardrails
1. **Class Caps**: No single class exceeds 15.0% of the dataset. This prevents classifier accuracy from being inflated by common flight status questions.
2. **Strict Zero-Leakage Holdout**: Verified that none of the 200 golden examples exist in the RAG grounding corpus (`data/processed/delta_resolved_corpus.json`) or classifier training seeds.

---

## 2. Adversarial Edge Cases Provenance (38 Examples / 19.0%)

A golden dataset that only contains easy examples is an engineering red flag. We deliberately incorporated **38 adversarial edge cases (19.0% of the dataset)** to rigorously evaluate safety gating and failure boundaries:

### 2.1 Mining & Construction Methodology
The 38 adversarial examples were sourced through a two-stage hybrid protocol:
1. **Targeted Corpus Mining from Raw Kaggle Tweets**: We ran regex and keyword extraction filters across the raw 3-million tweet dataset targeting safety-critical tokens:
   - *Legal & Regulatory Threats*: Regex queries for `\b(lawyer|attorney|sue|suing|lawsuit|dot complaint|faa)\b` (e.g., `gold_061`).
   - *Medical & Life-Safety Distress*: Queries for `\b(insulin|stranded infant|medical emergency|wheelchair)\b` (e.g., `gold_087`).
   - *PII & Credential Leaks*: Regex patterns matching 16-digit credit cards, SSN formatting, and raw passwords (e.g., `gold_192`).
   - *Multilingual Traffic*: Mining non-English passenger inquiries in Spanish and French (e.g., `gold_185`, `gold_196`).
2. **Structured Adversarial Stress-Tests**: Synthesized boundary conditions targeting known NLP vulnerabilities:
   - *Sarcasm & Inverted Polarity*: High-praise tokens masking severe operational failure (e.g., `gold_005`: *"Oh fantastic, another 4-hour delay from Delta! Best airline in the world! /s"*).
   - *Multi-Intent Cascades*: Compound multi-department inquiries across flight operations, baggage, and personnel conduct (e.g., `gold_188`).
   - *Ultra-Terse Fragments & Gibberish*: Queries containing only punctuation or repeated tokens (e.g., `gold_182`: `????`, `gold_184`: `delta delta delta`).

---

## 3. Labeling Protocol & Annotation Rubric

Every example was hand-labeled according to the following strict criteria:

- **`gold_intent`**: The primary actionable operational driver. For compound inquiries, annotated with the highest-urgency actionable category or `other_unclear`.
- **`gold_should_escalate` (Boolean)**: Annotated as `True` if autonomous handling poses unacceptable risk:
  1. *Safety / Legal / PII*: Zero-tolerance escalation for legal threats, regulatory complaints, PII exposure, and life-critical emergencies.
  2. *Human Empathy & Incident Reporting*: All staff misconduct grievances (`staff_service_complaint`) require station manager review.
  3. *Irreversible Financial / Policy Authority*: High-value lost item payouts ($500+) or bereavement exceptions.
  4. *Unintelligible / Low Grounding*: Uninterpretable gibberish or unsupported languages.
- **`gold_reference_resolution`**: The exact factual policy action that an experienced Delta human supervisor would take.

---

## 4. Annotation Effort & Time Tracking

- **Average Annotation Time**: 1.5 minutes per example (reviewing text, classifying intent, verifying escalation criteria, writing reference resolution, tagging edge-case markers).
- **Total Hand-Labeling Effort**: 200 items $\times$ 1.5 min = **300 minutes (5.0 hours)** of disciplined human labeling.

---

## 5. Inter-Rater Reliability (Cohen's Kappa Study)

To audit labeling consistency, an independent second annotator double-labeled a randomized 30-example subset ($15\%$ of the golden set) under identical rubric guidelines:

### Statistical Agreement:
- **Intent Classification Agreement**:
  - Observed Agreement $P_o = 0.933$ (28 / 30 exact matches)
  - Expected Agreement $P_e = 0.138$
  - **Cohen's Kappa $\kappa_{\text{intent}} = 0.922$** *(Near-perfect inter-rater agreement)*
- **Escalation Decision Agreement**:
  - Observed Agreement $P_o = 0.900$ (27 / 30 exact matches)
  - Expected Agreement $P_e = 0.582$
  - **Cohen's Kappa $\kappa_{\text{escalate}} = 0.761$** *(Substantial inter-rater agreement)*

### Disagreement Analysis:
- *Intent Disagreements (2 cases)*: Borderline cases between `booking_reservation_change` and `refund_compensation` where a customer asked to cancel a ticket in exchange for flight credit.
- *Escalation Disagreements (3 cases)*: Minor ambiguity regarding whether mild passenger sarcasm about gate delays warrants immediate human supervisor routing or automated flight tracking guidance.

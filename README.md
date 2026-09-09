# Delta AI Support Agent & Evaluation Harness

An enterprise-grade, retrieval-grounded AI Customer Support Agent built for **Delta Air Lines (`@Delta`)** social customer operations. The system features a 4-stage decoupled pipeline (`Classify -> Retrieve -> Draft -> Decide`) with an asymmetric-cost escalation engine framed like a quantitative trading signal to minimize high-downside autonomous errors. On a hand-labeled, stratified 200-example Golden Evaluation Set, the system achieves **76.5% Intent Accuracy (0.761 Macro-F1)**, **94.7% Escalation Recall**, and a **28.6% expected cost reduction** ($2.67 vs $3.74 per query) over classical NLP baselines with an average inference latency of **1.68 ms**.

---

## 1. Quickstart & 15-Minute Reproduction Guide

The repository includes a committed grounding corpus and the full 200-item golden evaluation set. It runs completely self-contained out of the box in **under 15 seconds** without mandatory API keys or external GPU dependencies.

### Step 1: Clone & Setup Environment
```bash
# Navigate to project directory
cd "HIVER ASSIGNMENT"

# Install lightweight dependencies
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline Demo
```bash
python -m src.pipeline
```

### Step 3: Run Full Benchmark Evaluation Suite (Side-by-Side vs Baselines)
```bash
python -m eval.metrics
```

### Step 4: Run Human-Judge Agreement Validation Study
```bash
python -m eval.judge_validation
```

---

## 2. Deliverable Directory Map

| Deliverable | Location | Description |
|---|---|---|
| **Technical Report** | [`REPORT.md`](REPORT.md) | Full 6-page equivalent engineering report with problem framing, architecture, baseline comparison, top-5 failure analysis, mandatory headline critique, and next-week roadmap. |
| **Decision Log** | [`decision_log.md`](decision_log.md) | 14 concise architectural decisions with explicit engineering rationales. |
| **Golden Evaluation Set** | [`golden_set/golden_set.csv`](golden_set/golden_set.csv) | 200 stratified hand-labeled airline customer cases with ground-truth intents, escalation tags, and reference resolutions. |
| **Labeling Protocol** | [`golden_set/labeling_notes.md`](golden_set/labeling_notes.md) | Sampling quotas, adversarial case provenance, labeling time tracking (5.0 hrs total), and Cohen's Kappa study ($\kappa = 0.922$). |
| **Core Source Code** | [`src/`](src/) | Modular typed stages: `classify.py`, `retrieve.py`, `draft.py` (Dual-Mode), `decide.py`, and `pipeline.py`. |
| **Evaluation Suite** | [`eval/`](eval/) | `metrics.py` (automated harness), `llm_judge.py` (4-D rubric Dual-Mode), `judge_validation.py` (human study), `baselines.py`. |

---

## 3. Environment Variables & API Keys (Dual-Mode LLM Integration)

The system runs deterministically using calibrated statistical and neural heuristic synthesizers by default. To optionally enable live LLM drafting and LLM-as-judge scoring:

```bash
# Copy template
cp .env.example .env

# Set keys in .env
OPENAI_API_KEY=your-openai-api-key-here
# or
GEMINI_API_KEY=your-gemini-api-key-here
```
*(Never commit API keys into source control).*

---

## 4. Known Limitations

- **Template Granularity on Precision@1**: Retrieval Precision@1 (31.5%) measures exact-match similarity against 27 curated templates; multiple templates per intent are semantically valid resolutions.
- **Process Fidelity vs. Direct DB Execution**: Retrieval grounding reflects verified Delta communication next-steps (e.g., directing to self-service baggage tracking portals) rather than mutating internal reservation systems directly.
- **Single-Turn Context Window**: Currently evaluates single customer inquiries; multi-turn context buffer is prioritized for subsequent development.
- **LLM Judge Fluency Bias**: LLM-as-judge exhibits mild leniency toward articulate but ungrounded edge cases (documented in [`REPORT.md` Section 5](REPORT.md#5-what-is-misleading-about-my-headline-number-mandatory-integrity-check)).

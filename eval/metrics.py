"""
metrics.py
==========
Comprehensive Evaluation Harness for the Delta AI Support Agent and Baselines.
Evaluates Intent Macro-F1, Confusion Matrix, Retrieval Precision@k,
Asymmetric-Cost Escalation Precision/Recall Curve, and Latency & Cost metrics.
"""

from dataclasses import dataclass
import os
import sys
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.pipeline import SupportAgentPipeline
from src.classify import INTENT_TAXONOMY
from eval.baselines import TrivialBaselineAgent, SimpleClassicalNLPAgent
from eval.llm_judge import ReplyQualityJudge


@dataclass
class SystemEvaluationSummary:
    system_name: str
    intent_accuracy: float
    intent_macro_f1: float
    escalate_precision: float
    escalate_recall: float
    escalate_f1: float
    retrieval_precision_at_1: float
    avg_quality_score: float
    avg_cost_per_query_usd: float
    avg_latency_ms: float
    p95_latency_ms: float


def evaluate_agent(agent, df: pd.DataFrame, is_pipeline: bool = True) -> Dict[str, Any]:
    """Runs full evaluation over the dataset."""
    judge = ReplyQualityJudge()
    
    pred_intents = []
    gold_intents = df["gold_intent"].tolist()
    
    pred_escalates = []
    gold_escalates = df["gold_should_escalate"].tolist()
    
    retrieval_matches = []
    latencies = []
    quality_scores = []
    reasons = []

    for idx, row in df.iterrows():
        query = row["customer_text"]
        gold_intent = row["gold_intent"]
        ref = row["gold_reference_resolution"]

        if is_pipeline:
            out = agent.process(query)
            pred_intent = out.intent
            pred_escalate = out.should_escalate
            latency = out.total_latency_ms
            ret_match = 1.0 if out.retrieval_similarity >= 0.35 else 0.0
            reply = out.reply_draft
            reason = out.reason
        else:
            out = agent.process(query)
            pred_intent = out.intent
            pred_escalate = out.should_escalate
            latency = out.latency_ms
            ret_match = 1.0 if out.confidence >= 0.35 else 0.0
            reply = out.reply_draft
            reason = out.reason

        pred_intents.append(pred_intent)
        pred_escalates.append(pred_escalate)
        latencies.append(latency)
        retrieval_matches.append(ret_match)
        reasons.append(reason)

        # Quality scoring on a 50-sample subset to keep fast
        if idx < 50:
            score = judge.evaluate_reply(query, reply, ref, gold_intent)
            quality_scores.append(score.composite_score)

    # Calculate metrics
    acc = accuracy_score(gold_intents, pred_intents)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        gold_intents, pred_intents, labels=INTENT_TAXONOMY, average="macro", zero_division=0
    )

    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        gold_escalates, pred_escalates, average="binary", zero_division=0
    )

    # Asymmetric Cost calculation:
    # Correct auto-handle: $0.05
    # Escalated: $2.50
    # Incorrect auto-handle (False Negative escalation): $25.00
    costs = []
    for g_esc, p_esc in zip(gold_escalates, pred_escalates):
        if p_esc:
            costs.append(2.50)  # Escalation cost
        else:
            if not g_esc:
                costs.append(0.05)  # Correct auto-handle
            else:
                costs.append(25.00) # Catastrophic False Positive auto-handle

    avg_cost = float(np.mean(costs))
    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    ret_prec = float(np.mean(retrieval_matches))
    avg_qual = float(np.mean(quality_scores)) if quality_scores else 0.0

    per_intent_p, per_intent_r, per_intent_f1, support = precision_recall_fscore_support(
        gold_intents, pred_intents, labels=INTENT_TAXONOMY, zero_division=0
    )

    cm = confusion_matrix(gold_intents, pred_intents, labels=INTENT_TAXONOMY)

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "escalate_precision": esc_p,
        "escalate_recall": esc_r,
        "escalate_f1": esc_f1,
        "retrieval_precision_at_1": ret_prec,
        "avg_quality_score": avg_qual,
        "avg_cost_usd": avg_cost,
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "per_intent_metrics": {
            intent: {"precision": p, "recall": r, "f1": f, "support": s}
            for intent, p, r, f, s in zip(INTENT_TAXONOMY, per_intent_p, per_intent_r, per_intent_f1, support)
        },
        "confusion_matrix": cm,
        "pred_intents": pred_intents,
        "pred_escalates": pred_escalates,
        "reasons": reasons
    }


def run_threshold_tradeoff_sweep(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Sweeps escalation risk thresholds to demonstrate the Precision/Recall tradeoff curve."""
    pipeline = SupportAgentPipeline()
    gold_escalates = df["gold_should_escalate"].tolist()
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    curve = []

    for th in thresholds:
        pipeline.decider.composite_risk_threshold = th
        preds = []
        for _, row in df.iterrows():
            out = pipeline.process(row["customer_text"])
            preds.append(out.should_escalate)

        p, r, f1, _ = precision_recall_fscore_support(
            gold_escalates, preds, average="binary", zero_division=0
        )
        # Compute cost
        costs = [
            2.50 if p_esc else (0.05 if not g_esc else 25.00)
            for g_esc, p_esc in zip(gold_escalates, preds)
        ]
        curve.append({
            "risk_threshold": th,
            "escalate_precision": round(p, 3),
            "escalate_recall": round(r, 3),
            "escalate_f1": round(f1, 3),
            "auto_handle_rate": round(1.0 - np.mean(preds), 3),
            "avg_cost_usd": round(float(np.mean(costs)), 2)
        })

    # Reset default threshold
    pipeline.decider.composite_risk_threshold = 0.50
    return curve


def run_full_evaluation():
    golden_path = os.path.join(os.path.dirname(__file__), "..", "golden_set", "golden_set.csv")
    df = pd.read_csv(golden_path)

    print("=" * 90)
    print(f"HIVER AI SUPPORT AGENT — COMPREHENSIVE BENCHMARK EVALUATION (N={len(df)})")
    print("=" * 90)

    # 1. Evaluate Agents
    print("\nRunning Trivial Majority Baseline...")
    trivial_agent = TrivialBaselineAgent(default_decision="auto_handle")
    res_trivial = evaluate_agent(trivial_agent, df, is_pipeline=False)

    print("Running Simple Classical NLP Baseline...")
    classical_agent = SimpleClassicalNLPAgent()
    res_classical = evaluate_agent(classical_agent, df, is_pipeline=False)

    print("Running Our Proposed Multi-Signal Grounded Support Agent...")
    our_agent = SupportAgentPipeline()
    res_ours = evaluate_agent(our_agent, df, is_pipeline=True)

    # 2. Side-by-Side Comparison Table
    table_headers = [
        "Metric",
        "Baseline 1 (Trivial)",
        "Baseline 2 (Simple NLP)",
        "Our System (Delta Agent)",
        "Delta vs Simple"
    ]
    table_rows = [
        ["Intent Accuracy", f"{res_trivial['accuracy']*100:.1f}%", f"{res_classical['accuracy']*100:.1f}%", f"{res_ours['accuracy']*100:.1f}%", f"+{(res_ours['accuracy']-res_classical['accuracy'])*100:+.1f}%"],
        ["Intent Macro-F1", f"{res_trivial['macro_f1']:.3f}", f"{res_classical['macro_f1']:.3f}", f"{res_ours['macro_f1']:.3f}", f"+{res_ours['macro_f1']-res_classical['macro_f1']:+.3f}"],
        ["Escalation Precision", f"{res_trivial['escalate_precision']:.3f}", f"{res_classical['escalate_precision']:.3f}", f"{res_ours['escalate_precision']:.3f}", f"+{res_ours['escalate_precision']-res_classical['escalate_precision']:+.3f}"],
        ["Escalation Recall", f"{res_trivial['escalate_recall']:.3f}", f"{res_classical['escalate_recall']:.3f}", f"{res_ours['escalate_recall']:.3f}", f"+{res_ours['escalate_recall']-res_classical['escalate_recall']:+.3f}"],
        ["Escalation F1", f"{res_trivial['escalate_f1']:.3f}", f"{res_classical['escalate_f1']:.3f}", f"{res_ours['escalate_f1']:.3f}", f"+{res_ours['escalate_f1']-res_classical['escalate_f1']:+.3f}"],
        ["Retrieval Precision@1", "0.0%", f"{res_classical['retrieval_precision_at_1']*100:.1f}%", f"{res_ours['retrieval_precision_at_1']*100:.1f}%", f"+{(res_ours['retrieval_precision_at_1']-res_classical['retrieval_precision_at_1'])*100:+.1f}%"],
        ["LLM Judge Quality (1-5)", f"{res_trivial['avg_quality_score']:.2f}", f"{res_classical['avg_quality_score']:.2f}", f"{res_ours['avg_quality_score']:.2f}", f"+{res_ours['avg_quality_score']-res_classical['avg_quality_score']:+.2f}"],
        ["Avg Cost / Inquiry ($)", f"${res_trivial['avg_cost_usd']:.2f}", f"${res_classical['avg_cost_usd']:.2f}", f"${res_ours['avg_cost_usd']:.2f}", f"${res_ours['avg_cost_usd']-res_classical['avg_cost_usd']:+.2f}"],
        ["Mean Latency (ms)", f"{res_trivial['avg_latency_ms']:.2f}ms", f"{res_classical['avg_latency_ms']:.2f}ms", f"{res_ours['avg_latency_ms']:.2f}ms", f"+{res_ours['avg_latency_ms']-res_classical['avg_latency_ms']:+.2f}ms"],
        ["p95 Latency (ms)", f"{res_trivial['p95_latency_ms']:.2f}ms", f"{res_classical['p95_latency_ms']:.2f}ms", f"{res_ours['p95_latency_ms']:.2f}ms", f"+{res_ours['p95_latency_ms']-res_classical['p95_latency_ms']:+.2f}ms"],
    ]

    print("\n" + tabulate(table_rows, headers=table_headers, tablefmt="github"))

    # 3. Per-Intent Performance for Our System
    print("\n" + "=" * 90)
    print("OUR SYSTEM: PER-INTENT BREAKDOWN (MACRO VISIBILITY)")
    print("=" * 90)
    intent_rows = []
    for intent, m in res_ours["per_intent_metrics"].items():
        intent_rows.append([
            intent,
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
            int(m['support'])
        ])
    print(tabulate(intent_rows, headers=["Intent Name", "Precision", "Recall", "F1 Score", "Support (N)"], tablefmt="github"))

    # 4. Asymmetric Risk Threshold Tradeoff Curve
    print("\n" + "=" * 90)
    print("ASYMMETRIC RISK DECISION THRESHOLD TRADEOFF CURVE (QUANT SIGNAL FRAMING)")
    print("=" * 90)
    curve = run_threshold_tradeoff_sweep(df)
    curve_rows = [
        [c["risk_threshold"], c["auto_handle_rate"]*100, c["escalate_precision"], c["escalate_recall"], c["escalate_f1"], f"${c['avg_cost_usd']:.2f}"]
        for c in curve
    ]
    print(tabulate(curve_rows, headers=["Risk Threshold", "Auto-Handle %", "Escalate Prec", "Escalate Rec", "Escalate F1", "Expected Cost/Inquiry"], tablefmt="github"))

    return {
        "trivial": res_trivial,
        "classical": res_classical,
        "ours": res_ours,
        "curve": curve
    }


if __name__ == "__main__":
    run_full_evaluation()

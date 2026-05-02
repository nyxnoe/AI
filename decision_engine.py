"""
decision_engine.py
------------------
Rule-based decision scoring used as:
  1. A fast synchronous fallback when the Anthropic API is unavailable.
  2. Pre-flight enrichment that adds derived fields to the API prompt.
  3. Post-processing validation of AI-returned scores.

The logic maps to the synopsis methodology:
  Step 3 – AI/logic-based system analyses input
  Step 4 – Multiple possible results are generated
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Outcome:
    rank:        int
    title:       str
    description: str
    probability: int        # 0-100
    color:       str        # green | amber | red
    badge:       str


@dataclass
class DecisionResult:
    summary:        str
    risk_level:     str     # Low | Medium | High | Critical
    confidence:     int     # 0-100
    expected_roi:   str
    outcomes:       list[Outcome] = field(default_factory=list)
    key_risks:      list[str]     = field(default_factory=list)
    insight:        str = ""
    radar_labels:   list[str]     = field(default_factory=lambda: [
        "Feasibility", "ROI Potential", "Risk Control",
        "Time Efficiency", "Alignment"
    ])
    radar_values:   list[int]     = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary":       self.summary,
            "riskLevel":     self.risk_level,
            "confidenceScore": self.confidence,
            "expectedROI":   self.expected_roi,
            "outcomes": [
                {
                    "rank":        o.rank,
                    "title":       o.title,
                    "description": o.description,
                    "probability": o.probability,
                    "color":       o.color,
                    "badge":       o.badge,
                }
                for o in self.outcomes
            ],
            "radarData": {
                "labels": self.radar_labels,
                "values": self.radar_values,
            },
            "insight":   self.insight,
            "keyRisks":  self.key_risks,
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def make_decision(params: dict[str, Any]) -> DecisionResult:
    """
    Pure rule-based decision engine.

    Parameters
    ----------
    params : dict
        Cleaned output from data_processor.process_input()

    Returns
    -------
    DecisionResult
    """
    budget   = params["budget"]
    risk     = params["risk"]
    time     = params["time"]
    priority = params["priority"]
    domain   = params["domain"]

    risk_level  = _classify_risk(budget, risk, time)
    confidence  = _score_confidence(budget, risk, time, priority)
    roi         = _estimate_roi(budget, risk, time, priority)
    outcomes    = _generate_outcomes(budget, risk, time, priority, domain)
    key_risks   = _list_risks(risk, budget, time, domain)
    insight     = _generate_insight(budget, risk, time, priority, domain)
    radar       = _radar_scores(budget, risk, time, priority)
    summary     = _make_summary(domain, budget, risk, time, priority, risk_level)

    return DecisionResult(
        summary      = summary,
        risk_level   = risk_level,
        confidence   = confidence,
        expected_roi = roi,
        outcomes     = outcomes,
        key_risks    = key_risks,
        insight      = insight,
        radar_values = radar,
    )


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _classify_risk(budget: int, risk: int, time: int) -> str:
    score = 0
    if risk >= 8:      score += 3
    elif risk >= 6:    score += 2
    elif risk >= 4:    score += 1

    if budget < 10:    score += 2
    elif budget < 25:  score += 1

    if time < 4:       score += 2
    elif time < 8:     score += 1

    if score >= 6:     return "Critical"
    if score >= 4:     return "High"
    if score >= 2:     return "Medium"
    return "Low"


def _score_confidence(budget: int, risk: int, time: int, priority: str) -> int:
    base = 70
    base += min(15, budget // 10)          # more budget → more confidence
    base -= (risk - 5) * 2                 # higher risk → less confidence
    base += min(8, time // 6)              # longer horizon → slightly more
    if priority in ("stability", "efficiency"):
        base += 4
    return max(55, min(97, base))


def _estimate_roi(budget: int, risk: int, time: int, priority: str) -> str:
    low  = max(2, risk * 1.5 + time * 0.3)
    high = low + budget * 0.15 + (10 - risk) * 1.2
    if priority == "growth":
        low  += 3
        high += 6
    elif priority == "stability":
        low  -= 1
        high -= 2
    return f"{low:.0f}–{high:.0f}%"


def _generate_outcomes(
    budget: int, risk: int, time: int, priority: str, domain: str
) -> list[Outcome]:
    high_budget = budget >= 40
    low_risk    = risk <= 4
    long_time   = time >= 20

    # Primary
    if high_budget and low_risk:
        primary = Outcome(1, "Aggressive Expansion",
            "Leverage strong budget headroom with controlled risk appetite "
            "to capture maximum market share. Execute in phased milestones "
            "with quarterly review gates.",
            probability=72, color="green", badge="Recommended")
    elif priority == "stability":
        primary = Outcome(1, "Steady-State Optimisation",
            "Consolidate existing assets and improve operational efficiency "
            "before any expansion. Prioritise cash-flow stability over "
            "short-term gains.",
            probability=68, color="green", badge="Recommended")
    else:
        primary = Outcome(1, "Balanced Growth Strategy",
            "Distribute budget across core operations (60%) and new "
            "initiatives (40%). Maintain a risk buffer equivalent to 15% "
            "of total allocation.",
            probability=65, color="green", badge="Recommended")

    # Alternative
    alt_prob = max(25, primary.probability - 18)
    alternative = Outcome(2, "Pilot-First Approach",
        f"Run a contained {min(time, 8)}-week pilot on the highest-confidence "
        "initiative before committing full budget. Reduces exposure while "
        "generating real-world validation data.",
        probability=alt_prob, color="amber", badge="Alternative")

    # Fallback
    fallback = Outcome(3, "Capital Preservation Mode",
        "Pause discretionary spend and redirect budget to risk mitigation "
        "and compliance. Re-evaluate when external conditions improve or "
        "more data is available.",
        probability=max(10, 100 - primary.probability - alt_prob),
        color="red", badge="Fallback")

    return [primary, alternative, fallback]


def _list_risks(risk: int, budget: int, time: int, domain: str) -> list[str]:
    risks = []
    if risk >= 7:
        risks.append("High volatility exposure — market conditions could shift "
                      "faster than the decision timeline allows.")
    if budget < 20:
        risks.append("Budget constraints may force mid-project scope cuts, "
                      "undermining ROI projections.")
    if time < 8:
        risks.append("Compressed time horizon increases execution risk and "
                      "limits contingency planning.")

    domain_risks = {
        "finance":    "Regulatory or interest-rate changes could affect return assumptions.",
        "healthcare": "Compliance and approval cycles may extend beyond the planned horizon.",
        "technology": "Rapid tech obsolescence could reduce the value of the chosen platform.",
        "business":   "Competitor response may erode first-mover advantage faster than modelled.",
        "education":  "Adoption resistance from stakeholders can delay measurable outcomes.",
        "risk":       "Residual tail risks not captured in the model may materialise simultaneously.",
    }
    if domain in domain_risks:
        risks.append(domain_risks[domain])

    if not risks:
        risks.append("No critical risk flags — maintain standard monitoring cadence.")

    return risks[:3]


def _generate_insight(
    budget: int, risk: int, time: int, priority: str, domain: str
) -> str:
    if risk >= 7 and budget < 25:
        return (f"High risk with a limited ₹{budget}L budget is a dangerous "
                "combination. Consider de-risking at least one variable — "
                "either increase the budget buffer or reduce scope before proceeding.")
    if priority == "growth" and time < 6:
        return ("A growth-oriented priority with fewer than 6 weeks is execution-heavy. "
                "Identify one key metric to hit in week 1 and build momentum from there.")
    if priority == "stability" and risk <= 3:
        return ("Low-risk + stability focus suggests you can afford to be deliberate. "
                "Use this window to document processes and build institutional knowledge.")
    return (f"For a ₹{budget}L {domain} decision over {time} weeks, "
            "allocate 20% of budget as a contingency reserve before committing "
            "to any single outcome strategy.")


def _radar_scores(budget: int, risk: int, time: int, priority: str) -> list[int]:
    feasibility     = min(95, 50 + budget // 2 - risk * 2)
    roi_potential   = min(95, 40 + risk * 4 + (5 if priority == "growth" else 0))
    risk_control    = min(95, 80 - risk * 5 + (5 if priority == "stability" else 0))
    time_efficiency = min(95, 40 + time)
    alignment       = min(95, 60 + (10 if priority in ("growth", "innovation") else 5))

    return [
        max(30, feasibility),
        max(30, roi_potential),
        max(30, risk_control),
        max(30, time_efficiency),
        max(30, alignment),
    ]


def _make_summary(
    domain: str, budget: int, risk: int, time: int, priority: str, risk_level: str
) -> str:
    return (
        f"This {domain} decision involves a ₹{budget}L allocation over {time} weeks "
        f"with a {priority}-focused objective and a {risk_level.lower()} risk profile. "
        f"The analysis identifies three viable paths with distinct risk-return tradeoffs "
        f"suited to risk tolerance level {risk}/10."
    )
"""
data_processor.py
-----------------
Validates and sanitises raw input from the simulation form
before it reaches the decision engine or the Anthropic API.
"""

from __future__ import annotations
import re
from typing import Any


# ── Constants ─────────────────────────────────────────────────────────────────

VALID_DOMAINS = {
    "business",
    "finance",
    "healthcare",
    "education",
    "technology",
    "risk",
}

VALID_PRIORITIES = {"growth", "stability", "efficiency", "innovation"}

BUDGET_MIN, BUDGET_MAX   = 1, 500      # ₹ Lakhs
RISK_MIN,   RISK_MAX     = 1, 10
TIME_MIN,   TIME_MAX     = 1, 104      # weeks


# ── Public API ────────────────────────────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when input fails validation."""


def process_input(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and clean raw form data.

    Parameters
    ----------
    raw : dict
        Keys expected: domain, scenario, budget, risk, time, priority

    Returns
    -------
    dict with cleaned, type-coerced values.

    Raises
    ------
    ValidationError on bad input.
    """
    errors: list[str] = []

    # ── domain ────────────────────────────────────────────────────────────────
    domain = str(raw.get("domain", "")).strip().lower()
    if domain not in VALID_DOMAINS:
        errors.append(f"domain must be one of {sorted(VALID_DOMAINS)}")

    # ── scenario ──────────────────────────────────────────────────────────────
    scenario = str(raw.get("scenario", "")).strip()
    scenario = _sanitise_text(scenario)
    if not scenario:
        scenario = "General strategic decision"
    if len(scenario) > 600:
        scenario = scenario[:600]

    # ── budget ────────────────────────────────────────────────────────────────
    budget = _coerce_int(raw.get("budget"), "budget", errors)
    if budget is not None:
        budget = _clamp(budget, BUDGET_MIN, BUDGET_MAX, "budget", errors)

    # ── risk ──────────────────────────────────────────────────────────────────
    risk = _coerce_int(raw.get("risk"), "risk", errors)
    if risk is not None:
        risk = _clamp(risk, RISK_MIN, RISK_MAX, "risk", errors)

    # ── time ──────────────────────────────────────────────────────────────────
    time_weeks = _coerce_int(raw.get("time"), "time", errors)
    if time_weeks is not None:
        time_weeks = _clamp(time_weeks, TIME_MIN, TIME_MAX, "time", errors)

    # ── priority ──────────────────────────────────────────────────────────────
    priority = str(raw.get("priority", "")).strip().lower()
    if priority not in VALID_PRIORITIES:
        errors.append(f"priority must be one of {sorted(VALID_PRIORITIES)}")

    if errors:
        raise ValidationError("; ".join(errors))

    return {
        "domain":   domain,
        "scenario": scenario,
        "budget":   budget,
        "risk":     risk,
        "time":     time_weeks,
        "priority": priority,
    }


def summarise(params: dict[str, Any]) -> str:
    """Return a one-line human-readable summary of the parameters."""
    return (
        f"{params['domain'].title()} | "
        f"₹{params['budget']}L budget | "
        f"Risk {params['risk']}/10 | "
        f"{params['time']}w | "
        f"{params['priority'].title()} priority"
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sanitise_text(text: str) -> str:
    """Strip HTML tags and control characters from free-text input."""
    text = re.sub(r"<[^>]+>", "", text)          # strip HTML
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)  # control chars
    return text.strip()


def _coerce_int(value: Any, field: str, errors: list[str]) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be an integer")
        return None


def _clamp(value: int, lo: int, hi: int, field: str, errors: list[str]) -> int:
    if value < lo or value > hi:
        errors.append(f"{field} must be between {lo} and {hi}, got {value}")
        return max(lo, min(hi, value))   # clamp anyway so pipeline can continue
    return value
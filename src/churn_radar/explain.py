"""Plain-language risk tiers, drivers, and retention actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from churn_radar.config import HIGH_RISK, MEDIUM_RISK


def risk_tier(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "high"
    if probability >= MEDIUM_RISK:
        return "medium"
    return "low"


def risk_drivers(customer: Mapping[str, Any]) -> list[str]:
    """Flag the attributes that the EDA showed carry the most churn risk."""
    drivers = []
    if customer.get("Contract") == "Month-to-month":
        drivers.append("month-to-month contract (top churn driver)")
    if int(customer.get("tenure", 0)) <= 6:
        drivers.append("very new customer (6 months tenure or less)")
    if customer.get("InternetService") == "Fiber optic":
        drivers.append("fiber-optic plan, a price-sensitive segment")
    if customer.get("PaymentMethod") == "Electronic check":
        drivers.append("pays by electronic check (historically churn-prone)")
    if customer.get("TechSupport") in ("No", "No internet service"):
        drivers.append("no tech-support add-on")
    return drivers


def recommended_action(probability: float) -> str:
    tier = risk_tier(probability)
    if tier == "high":
        return "Call within 48 hours: offer a 1-year contract upgrade with a loyalty discount"
    if tier == "medium":
        return "Add to the retention email campaign with a contract-upgrade offer"
    return "Standard engagement, no action needed"

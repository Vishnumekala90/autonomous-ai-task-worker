"""Verification: compare what we expected to write against what the company system holds.

Used both by the check_company_record tool and by the graph's independent verify_result
node, so there is exactly one definition of "matches".
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.database.database import Database
from app.database.repository import get_company_record

REQUIRED_FIELDS = ("invoice_id", "company", "amount", "due_date")
AMOUNT_TOLERANCE = 0.005


class FieldCheck(BaseModel):
    field: str
    expected: Any
    actual: Any
    match: bool


class VerificationResult(BaseModel):
    passed: bool
    checks: list[FieldCheck]
    reason: str
    record_id: str | None = None


def _values_match(field: str, expected: Any, actual: Any) -> bool:
    if actual is None or expected is None:
        return False
    if field == "amount":
        return abs(float(expected) - float(actual)) <= AMOUNT_TOLERANCE
    return str(expected).strip().casefold() == str(actual).strip().casefold()


def compare_fields(expected: dict[str, Any], actual: dict[str, Any] | None) -> VerificationResult:
    if actual is None:
        return VerificationResult(
            passed=False,
            checks=[],
            reason=f"No company record exists for {expected.get('invoice_id', 'the invoice')}.",
        )
    checks = [
        FieldCheck(
            field=f,
            expected=expected.get(f),
            actual=actual.get(f),
            match=_values_match(f, expected.get(f), actual.get(f)),
        )
        for f in REQUIRED_FIELDS
        if f in expected
    ]
    mismatched = [c.field for c in checks if not c.match]
    passed = bool(checks) and not mismatched
    reason = "All required fields match." if passed else (
        "Mismatch in: " + ", ".join(mismatched) if mismatched else "Nothing to compare."
    )
    return VerificationResult(
        passed=passed, checks=checks, reason=reason, record_id=actual.get("record_id")
    )


def verify_record(db: Database, expected: dict[str, Any]) -> VerificationResult:
    """Read the internal system fresh and compare against the expected values."""
    return compare_fields(expected, get_company_record(db, expected["invoice_id"]))

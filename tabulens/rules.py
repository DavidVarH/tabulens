"""Declarative validation rules for pandas DataFrames (definition only; no execution)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


def _validate_column_name(column: str) -> str:
    if not isinstance(column, str) or not column.strip():
        raise ValueError("column must be a non-empty string")
    return column.strip()


@dataclass(frozen=True)
class ValidationRule:
    """Base record for a single validation constraint on one column."""

    rule_name: str
    column: str


@dataclass(frozen=True)
class NotNullRule(ValidationRule):
    """Require non-null values in ``column``."""

    pass


@dataclass(frozen=True)
class UniqueRule(ValidationRule):
    """Require distinct non-null values in ``column``."""

    pass


@dataclass(frozen=True)
class RangeRule(ValidationRule):
    """Require numeric values to fall within an inclusive bounds window."""

    min_value: float | None
    max_value: float | None


@dataclass(frozen=True)
class AllowedValuesRule(ValidationRule):
    """Require values to be members of ``allowed_values``."""

    allowed_values: tuple[Any, ...]


@dataclass(frozen=True)
class RegexRule(ValidationRule):
    """Require string values to match ``pattern`` (Python ``re`` semantics)."""

    pattern: str


class RuleSet:
    """
    Builder for a list of :class:`ValidationRule` instances.

    Methods return ``self`` so calls can be chained. Rules are stored in
    definition order for later evaluation by a validation engine.
    """

    __slots__ = ("_rules",)

    def __init__(self) -> None:
        self._rules: list[ValidationRule] = []

    def not_null(self, column: str) -> RuleSet:
        """Append a :class:`NotNullRule` for ``column``."""
        col = _validate_column_name(column)
        self._rules.append(
            NotNullRule(rule_name="not_null", column=col),
        )
        return self

    def unique(self, column: str) -> RuleSet:
        """Append a :class:`UniqueRule` for ``column``."""
        col = _validate_column_name(column)
        self._rules.append(
            UniqueRule(rule_name="unique", column=col),
        )
        return self

    def in_range(
        self,
        column: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> RuleSet:
        """Append a :class:`RangeRule`; at least one bound must be provided."""
        col = _validate_column_name(column)
        if min_value is None and max_value is None:
            raise ValueError("in_range requires at least one of min_value or max_value")
        self._rules.append(
            RangeRule(
                rule_name="in_range",
                column=col,
                min_value=min_value,
                max_value=max_value,
            ),
        )
        return self

    def allowed_values(self, column: str, values: list[Any]) -> RuleSet:
        """Append an :class:`AllowedValuesRule` using the given allowed set."""
        col = _validate_column_name(column)
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError("allowed_values requires a non-empty list")
        self._rules.append(
            AllowedValuesRule(
                rule_name="allowed_values",
                column=col,
                allowed_values=tuple(values),
            ),
        )
        return self

    def regex(self, column: str, pattern: str) -> RuleSet:
        """Append a :class:`RegexRule` with the given regex ``pattern``."""
        col = _validate_column_name(column)
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("regex requires a non-empty pattern string")
        self._rules.append(
            RegexRule(rule_name="regex", column=col, pattern=pattern),
        )
        return self

    def to_list(self) -> list[ValidationRule]:
        """Return a shallow copy of the configured rules."""
        return list(self._rules)

    def summary(self) -> dict[str, Any]:
        """Counts and totals for quick inspection or logging."""
        counts = Counter(r.rule_name for r in self._rules)
        return {
            "total_rules": len(self._rules),
            "counts_by_rule_type": dict(sorted(counts.items())),
        }


__all__ = [
    "AllowedValuesRule",
    "NotNullRule",
    "RangeRule",
    "RegexRule",
    "RuleSet",
    "UniqueRule",
    "ValidationRule",
]

"""Rule evaluation engine for Tabulens RuleSet objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from tabulens.rules import (
    AllowedValuesRule,
    NotNullRule,
    RangeRule,
    RegexRule,
    RuleSet,
    UniqueRule,
    ValidationRule,
)


@dataclass(frozen=True)
class RuleResult:
    rule_name: str
    column: str
    status: str  # "passed", "failed", "near_failing"
    affected_rows: int
    row_indices: list[Any]
    message: str


@dataclass(frozen=True)
class ValidationReport:
    results: list[RuleResult]

    def summary(self) -> dict[str, Any]:
        return {
            "total_rules": len(self.results),
            "passed": len(self.passed()),
            "failed": len(self.failed()),
            "near_failing": len(self.near_failing()),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def passed(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == "passed"]

    def failed(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == "failed"]

    def near_failing(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == "near_failing"]

    def passed_indices(self) -> dict[str, list[Any]]:
        return {f"{r.rule_name}:{r.column}": list(r.row_indices) for r in self.passed()}

    def failed_indices(self) -> dict[str, list[Any]]:
        return {f"{r.rule_name}:{r.column}": list(r.row_indices) for r in self.failed()}

    def near_failing_indices(self) -> dict[str, list[Any]]:
        return {
            f"{r.rule_name}:{r.column}": list(r.row_indices)
            for r in self.near_failing()
        }

    def render_text(self) -> str:
        lines: list[str] = [
            "VALIDATION REPORT",
            "-" * 17,
            "",
        ]

        def add_block(title: str, items: list[RuleResult]) -> None:
            lines.append(f"{title}:")
            if not items:
                lines.append("(none)")
            else:
                for r in items:
                    lines.append(
                        f"- {r.rule_name}({r.column}): {r.status} "
                        f"(affected_rows={r.affected_rows}) "
                        f"indices={_format_row_indices(r.row_indices)} — {r.message}"
                    )
            lines.append("")

        add_block("PASSED RULES", self.passed())
        add_block("FAILED RULES", self.failed())
        add_block("NEAR FAILING", self.near_failing())

        s = self.summary()
        lines.extend(
            [
                "SUMMARY:",
                f"- total_rules: {s['total_rules']}",
                f"- passed: {s['passed']}",
                f"- failed: {s['failed']}",
                f"- near_failing: {s['near_failing']}",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"


def _validate_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
    return df


def _validate_ruleset(rules: Any) -> RuleSet:
    if not isinstance(rules, RuleSet):
        raise TypeError(f"expected RuleSet, got {type(rules).__name__}")
    return rules


def _status_from_failure_rate(failure_rate: float) -> str:
    if failure_rate <= 0.0:
        return "passed"
    if failure_rate <= 0.10:
        return "near_failing"
    return "failed"


def _failure_rate(affected_rows: int, row_count: int) -> float:
    if row_count <= 0:
        return 0.0 if affected_rows == 0 else 1.0
    return affected_rows / row_count


def _missing_column_result(rule: ValidationRule, df: pd.DataFrame) -> RuleResult:
    row_indices = df.index.tolist()
    return RuleResult(
        rule_name=rule.rule_name,
        column=rule.column,
        status="failed",
        affected_rows=len(row_indices),
        row_indices=row_indices,
        message=f"Column '{rule.column}' is missing from the DataFrame.",
    )


def _result(
    rule: ValidationRule, row_indices: list[Any], row_count: int, message: str
) -> RuleResult:
    affected_rows = len(row_indices)
    status = _status_from_failure_rate(_failure_rate(affected_rows, row_count))
    return RuleResult(
        rule_name=rule.rule_name,
        column=rule.column,
        status=status,
        affected_rows=affected_rows,
        row_indices=list(row_indices),
        message=message,
    )


def _mask_to_row_indices(mask: pd.Series) -> list[Any]:
    """Convert a boolean mask aligned to the DataFrame into original row indices."""
    return mask.index[mask].tolist()


def _format_row_indices(indices: list[Any], *, head: int = 6, tail: int = 3) -> str:
    if not indices:
        return "[]"
    if len(indices) <= (head + tail + 1):
        return "[" + ", ".join(str(i) for i in indices) + "]"
    start = ", ".join(str(i) for i in indices[:head])
    end = ", ".join(str(i) for i in indices[-tail:])
    return f"[{start}, ..., {end}]"


def _eval_not_null(series: pd.Series, rule: NotNullRule, row_count: int) -> RuleResult:
    mask = series.isna()
    idx = _mask_to_row_indices(mask)
    if not idx:
        return _result(
            rule, [], row_count, f"Column '{rule.column}' has no null values."
        )
    return _result(
        rule, idx, row_count, f"Column '{rule.column}' has {len(idx)} null values."
    )


def _eval_unique(series: pd.Series, rule: UniqueRule, row_count: int) -> RuleResult:
    mask = series.notna() & series.duplicated(keep=False)
    idx = _mask_to_row_indices(mask)
    if not idx:
        return _result(
            rule, [], row_count, f"Column '{rule.column}' has no duplicate values."
        )
    return _result(
        rule,
        idx,
        row_count,
        f"Column '{rule.column}' has {len(idx)} duplicated value(s) (nulls ignored).",
    )


def _eval_range(series: pd.Series, rule: RangeRule, row_count: int) -> RuleResult:
    non_null = series.notna()
    if not bool(non_null.any()):
        return _result(
            rule,
            [],
            row_count,
            f"Column '{rule.column}' has no non-null values to check.",
        )

    numeric = pd.to_numeric(series, errors="coerce")
    bad_non_numeric = non_null & numeric.isna()

    below = pd.Series(False, index=series.index)
    above = pd.Series(False, index=series.index)

    if rule.min_value is not None:
        below = non_null & (numeric < rule.min_value)
    if rule.max_value is not None:
        above = non_null & (numeric > rule.max_value)

    mask = bad_non_numeric | below | above
    idx = _mask_to_row_indices(mask)

    if not idx:
        bounds = []
        if rule.min_value is not None:
            bounds.append(f">= {rule.min_value}")
        if rule.max_value is not None:
            bounds.append(f"<= {rule.max_value}")
        window = " and ".join(bounds) if bounds else "within range"
        return _result(
            rule, [], row_count, f"Column '{rule.column}' values are {window}."
        )

    return _result(
        rule,
        idx,
        row_count,
        f"Column '{rule.column}' has {len(idx)} value(s) outside the allowed range or non-numeric.",
    )


def _eval_allowed_values(
    series: pd.Series, rule: AllowedValuesRule, row_count: int
) -> RuleResult:
    non_null = series.notna()
    if not bool(non_null.any()):
        return _result(
            rule,
            [],
            row_count,
            f"Column '{rule.column}' has no non-null values to check.",
        )

    allowed = set(rule.allowed_values)
    mask = non_null & (~series.isin(allowed))
    idx = _mask_to_row_indices(mask)

    if not idx:
        return _result(
            rule, [], row_count, f"Column '{rule.column}' contains only allowed values."
        )

    return _result(
        rule,
        idx,
        row_count,
        f"Column '{rule.column}' has {len(idx)} disallowed value(s).",
    )


def _eval_regex(series: pd.Series, rule: RegexRule, row_count: int) -> RuleResult:
    non_null = series.notna()
    if not bool(non_null.any()):
        return _result(
            rule,
            [],
            row_count,
            f"Column '{rule.column}' has no non-null values to check.",
        )

    matches = series.astype(str).str.match(rule.pattern, na=False)
    mask = non_null & (~matches)
    idx = _mask_to_row_indices(mask)

    if not idx:
        return _result(
            rule, [], row_count, f"Column '{rule.column}' matches the expected format."
        )

    return _result(
        rule,
        idx,
        row_count,
        f"Column '{rule.column}' contains invalid format values.",
    )


def validate_dataframe(df: pd.DataFrame, rules: RuleSet) -> ValidationReport:
    """
    Evaluate a RuleSet against df.

    Failure-rate heuristic:
    - 0% failures -> passed
    - >0% and <=10% failures -> near_failing
    - >10% failures -> failed
    """
    df = _validate_dataframe(df)
    rules = _validate_ruleset(rules)

    row_count = len(df)
    results: list[RuleResult] = []

    for rule in rules.to_list():
        if rule.column not in df.columns:
            results.append(_missing_column_result(rule, df))
            continue

        series = df[rule.column]

        if isinstance(rule, NotNullRule):
            results.append(_eval_not_null(series, rule, row_count))
        elif isinstance(rule, UniqueRule):
            results.append(_eval_unique(series, rule, row_count))
        elif isinstance(rule, RangeRule):
            results.append(_eval_range(series, rule, row_count))
        elif isinstance(rule, AllowedValuesRule):
            results.append(_eval_allowed_values(series, rule, row_count))
        elif isinstance(rule, RegexRule):
            results.append(_eval_regex(series, rule, row_count))
        else:
            results.append(
                RuleResult(
                    rule_name=rule.rule_name,
                    column=rule.column,
                    status="failed",
                    affected_rows=len(df.index),
                    row_indices=df.index.tolist(),
                    message=f"Unsupported rule type: {type(rule).__name__}",
                )
            )

    return ValidationReport(results=results)


__all__ = [
    "RuleResult",
    "ValidationReport",
    "validate_dataframe",
]

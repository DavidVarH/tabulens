"""Selective application of structural recommendations from profiling."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tabulens.profiling import ProfileReport, StructuralRecommendation

_REVIEW_ONLY_RULE_IDS: frozenset[str] = frozenset(
    {
        "review_identifier_column",
        "review_near_constant_column",
    }
)


def _validate_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
    return df


def _validate_profile(profile: Any) -> ProfileReport:
    if not isinstance(profile, ProfileReport):
        raise TypeError(f"expected ProfileReport, got {type(profile).__name__}")
    return profile


def _dataframe_memory_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(deep=True, index=False).sum())


def _resolve_column_label(df: pd.DataFrame, column_key: str) -> Any | None:
    for col in df.columns:
        if str(col) == column_key:
            return col
    return None


def _selection_filters(
    selected_rule_ids: list[str] | None,
    selected_columns: list[str] | None,
) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    rules = frozenset(selected_rule_ids) if selected_rule_ids is not None else None
    cols = frozenset(selected_columns) if selected_columns is not None else None
    return rules, cols


def _matches_selection(
    rec: StructuralRecommendation,
    rule_filter: frozenset[str] | None,
    col_filter: frozenset[str] | None,
) -> bool:
    """True when this recommendation is eligible under the given filters."""
    if rule_filter is None and col_filter is None:
        return False
    if rule_filter is not None and rec.rule_id not in rule_filter:
        return False
    if col_filter is not None and rec.column not in col_filter:
        return False
    return True


def _apply_drop_constant(working: pd.DataFrame, col_label: Any) -> AppliedOptimization:
    working.drop(columns=[col_label], inplace=True)
    return AppliedOptimization(
        rule_id="drop_constant_column",
        column=str(col_label),
        action="drop_column",
        success=True,
        message=f'Column "{col_label}" was dropped.',
    )


def _apply_convert_category(working: pd.DataFrame, col_label: Any) -> AppliedOptimization:
    working[col_label] = working[col_label].astype("category")
    return AppliedOptimization(
        rule_id="convert_to_category",
        column=str(col_label),
        action="convert_to_category",
        success=True,
        message=f'Column "{col_label}" was cast to pandas category dtype.',
    )


def _apply_convert_datetime(working: pd.DataFrame, col_label: Any) -> AppliedOptimization:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        working[col_label] = pd.to_datetime(working[col_label], errors="coerce")
    return AppliedOptimization(
        rule_id="convert_to_datetime",
        column=str(col_label),
        action="convert_to_datetime",
        success=True,
        message=f'Column "{col_label}" was parsed to datetime (invalid values coerced to NaT).',
    )


@dataclass(frozen=True)
class AppliedOptimization:
    rule_id: str
    column: str
    action: str
    success: bool
    message: str


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of running ``optimize_dataframe`` on a copy of the input frame."""

    optimized_df: pd.DataFrame
    applied_optimizations: list[AppliedOptimization]
    skipped_optimizations: list[AppliedOptimization]
    memory_before_bytes: int
    memory_after_bytes: int
    memory_saved_bytes: int

    def summary(self) -> dict[str, Any]:
        """Compact counts and memory deltas for logging or APIs."""
        return {
            "applied_count": len(self.applied_optimizations),
            "skipped_count": len(self.skipped_optimizations),
            "memory_before_bytes": self.memory_before_bytes,
            "memory_after_bytes": self.memory_after_bytes,
            "memory_saved_bytes": self.memory_saved_bytes,
        }


def optimize_dataframe(
    df: pd.DataFrame,
    profile: ProfileReport,
    selected_rule_ids: list[str] | None = None,
    selected_columns: list[str] | None = None,
) -> OptimizationResult:
    """
    Apply a subset of ``profile.structural_recommendations`` to a **copy** of ``df``.

    By default (both ``selected_rule_ids`` and ``selected_columns`` are ``None``),
    nothing is applied. Review-style recommendations are never transformed here,
    even when explicitly selected; they are recorded as skipped with an
    explanatory message.
    """
    df = _validate_dataframe(df)
    profile = _validate_profile(profile)

    working = df.copy()
    memory_before_bytes = _dataframe_memory_bytes(working)

    rule_filter, col_filter = _selection_filters(selected_rule_ids, selected_columns)
    applied: list[AppliedOptimization] = []
    skipped: list[AppliedOptimization] = []

    for rec in profile.structural_recommendations:
        if not _matches_selection(rec, rule_filter, col_filter):
            skipped.append(
                AppliedOptimization(
                    rule_id=rec.rule_id,
                    column=rec.column,
                    action=rec.suggested_action,
                    success=False,
                    message=_skip_message_not_selected(rec, rule_filter, col_filter),
                )
            )
            continue

        if rec.rule_id in _REVIEW_ONLY_RULE_IDS:
            skipped.append(
                AppliedOptimization(
                    rule_id=rec.rule_id,
                    column=rec.column,
                    action=rec.suggested_action,
                    success=False,
                    message=(
                        "Review-only recommendation: not applied automatically; "
                        "inspect the column in context before changing or removing it."
                    ),
                )
            )
            continue

        col_label = _resolve_column_label(working, rec.column)
        if col_label is None:
            skipped.append(
                AppliedOptimization(
                    rule_id=rec.rule_id,
                    column=rec.column,
                    action=rec.suggested_action,
                    success=False,
                    message=(
                        f'Column "{rec.column}" is not present in the working DataFrame; '
                        "skipping this recommendation."
                    ),
                )
            )
            continue

        try:
            if rec.rule_id == "drop_constant_column":
                applied.append(_apply_drop_constant(working, col_label))
            elif rec.rule_id == "convert_to_category":
                applied.append(_apply_convert_category(working, col_label))
            elif rec.rule_id == "convert_to_datetime":
                applied.append(_apply_convert_datetime(working, col_label))
            else:
                skipped.append(
                    AppliedOptimization(
                        rule_id=rec.rule_id,
                        column=rec.column,
                        action=rec.suggested_action,
                        success=False,
                        message=(
                            f'No automated handler is implemented for rule_id "{rec.rule_id}".'
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001 — surface as skipped, not crash
            skipped.append(
                AppliedOptimization(
                    rule_id=rec.rule_id,
                    column=rec.column,
                    action=rec.suggested_action,
                    success=False,
                    message=f"Could not apply transformation safely: {exc}",
                )
            )

    memory_after_bytes = _dataframe_memory_bytes(working)
    memory_saved_bytes = memory_before_bytes - memory_after_bytes

    return OptimizationResult(
        optimized_df=working,
        applied_optimizations=applied,
        skipped_optimizations=skipped,
        memory_before_bytes=memory_before_bytes,
        memory_after_bytes=memory_after_bytes,
        memory_saved_bytes=memory_saved_bytes,
    )


def _skip_message_not_selected(
    rec: StructuralRecommendation,
    rule_filter: frozenset[str] | None,
    col_filter: frozenset[str] | None,
) -> str:
    if rule_filter is None and col_filter is None:
        return (
            "No selection criteria were provided (``selected_rule_ids`` and "
            "``selected_columns`` are both unset); structural optimizations are not applied."
        )
    parts: list[str] = []
    if rule_filter is not None and rec.rule_id not in rule_filter:
        parts.append(f'rule_id "{rec.rule_id}" is not in the active rule filter')
    if col_filter is not None and rec.column not in col_filter:
        parts.append(f'column "{rec.column}" is not in the active column filter')
    detail = "; ".join(parts) if parts else "filters did not match this recommendation"
    return f"Skipped — not selected ({detail})."


__all__ = [
    "AppliedOptimization",
    "OptimizationResult",
    "optimize_dataframe",
]

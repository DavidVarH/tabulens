"""DataFrame profiling for Tabulens — structured reports, no side effects."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

# Minimum non-null count before a column can be flagged as a suspected identifier.
_MIN_NON_NULL_FOR_IDENTIFIER: int = 20


def _col_key(name: Any) -> str:
    return str(name)


def _validate_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
    return df


def _dtypes_map(df: pd.DataFrame) -> dict[str, str]:
    return {_col_key(c): str(df[c].dtype) for c in df.columns}


def _missing_stats(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    counts: dict[str, int] = {}
    percents: dict[str, float] = {}
    n = len(df)
    for c in df.columns:
        key = _col_key(c)
        miss = int(df[c].isna().sum())
        counts[key] = miss
        percents[key] = (miss / n * 100.0) if n else 0.0
    return counts, percents


def _unique_counts(df: pd.DataFrame) -> dict[str, int]:
    return {_col_key(c): int(df[c].nunique(dropna=False)) for c in df.columns}


def _is_category_like_dtype(dtype: Any) -> bool:
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _is_object_or_string_dtype(dtype: Any) -> bool:
    return bool(
        pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
    )


def _constant_and_near_constant(
    df: pd.DataFrame, near_constant_threshold: float
) -> tuple[list[str], list[str]]:
    constant: list[str] = []
    near_constant: list[str] = []
    for c in df.columns:
        key = _col_key(c)
        s = df[c]
        n_distinct = int(s.nunique(dropna=False))

        if n_distinct <= 1:
            constant.append(key)
            continue

        non_null = int(s.notna().sum())
        if non_null == 0:
            continue

        vc = s.value_counts(dropna=True)
        top_freq = int(vc.iloc[0]) if len(vc) else 0
        ratio = top_freq / non_null
        if ratio >= near_constant_threshold:
            near_constant.append(key)

    return constant, near_constant


def _suspected_identifier_columns(
    df: pd.DataFrame,
    *,
    row_count: int,
    uniqueness_threshold: float,
    category_ratio_threshold: float,
) -> list[str]:
    """High uniqueness among non-null values; skips clearly low-cardinality columns."""
    out: list[str] = []
    for c in df.columns:
        key = _col_key(c)
        s = df[c]
        non_null = int(s.notna().sum())
        if (
            non_null < _MIN_NON_NULL_FOR_IDENTIFIER
            and row_count >= _MIN_NON_NULL_FOR_IDENTIFIER
        ):
            continue
        if non_null < 2:
            continue

        uniq = int(s.nunique(dropna=True))
        if uniq < 2:
            continue

        ratio_non_null = uniq / non_null
        if ratio_non_null < uniqueness_threshold:
            continue

        # Dense columns with few distinct levels relative to table size are categorical, not IDs.
        fill = non_null / row_count if row_count else 0.0
        if (
            fill >= 0.8
            and row_count > 0
            and (uniq / row_count) <= category_ratio_threshold
        ):
            continue

        out.append(key)
    return out


def _suspected_datetime_columns(
    df: pd.DataFrame,
    *,
    sample_size: int,
    parse_threshold: float,
) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if not _is_object_or_string_dtype(df[c].dtype):
            continue
        key = _col_key(c)
        s = df[c]
        non_null = s.dropna()
        if non_null.empty:
            continue

        n_take = min(int(sample_size), int(non_null.shape[0]))
        sample = (
            non_null.sample(n=n_take, random_state=42)
            if len(non_null) > n_take
            else non_null
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        ok = int(parsed.notna().sum())
        if ok / max(len(sample), 1) >= parse_threshold:
            out.append(key)
    return out


def _suggested_categorical_columns(
    df: pd.DataFrame,
    *,
    row_count: int,
    category_ratio_threshold: float,
    identifier_keys: frozenset[str],
    datetime_keys: frozenset[str],
    constant_keys: frozenset[str],
    near_constant_keys: frozenset[str],
) -> list[str]:
    if row_count <= 0:
        return []

    out: list[str] = []
    for c in df.columns:
        key = _col_key(c)
        if key in identifier_keys or key in datetime_keys:
            continue
        if key in constant_keys or key in near_constant_keys:
            continue

        if not _is_category_like_dtype(df[c].dtype):
            continue

        uniq = int(df[c].nunique(dropna=False))
        if (uniq / row_count) > category_ratio_threshold:
            continue

        out.append(key)
    return out


def _memory_usage(df: pd.DataFrame) -> tuple[int, dict[str, int]]:
    usage = df.memory_usage(deep=True, index=False)
    by_col = {str(k): int(v) for k, v in usage.items()}
    total = int(usage.sum())
    return total, by_col


def _validate_profile_params(
    near_constant_threshold: float,
    category_ratio_threshold: float,
    identifier_uniqueness_threshold: float,
    datetime_sample_size: int,
    datetime_parse_threshold: float,
) -> None:
    for name, value, lo, hi in (
        ("near_constant_threshold", near_constant_threshold, 0.0, 1.0),
        ("category_ratio_threshold", category_ratio_threshold, 0.0, 1.0),
        (
            "identifier_uniqueness_threshold",
            identifier_uniqueness_threshold,
            0.0,
            1.0,
        ),
        ("datetime_parse_threshold", datetime_parse_threshold, 0.0, 1.0),
    ):
        if not (lo < value <= hi):
            raise ValueError(f"{name} must be in ({lo}, {hi}]")
    if datetime_sample_size < 1:
        raise ValueError("datetime_sample_size must be >= 1")


@dataclass(frozen=True)
class StructuralRecommendation:
    rule_id: str
    column: str
    message: str
    suggested_action: str


def _build_structural_recommendations(
    *,
    row_count: int,
    unique_counts: dict[str, int],
    constant_columns: list[str],
    near_constant_columns: list[str],
    suggested_categorical: list[str],
    suspected_datetime_columns: list[str],
    suspected_identifier_columns: list[str],
    near_constant_threshold: float,
) -> list[StructuralRecommendation]:
    """Turn detected structural signals into actionable, machine-readable hints."""
    out: list[StructuralRecommendation] = []

    for col in constant_columns:
        out.append(
            StructuralRecommendation(
                rule_id="drop_constant_column",
                column=col,
                message=(
                    f'Column "{col}" is constant and may add little analytical value.'
                ),
                suggested_action="drop_column",
            )
        )

    for col in near_constant_columns:
        out.append(
            StructuralRecommendation(
                rule_id="review_near_constant_column",
                column=col,
                message=(
                    f'Column "{col}" is near-constant: one value accounts for at least '
                    f"{near_constant_threshold:.0%} of non-null rows; review before modeling."
                ),
                suggested_action="review_column",
            )
        )

    for col in suggested_categorical:
        uniq = int(unique_counts.get(col, 0))
        out.append(
            StructuralRecommendation(
                rule_id="convert_to_category",
                column=col,
                message=(
                    f'Column "{col}" has low relative cardinality '
                    f"({uniq} distinct value(s) across {row_count} row(s)) "
                    f"and may benefit from category dtype."
                ),
                suggested_action="convert_to_category",
            )
        )

    for col in suspected_datetime_columns:
        out.append(
            StructuralRecommendation(
                rule_id="convert_to_datetime",
                column=col,
                message=(
                    f'Column "{col}" appears to contain datetime-like text values.'
                ),
                suggested_action="convert_to_datetime",
            )
        )

    for col in suspected_identifier_columns:
        out.append(
            StructuralRecommendation(
                rule_id="review_identifier_column",
                column=col,
                message=(
                    f'Column "{col}" appears to behave like an identifier and may not be '
                    f"analytically useful as a feature."
                ),
                suggested_action="review_identifier",
            )
        )

    return out


@dataclass(frozen=True)
class ProfileReport:
    row_count: int
    column_count: int
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    missing_percentages: dict[str, float]
    duplicate_count: int
    unique_counts: dict[str, int]
    memory_usage_bytes: int
    memory_usage_by_column: dict[str, int]
    constant_columns: list[str]
    near_constant_columns: list[str]
    suggested_categorical: list[str]
    suspected_identifier_columns: list[str]
    suspected_datetime_columns: list[str]
    structural_recommendations: list[StructuralRecommendation]

    def to_dict(self) -> dict[str, Any]:
        """Full report as plain built-in types (JSON-serializable keys)."""
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        """Compact roll-up for quick inspection or APIs."""
        total_missing = sum(self.missing_counts.values())
        cells = self.row_count * self.column_count
        return {
            "shape": (self.row_count, self.column_count),
            "total_missing_cells": total_missing,
            "overall_missing_fraction": (total_missing / cells) if cells else 0.0,
            "columns_with_any_missing": sum(
                1 for v in self.missing_counts.values() if v
            ),
            "duplicate_rows": self.duplicate_count,
            "constant_columns": list(self.constant_columns),
            "near_constant_columns": list(self.near_constant_columns),
            "approx_memory_bytes": self.memory_usage_bytes,
            "suggested_categorical_count": len(self.suggested_categorical),
            "suspected_identifier_columns_count": len(
                self.suspected_identifier_columns
            ),
            "suspected_datetime_columns_count": len(self.suspected_datetime_columns),
            "structural_recommendations_count": len(self.structural_recommendations),
        }


def profile_dataframe(
    df: pd.DataFrame,
    near_constant_threshold: float = 0.95,
    category_ratio_threshold: float = 0.5,
    identifier_uniqueness_threshold: float = 0.95,
    datetime_sample_size: int = 50,
    datetime_parse_threshold: float = 0.8,
) -> ProfileReport:
    """
    Build a structured profile of ``df``.

    Near-constant columns use ``near_constant_threshold`` on the share of the
    dominant non-null value. Categorical, identifier, and datetime hints are
    conservative heuristics for downstream review.
    """
    df = _validate_dataframe(df)
    _validate_profile_params(
        near_constant_threshold,
        category_ratio_threshold,
        identifier_uniqueness_threshold,
        datetime_sample_size,
        datetime_parse_threshold,
    )

    row_count = len(df)
    column_count = df.shape[1]
    dtypes = _dtypes_map(df)
    missing_counts, missing_percentages = _missing_stats(df)
    duplicate_count = int(df.duplicated().sum())
    unique_counts = _unique_counts(df)
    memory_usage_bytes, memory_usage_by_column = _memory_usage(df)
    constant_columns, near_constant_columns = _constant_and_near_constant(
        df, near_constant_threshold
    )

    suspected_identifier_columns = _suspected_identifier_columns(
        df,
        row_count=row_count,
        uniqueness_threshold=identifier_uniqueness_threshold,
        category_ratio_threshold=category_ratio_threshold,
    )
    suspected_datetime_columns = _suspected_datetime_columns(
        df,
        sample_size=datetime_sample_size,
        parse_threshold=datetime_parse_threshold,
    )
    suggested_categorical = _suggested_categorical_columns(
        df,
        row_count=row_count,
        category_ratio_threshold=category_ratio_threshold,
        identifier_keys=frozenset(suspected_identifier_columns),
        datetime_keys=frozenset(suspected_datetime_columns),
        constant_keys=frozenset(constant_columns),
        near_constant_keys=frozenset(near_constant_columns),
    )

    structural_recommendations = _build_structural_recommendations(
        row_count=row_count,
        unique_counts=unique_counts,
        constant_columns=constant_columns,
        near_constant_columns=near_constant_columns,
        suggested_categorical=suggested_categorical,
        suspected_datetime_columns=suspected_datetime_columns,
        suspected_identifier_columns=suspected_identifier_columns,
        near_constant_threshold=near_constant_threshold,
    )

    return ProfileReport(
        row_count=row_count,
        column_count=column_count,
        dtypes=dtypes,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        duplicate_count=duplicate_count,
        unique_counts=unique_counts,
        memory_usage_bytes=memory_usage_bytes,
        memory_usage_by_column=memory_usage_by_column,
        constant_columns=constant_columns,
        near_constant_columns=near_constant_columns,
        suggested_categorical=suggested_categorical,
        suspected_identifier_columns=suspected_identifier_columns,
        suspected_datetime_columns=suspected_datetime_columns,
        structural_recommendations=structural_recommendations,
    )


__all__ = [
    "ProfileReport",
    "StructuralRecommendation",
    "profile_dataframe",
]

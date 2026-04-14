"""Filtering helpers built from validation rule outcomes."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tabulens.validation import ValidationReport


def _validate_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
    return df


def _validate_validation_report(validation: Any) -> ValidationReport:
    if not isinstance(validation, ValidationReport):
        raise TypeError(f"expected ValidationReport, got {type(validation).__name__}")
    return validation


def _collect_invalid_indices(
    validation: ValidationReport,
    *,
    include_near_failing: bool,
) -> set[Any]:
    """
    Collect index labels considered invalid from rule results.

    - failed rows are always invalid
    - near_failing rows are invalid only when include_near_failing=True
    - passed rows are always valid
    """
    invalid: set[Any] = set()
    for result in validation.results:
        if result.status == "failed":
            invalid.update(result.row_indices)
        elif include_near_failing and result.status == "near_failing":
            invalid.update(result.row_indices)
    return invalid


def keep_valid_rows(
    df: pd.DataFrame,
    validation: ValidationReport,
    include_near_failing: bool = False,
) -> pd.DataFrame:
    """Return a copy of rows considered valid under the validation report."""
    df = _validate_dataframe(df)
    validation = _validate_validation_report(validation)
    invalid_idx = _collect_invalid_indices(
        validation, include_near_failing=include_near_failing
    )
    mask = ~df.index.isin(invalid_idx)
    return df.loc[mask].copy()


def keep_invalid_rows(
    df: pd.DataFrame,
    validation: ValidationReport,
    include_near_failing: bool = False,
) -> pd.DataFrame:
    """Return a copy of rows considered invalid under the validation report."""
    df = _validate_dataframe(df)
    validation = _validate_validation_report(validation)
    invalid_idx = _collect_invalid_indices(
        validation, include_near_failing=include_near_failing
    )
    mask = df.index.isin(invalid_idx)
    return df.loc[mask].copy()


def split_valid_invalid_rows(
    df: pd.DataFrame,
    validation: ValidationReport,
    include_near_failing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(valid_df, invalid_df)`` using the same filtering criteria."""
    df = _validate_dataframe(df)
    validation = _validate_validation_report(validation)
    invalid_idx = _collect_invalid_indices(
        validation, include_near_failing=include_near_failing
    )
    invalid_mask = df.index.isin(invalid_idx)
    valid_df = df.loc[~invalid_mask].copy()
    invalid_df = df.loc[invalid_mask].copy()
    return valid_df, invalid_df


__all__ = [
    "keep_valid_rows",
    "keep_invalid_rows",
    "split_valid_invalid_rows",
]


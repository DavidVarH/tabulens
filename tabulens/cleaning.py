"""Configurable null and duplicate preprocessing for DataFrames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CleaningAction:
    action_type: str
    column: str | None
    strategy: str
    affected_rows: int
    message: str


@dataclass(frozen=True)
class CleaningResult:
    cleaned_df: pd.DataFrame
    actions: list[CleaningAction]
    rows_before: int
    rows_after: int

    def summary(self) -> dict[str, Any]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": self.rows_before - self.rows_after,
            "total_actions": len(self.actions),
        }

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cleaned_df"] = self.cleaned_df.to_dict(orient="records")
        return out

    def render_text(self) -> str:
        lines = ["CLEANING REPORT", "-" * 15, "", "ACTIONS:"]
        if not self.actions:
            lines.append("(none)")
        else:
            for a in self.actions:
                col = f" column={a.column!r}," if a.column is not None else ""
                lines.append(
                    f"- type={a.action_type},{col} strategy={a.strategy}, "
                    f"affected_rows={a.affected_rows} — {a.message}"
                )
        s = self.summary()
        lines.extend(
            [
                "",
                "SUMMARY:",
                f"- rows_before: {s['rows_before']}",
                f"- rows_after: {s['rows_after']}",
                f"- rows_removed: {s['rows_removed']}",
                f"- total_actions: {s['total_actions']}",
            ]
        )
        return "\n".join(lines)


def _validate_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
    return df


def _validate_null_strategy(value: Any) -> dict[str, str | dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("null_strategy must be a dict or None")
    return value


def _validate_column_exists(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"column '{column}' not found in DataFrame")


def _apply_null_strategy(
    df: pd.DataFrame,
    column: str,
    strategy: str | dict[str, Any],
) -> tuple[pd.DataFrame, CleaningAction]:
    _validate_column_exists(df, column)
    before = len(df)
    null_count = int(df[column].isna().sum())

    if isinstance(strategy, str):
        name = strategy
        if name == "drop_rows":
            next_df = df[df[column].notna()].copy()
            removed = before - len(next_df)
            return next_df, CleaningAction(
                action_type="null_cleaning",
                column=column,
                strategy=name,
                affected_rows=removed,
                message=f"Dropped {removed} row(s) where '{column}' was null.",
            )
        if name in {"mean", "median"}:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(
                    f"strategy '{name}' requires numeric column '{column}'"
                )
            fill = df[column].mean() if name == "mean" else float(df[column].median())
            next_df = df.copy()
            next_df[column] = next_df[column].fillna(fill)
            return next_df, CleaningAction(
                action_type="null_cleaning",
                column=column,
                strategy=name,
                affected_rows=null_count,
                message=f"Filled {null_count} null value(s) in '{column}' using {name}.",
            )
        if name == "mode":
            mode = df[column].mode(dropna=True)
            if mode.empty:
                raise ValueError(
                    f"strategy 'mode' cannot be applied on all-null column '{column}'"
                )
            fill = mode.iloc[0]
            next_df = df.copy()
            next_df[column] = next_df[column].fillna(fill)
            return next_df, CleaningAction(
                action_type="null_cleaning",
                column=column,
                strategy=name,
                affected_rows=null_count,
                message=f"Filled {null_count} null value(s) in '{column}' using mode (value={fill}).",
            )
        if name in {"ffill", "bfill"}:
            next_df = df.copy()
            if name == "ffill":
                next_df[column] = next_df[column].ffill()
            else:
                next_df[column] = next_df[column].bfill()
            affected = int(null_count - next_df[column].isna().sum())
            return next_df, CleaningAction(
                action_type="null_cleaning",
                column=column,
                strategy=name,
                affected_rows=affected,
                message=f"Filled {affected} null value(s) in '{column}' using {name}.",
            )
        raise ValueError(f"unsupported null strategy '{name}' for column '{column}'")

    if isinstance(strategy, dict):
        method = strategy.get("method")
        if method != "fill_value":
            raise ValueError(
                f"dict strategy for column '{column}' must be "
                "{'method': 'fill_value', 'value': ...}"
            )
        if "value" not in strategy:
            raise ValueError(
                f"dict strategy for column '{column}' requires key 'value'"
            )
        fill = strategy["value"]
        next_df = df.copy()
        next_df[column] = next_df[column].fillna(fill)
        return next_df, CleaningAction(
            action_type="null_cleaning",
            column=column,
            strategy="fill_value",
            affected_rows=null_count,
            message=f"Filled {null_count} null value(s) in '{column}' with a fixed value.",
        )

    raise ValueError(f"invalid strategy for column '{column}'")


def _apply_duplicate_strategy(
    df: pd.DataFrame, strategy: dict[str, Any] | str
) -> tuple[pd.DataFrame, CleaningAction]:
    if isinstance(strategy, str):
        if strategy != "drop":
            raise ValueError("duplicate_strategy string must be 'drop'")
        dedup = df.drop_duplicates(keep="first").copy()
        removed = len(df) - len(dedup)
        return dedup, CleaningAction(
            action_type="duplicate_cleaning",
            column=None,
            strategy="drop",
            affected_rows=removed,
            message=f"Dropped {removed} duplicate row(s) using all columns.",
        )

    if not isinstance(strategy, dict):
        raise ValueError("duplicate_strategy must be None, 'drop', or a dict")

    subset = strategy.get("subset")
    keep = strategy.get("keep", "first")
    if subset is not None:
        if not isinstance(subset, list):
            raise ValueError("duplicate_strategy['subset'] must be a list[str] or None")
        missing = [c for c in subset if c not in df.columns]
        if missing:
            raise ValueError(f"duplicate subset has missing columns: {missing}")

    if keep not in {"first", "last", False}:
        raise ValueError("duplicate_strategy['keep'] must be 'first', 'last', or False")

    dedup = df.drop_duplicates(subset=subset, keep=keep).copy()
    removed = len(df) - len(dedup)
    subset_msg = "all columns" if subset is None else f"subset={subset}"
    return dedup, CleaningAction(
        action_type="duplicate_cleaning",
        column=None,
        strategy=f"drop(keep={keep})",
        affected_rows=removed,
        message=f"Dropped {removed} duplicate row(s) using {subset_msg}.",
    )


def clean_dataframe(
    df: pd.DataFrame,
    null_strategy: dict[str, str | dict[str, Any]] | None = None,
    duplicate_strategy: dict[str, Any] | str | None = None,
) -> CleaningResult:
    """
    Apply user-selected null and duplicate preprocessing to a copy of ``df``.

    Only explicitly configured columns/strategies are changed. Missing strategy
    entries mean "leave unchanged".
    """
    df = _validate_dataframe(df)
    null_strategy = _validate_null_strategy(null_strategy)

    working = df.copy()
    rows_before = len(working)
    actions: list[CleaningAction] = []

    for column, strategy in null_strategy.items():
        if not isinstance(column, str) or not column:
            raise ValueError("null_strategy keys must be non-empty column names")
        working, action = _apply_null_strategy(working, column, strategy)
        actions.append(action)

    if duplicate_strategy is not None:
        working, action = _apply_duplicate_strategy(working, duplicate_strategy)
        actions.append(action)

    return CleaningResult(
        cleaned_df=working,
        actions=actions,
        rows_before=rows_before,
        rows_after=len(working),
    )


__all__ = [
    "CleaningAction",
    "CleaningResult",
    "clean_dataframe",
]

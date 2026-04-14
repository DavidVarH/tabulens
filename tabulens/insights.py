"""Content-level insights for pandas DataFrames (not structural profiling)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CategoricalInsight:
    column: str
    most_frequent_value: str
    most_frequent_count: int
    most_frequent_percentage: float
    unique_count: int
    is_dominant: bool


@dataclass(frozen=True)
class NumericInsight:
    column: str
    mean: float
    median: float
    minimum: float
    maximum: float
    std_dev: float
    q1: float
    q3: float


@dataclass(frozen=True)
class InsightsReport:
    """First-pass value insights for categorical-like and numeric columns."""

    categorical_insights: list[CategoricalInsight]
    numeric_insights: list[NumericInsight]
    messages: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to plain Python types."""
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        """High-level counts for dashboards or logging."""
        return {
            "categorical_columns_analyzed": len(self.categorical_insights),
            "numeric_columns_analyzed": len(self.numeric_insights),
            "message_count": len(self.messages),
            "dominant_categorical_columns": sum(
                1 for c in self.categorical_insights if c.is_dominant
            ),
        }

    def render_text(self) -> str:
        """Multi-line text suitable for terminals or notebooks."""
        blocks: list[str] = []

        cat_block = ["CATEGORICAL INSIGHTS", "-" * 24]
        if not self.categorical_insights:
            cat_block.append("(none)")
        else:
            for ins in self.categorical_insights:
                cat_block.append(
                    f"- {ins.column}: mode={ins.most_frequent_value!r} "
                    f"({ins.most_frequent_percentage:.1f}% of rows, "
                    f"{ins.unique_count} unique); "
                    f"dominant={ins.is_dominant}"
                )
        blocks.append("\n".join(cat_block))

        num_block = ["NUMERIC INSIGHTS", "-" * 18]
        if not self.numeric_insights:
            num_block.append("(none)")
        else:
            for ins in self.numeric_insights:
                num_block.append(
                    f"- {ins.column}: mean={ins.mean:.4g}, median={ins.median:.4g}, "
                    f"min={ins.minimum:.4g}, max={ins.maximum:.4g}, "
                    f"std={ins.std_dev:.4g}, Q1={ins.q1:.4g}, Q3={ins.q3:.4g}"
                )
        blocks.append("\n".join(num_block))

        msg_block = ["MESSAGES", "-" * 8]
        if not self.messages:
            msg_block.append("(none)")
        else:
            msg_block.extend(f"- {m}" for m in self.messages)
        blocks.append("\n".join(msg_block))

        return "\n\n".join(blocks)


def _validate_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
    return df


def _validate_params(dominance_threshold: float, max_categorical_unique: int) -> None:
    if not (0.0 < dominance_threshold <= 1.0):
        raise ValueError("dominance_threshold must be in (0.0, 1.0]")
    if max_categorical_unique < 1:
        raise ValueError("max_categorical_unique must be >= 1")


def _col_key(name: Any) -> str:
    return str(name)


def _is_categorical_like_column(
    series: pd.Series,
    *,
    max_categorical_unique: int,
) -> bool:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return True
    if (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    ):
        return True
    if pd.api.types.is_integer_dtype(dtype):
        uniq = int(series.nunique(dropna=True))
        return uniq <= max_categorical_unique
    return False


def _is_numeric_measure_column(series: pd.Series) -> bool:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return False
    return bool(pd.api.types.is_numeric_dtype(dtype))


def _categorical_insight_for_column(
    series: pd.Series,
    column: str,
    row_count: int,
    dominance_threshold: float,
) -> CategoricalInsight | None:
    if series.isna().all():
        return None
    if row_count <= 0:
        return None

    vc = series.value_counts(dropna=True)
    if vc.empty:
        return None

    most = vc.index[0]
    count = int(vc.iloc[0])
    unique_count = int(series.nunique(dropna=True))
    pct = (count / row_count) * 100.0
    is_dominant = (count / row_count) >= dominance_threshold

    return CategoricalInsight(
        column=column,
        most_frequent_value=str(most),
        most_frequent_count=count,
        most_frequent_percentage=float(pct),
        unique_count=unique_count,
        is_dominant=is_dominant,
    )


def _numeric_insight_for_column(series: pd.Series, column: str) -> NumericInsight | None:
    if series.isna().all():
        return None

    s = series.dropna()
    if s.empty:
        return None

    return NumericInsight(
        column=column,
        mean=float(s.mean()),
        median=float(s.median()),
        minimum=float(s.min()),
        maximum=float(s.max()),
        std_dev=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        q1=float(s.quantile(0.25)),
        q3=float(s.quantile(0.75)),
    )


def _messages_for_categorical(ins: CategoricalInsight) -> list[str]:
    col = ins.column
    val = ins.most_frequent_value
    pct = ins.most_frequent_percentage
    if ins.is_dominant:
        return [
            f'Column "{col}": one category dominates the column ({pct:.1f}%).',
        ]
    return [
        (
            f'Column "{col}": most frequent value is {val!r} '
            f"({pct:.1f}% of rows)."
        ),
    ]


def _messages_for_numeric(ins: NumericInsight) -> list[str]:
    col = ins.column
    return [
        (
            f'Column "{col}": median is {ins.median:.1f}, '
            f"with values ranging from {ins.minimum:.1f} to {ins.maximum:.1f}."
        ),
        (
            f'Column "{col}": the middle 50% of values lies between '
            f"{ins.q1:.1f} and {ins.q3:.1f}."
        ),
    ]


def generate_insights(
    df: pd.DataFrame,
    dominance_threshold: float = 0.5,
    max_categorical_unique: int = 20,
) -> InsightsReport:
    """
    Produce first-pass categorical and numeric summaries plus short messages.

    ``dominance_threshold`` is a fraction of total rows (0--1] for ``is_dominant``.
    Integer columns are treated as categorical only when they have at most
    ``max_categorical_unique`` distinct non-null values.
    """
    df = _validate_dataframe(df)
    _validate_params(dominance_threshold, max_categorical_unique)

    row_count = len(df)
    categorical: list[CategoricalInsight] = []
    numeric: list[NumericInsight] = []
    messages: list[str] = []

    for col in df.columns:
        key = _col_key(col)
        series = df[col]

        if _is_categorical_like_column(series, max_categorical_unique=max_categorical_unique):
            cat_ins = _categorical_insight_for_column(
                series, key, row_count, dominance_threshold
            )
            if cat_ins is not None:
                categorical.append(cat_ins)
                messages.extend(_messages_for_categorical(cat_ins))
        elif _is_numeric_measure_column(series):
            num_ins = _numeric_insight_for_column(series, key)
            if num_ins is not None:
                numeric.append(num_ins)
                messages.extend(_messages_for_numeric(num_ins))

    return InsightsReport(
        categorical_insights=categorical,
        numeric_insights=numeric,
        messages=messages,
    )


__all__ = [
    "CategoricalInsight",
    "InsightsReport",
    "NumericInsight",
    "generate_insights",
]

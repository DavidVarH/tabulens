import pandas as pd
import pytest

from tabulens.insights import (
    CategoricalInsight,
    InsightsReport,
    NumericInsight,
    generate_insights,
)


def test_generate_insights_returns_report() -> None:
    df = pd.DataFrame(
        {
            "producto": ["chicles", "chicles", "chicles", "goma", "goma", "goma"],
            "ventas": [10.0, 50.0, 120.0, 200.0, 500.0, 80.0],
            "edad": [18, 22, 30, 35, 40, 55],
        }
    )
    report = generate_insights(df, dominance_threshold=0.45)

    assert isinstance(report, InsightsReport)
    assert all(isinstance(c, CategoricalInsight) for c in report.categorical_insights)
    assert all(isinstance(n, NumericInsight) for n in report.numeric_insights)
    assert any(n.column == "ventas" for n in report.numeric_insights)
    assert any(c.column == "edad" for c in report.categorical_insights)
    assert any("producto" in m for m in report.messages)
    assert any("ventas" in m and "median" in m for m in report.messages)
    assert any("edad" in m for m in report.messages)


def test_dominant_categorical_message() -> None:
    df = pd.DataFrame({"estado": ["A"] * 8 + ["B"] * 2})
    report = generate_insights(df, dominance_threshold=0.7)

    cat = next(c for c in report.categorical_insights if c.column == "estado")
    assert cat.is_dominant is True
    assert any("dominates the column" in m for m in report.messages)


def test_low_cardinality_int_is_categorical_not_numeric() -> None:
    df = pd.DataFrame({"code": [1, 2, 1, 2, 1]})
    report = generate_insights(df, max_categorical_unique=10)

    assert any(c.column == "code" for c in report.categorical_insights)
    assert not any(n.column == "code" for n in report.numeric_insights)


def test_high_cardinality_int_is_numeric_only() -> None:
    df = pd.DataFrame({"x": list(range(30))})
    report = generate_insights(df, max_categorical_unique=20)

    assert any(n.column == "x" for n in report.numeric_insights)
    assert not any(c.column == "x" for c in report.categorical_insights)


def test_skips_all_null_column() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [pd.NA, pd.NA]})
    report = generate_insights(df)

    assert not any(c.column == "b" for c in report.categorical_insights)
    assert not any(n.column == "b" for n in report.numeric_insights)


def test_type_error_on_invalid_df() -> None:
    with pytest.raises(TypeError):
        generate_insights("not a frame")  # type: ignore[arg-type]


def test_value_error_on_invalid_thresholds() -> None:
    df = pd.DataFrame({"x": [1]})

    with pytest.raises(ValueError):
        generate_insights(df, dominance_threshold=0.0)

    with pytest.raises(ValueError):
        generate_insights(df, dominance_threshold=1.5)

    with pytest.raises(ValueError):
        generate_insights(df, max_categorical_unique=0)


def test_numeric_iqr_message_for_non_integer_column() -> None:
    df = pd.DataFrame({"ventas": [10.0, 50.0, 120.0, 200.0, 500.0, 80.0]})
    report = generate_insights(df)

    assert any("ventas" in m and "middle 50%" in m for m in report.messages)


def test_render_text_sections() -> None:
    df = pd.DataFrame({"c": ["a", "b"], "n": [1.0, 2.0]})
    text = generate_insights(df).render_text()

    assert "CATEGORICAL INSIGHTS" in text
    assert "NUMERIC INSIGHTS" in text
    assert "MESSAGES" in text
    assert "c" in text or "n" in text


def test_summary_and_to_dict() -> None:
    df = pd.DataFrame({"c": ["x", "y"], "n": [1, 2]})
    report = generate_insights(df)

    s = report.summary()
    assert s["categorical_columns_analyzed"] >= 1
    # Integer columns with few distinct values are treated as categorical, not numeric.
    assert s["numeric_columns_analyzed"] == 0

    d = report.to_dict()
    assert "categorical_insights" in d
    assert "numeric_insights" in d
    assert isinstance(d["messages"], list)

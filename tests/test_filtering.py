import pandas as pd
import pytest

from tabulens.rules import RuleSet
from tabulens.validation import validate_dataframe
from tabulens.filtering import (
    keep_valid_rows,
    keep_invalid_rows,
    split_valid_invalid_rows,
)


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edad": [20, None, 30, None],
            "email": ["a@a.com", "bad", "b@b.com", "c@c.com"],
        }
    )


def _validation_report(df: pd.DataFrame):
    rules = RuleSet().not_null("edad").regex("email", r"^[^@]+@[^@]+\.[^@]+$")
    return validate_dataframe(df, rules)


# --------------------------------------------------
# BASIC FUNCTIONALITY
# --------------------------------------------------


def test_keep_valid_rows_default_behavior() -> None:
    df = _base_df()
    report = _validation_report(df)

    valid_df = keep_valid_rows(df, report)

    # filas válidas: índice 0 y 2
    assert list(valid_df.index) == [0, 2]


def test_keep_invalid_rows_default_behavior() -> None:
    df = _base_df()
    report = _validation_report(df)

    invalid_df = keep_invalid_rows(df, report)

    # inválidas: índice 1 (null + bad email) y 3 (null edad)
    assert list(invalid_df.index) == [1, 3]


def test_split_valid_invalid_rows() -> None:
    df = _base_df()
    report = _validation_report(df)

    valid_df, invalid_df = split_valid_invalid_rows(df, report)

    assert list(valid_df.index) == [0, 2]
    assert list(invalid_df.index) == [1, 3]


# --------------------------------------------------
# NEAR FAILING BEHAVIOR
# --------------------------------------------------


def test_include_near_failing_false() -> None:
    df = pd.DataFrame({"x": list(range(99)) + [1000]})
    rules = RuleSet().in_range("x", min_value=0, max_value=200)

    report = validate_dataframe(df, rules)

    # el único error es near_failing
    valid_df = keep_valid_rows(df, report, include_near_failing=False)

    # NO se elimina
    assert len(valid_df) == 100


def test_include_near_failing_true() -> None:
    df = pd.DataFrame({"x": list(range(99)) + [1000]})
    rules = RuleSet().in_range("x", min_value=0, max_value=200)

    report = validate_dataframe(df, rules)

    valid_df = keep_valid_rows(df, report, include_near_failing=True)

    # se elimina la fila inválida
    assert len(valid_df) == 99
    assert 99 not in valid_df.index


# --------------------------------------------------
# TYPE VALIDATION
# --------------------------------------------------


def test_invalid_inputs_raise() -> None:
    df = _base_df()
    report = _validation_report(df)

    with pytest.raises(TypeError):
        keep_valid_rows("bad", report)  # type: ignore

    with pytest.raises(TypeError):
        keep_valid_rows(df, "bad")  # type: ignore

    with pytest.raises(TypeError):
        keep_invalid_rows("bad", report)  # type: ignore

    with pytest.raises(TypeError):
        split_valid_invalid_rows(df, "bad")  # type: ignore


# --------------------------------------------------
# IMMUTABILITY
# --------------------------------------------------


def test_original_dataframe_not_modified() -> None:
    df = _base_df()
    df_copy = df.copy()

    report = _validation_report(df)
    _ = keep_valid_rows(df, report)

    pd.testing.assert_frame_equal(df, df_copy)

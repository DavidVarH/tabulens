import pandas as pd
import pytest

from tabulens.cleaning import CleaningResult, clean_dataframe


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edad": [20.0, None, 30.0, None],
            "estado": ["CDMX", None, "Jalisco", None],
            "score": [1, 1, 2, 2],
            "email": ["a@a.com", "b@b.com", "b@b.com", "c@c.com"],
        }
    )


def test_no_strategies_changes_nothing() -> None:
    df = _base_df()
    result = clean_dataframe(df)

    assert isinstance(result, CleaningResult)
    pd.testing.assert_frame_equal(result.cleaned_df, df)
    pd.testing.assert_frame_equal(df, _base_df())
    assert result.actions == []


def test_null_drop_rows_only_on_selected_column() -> None:
    df = _base_df()
    result = clean_dataframe(df, null_strategy={"estado": "drop_rows"})

    assert len(result.cleaned_df) == 2
    assert result.actions[0].affected_rows == 2
    assert result.actions[0].strategy == "drop_rows"
    pd.testing.assert_frame_equal(df, _base_df())


def test_null_mean_and_median() -> None:
    df = _base_df()

    r_mean = clean_dataframe(df, null_strategy={"edad": "mean"})
    assert r_mean.cleaned_df["edad"].isna().sum() == 0
    assert r_mean.actions[0].affected_rows == 2

    r_median = clean_dataframe(df, null_strategy={"edad": "median"})
    assert r_median.cleaned_df["edad"].isna().sum() == 0
    assert r_median.actions[0].affected_rows == 2

    pd.testing.assert_frame_equal(df, _base_df())


def test_null_mode_ffill_bfill_fill_value() -> None:
    df = _base_df()

    r_mode = clean_dataframe(df, null_strategy={"estado": "mode"})
    assert r_mode.cleaned_df["estado"].isna().sum() == 0
    assert r_mode.actions[0].affected_rows == 2

    r_ffill = clean_dataframe(df, null_strategy={"estado": "ffill"})
    assert r_ffill.actions[0].affected_rows == 2

    r_bfill = clean_dataframe(df, null_strategy={"estado": "bfill"})
    assert r_bfill.actions[0].affected_rows == 1

    r_fixed = clean_dataframe(
        df, null_strategy={"estado": {"method": "fill_value", "value": "UNKNOWN"}}
    )
    assert "UNKNOWN" in r_fixed.cleaned_df["estado"].values
    assert r_fixed.actions[0].affected_rows == 2

    pd.testing.assert_frame_equal(df, _base_df())


def test_duplicate_drop_and_subset() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [9, 9, 8, 7]})

    r1 = clean_dataframe(df, duplicate_strategy="drop")
    assert len(r1.cleaned_df) == 3
    assert r1.actions[0].affected_rows == 1

    r2 = clean_dataframe(df, duplicate_strategy={"subset": ["a"], "keep": "first"})
    assert len(r2.cleaned_df) == 2
    assert r2.actions[0].affected_rows == 2

    pd.testing.assert_frame_equal(
        df, pd.DataFrame({"a": [1, 1, 2, 2], "b": [9, 9, 8, 7]})
    )


def test_invalid_inputs_raise() -> None:
    df = _base_df()

    with pytest.raises(TypeError):
        clean_dataframe("bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        clean_dataframe(df, null_strategy="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        clean_dataframe(df, null_strategy={"missing": "mean"})

    with pytest.raises(ValueError):
        clean_dataframe(df, null_strategy={"estado": "mean"})  # non numeric

    with pytest.raises(ValueError):
        clean_dataframe(df, null_strategy={"estado": {"method": "other"}})

    with pytest.raises(ValueError):
        clean_dataframe(df, duplicate_strategy="other")

    with pytest.raises(ValueError):
        clean_dataframe(df, duplicate_strategy={"subset": ["missing"], "keep": "first"})

    with pytest.raises(ValueError):
        clean_dataframe(df, duplicate_strategy={"subset": ["edad"], "keep": "bad"})


def test_summary_to_dict_render_text() -> None:
    df = _base_df()
    result = clean_dataframe(
        df,
        null_strategy={"edad": "median"},
        duplicate_strategy="drop",
    )

    summary = result.summary()
    assert summary["rows_before"] == 4
    assert summary["rows_after"] == len(result.cleaned_df)
    assert summary["rows_removed"] == summary["rows_before"] - summary["rows_after"]
    assert summary["total_actions"] == 2

    out = result.to_dict()
    assert "cleaned_df" in out
    assert isinstance(out["cleaned_df"], list)

    text = result.render_text()
    assert "CLEANING REPORT" in text
    assert "SUMMARY" in text

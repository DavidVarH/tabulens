import pandas as pd

from tabulens.optimization import OptimizationResult, optimize_dataframe
from tabulens.profiling import profile_dataframe


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category": ["a", "a", "b", "a", "b"],
            "constant": [1, 1, 1, 1, 1],
            "dates": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
            ],
        }
    )


def test_optimize_default_selects_nothing() -> None:
    df = _sample_frame()
    profile = profile_dataframe(df)
    result = optimize_dataframe(df, profile)

    assert isinstance(result, OptimizationResult)
    assert result.applied_optimizations == []
    assert len(result.skipped_optimizations) == len(profile.structural_recommendations)
    assert all(not s.success for s in result.skipped_optimizations)

    summary = result.summary()
    assert summary["applied_count"] == 0
    assert summary["skipped_count"] == len(profile.structural_recommendations)

    pd.testing.assert_frame_equal(df, _sample_frame())
    pd.testing.assert_frame_equal(result.optimized_df, df)


def test_optimize_drops_constant_when_rule_selected() -> None:
    df = _sample_frame()
    profile = profile_dataframe(df)
    result = optimize_dataframe(df, profile, selected_rule_ids=["drop_constant_column"])

    assert "constant" not in result.optimized_df.columns
    assert len(result.applied_optimizations) == 1
    assert result.applied_optimizations[0].rule_id == "drop_constant_column"
    assert result.applied_optimizations[0].success is True
    assert "constant" in df.columns


def test_review_rules_are_never_applied_even_when_selected() -> None:
    df = pd.DataFrame({"id": list(range(25)), "x": [1] * 25})
    profile = profile_dataframe(df)
    result = optimize_dataframe(
        df,
        profile,
        selected_rule_ids=["review_identifier_column"],
    )

    assert result.applied_optimizations == []
    assert any(
        s.rule_id == "review_identifier_column" and not s.success
        for s in result.skipped_optimizations
    )
    assert "Review-only" in next(
        s.message
        for s in result.skipped_optimizations
        if s.rule_id == "review_identifier_column"
    )


def test_column_and_rule_filters_combine_with_and() -> None:
    df = _sample_frame()
    profile = profile_dataframe(df)
    result = optimize_dataframe(
        df,
        profile,
        selected_rule_ids=["convert_to_category"],
        selected_columns=["category"],
    )

    assert len(result.applied_optimizations) == 1
    assert result.applied_optimizations[0].action == "convert_to_category"
    assert isinstance(result.optimized_df["category"].dtype, pd.CategoricalDtype)


def test_column_filter_without_rule_applies_matching_columns() -> None:
    df = _sample_frame()
    profile = profile_dataframe(df)
    result = optimize_dataframe(df, profile, selected_columns=["constant"])

    dropped = [a for a in result.applied_optimizations if a.action == "drop_column"]
    assert len(dropped) == 1
    assert dropped[0].column == "constant"


def test_missing_column_skipped_without_crash() -> None:
    df = _sample_frame()
    profile = profile_dataframe(df)
    df2 = df.drop(columns=["dates"])
    result = optimize_dataframe(
        df2,
        profile,
        selected_rule_ids=["convert_to_datetime"],
    )

    assert result.applied_optimizations == []
    assert any("not present" in s.message for s in result.skipped_optimizations)


def test_type_errors_for_bad_inputs() -> None:
    df = _sample_frame()
    profile = profile_dataframe(df)
    try:
        optimize_dataframe([], profile)  # type: ignore[arg-type]
    except TypeError:
        assert True
    else:
        assert False

    try:
        optimize_dataframe(df, object())  # type: ignore[arg-type]
    except TypeError:
        assert True
    else:
        assert False


def test_memory_fields_present() -> None:
    df = _sample_frame()
    profile = profile_dataframe(df)
    result = optimize_dataframe(df, profile, selected_rule_ids=["drop_constant_column"])
    s = result.summary()
    assert s["memory_before_bytes"] >= 0
    assert s["memory_after_bytes"] >= 0
    assert s["memory_saved_bytes"] == s["memory_before_bytes"] - s["memory_after_bytes"]


def test_optimize_converts_datetime_when_selected() -> None:
    df = _sample_frame()
    profile = profile_dataframe(df)
    result = optimize_dataframe(
        df,
        profile,
        selected_rule_ids=["convert_to_datetime"],
        selected_columns=["dates"],
    )

    assert len(result.applied_optimizations) == 1
    assert result.applied_optimizations[0].rule_id == "convert_to_datetime"
    assert pd.api.types.is_datetime64_any_dtype(result.optimized_df["dates"])

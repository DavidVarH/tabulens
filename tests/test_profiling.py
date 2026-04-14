import pandas as pd

from tabulens.profiling import (
    ProfileReport,
    StructuralRecommendation,
    profile_dataframe,
)


def test_profile_dataframe_returns_profile_report() -> None:
    df = pd.DataFrame(
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

    report = profile_dataframe(df)

    assert isinstance(report, ProfileReport)
    assert report.row_count == 5
    assert report.column_count == 4
    assert "constant" in report.constant_columns
    assert "category" in report.suggested_categorical
    assert "dates" in report.suspected_datetime_columns


def test_profile_dataframe_raises_type_error_for_invalid_input() -> None:
    try:
        profile_dataframe([1, 2, 3])  # type: ignore[arg-type]
    except TypeError:
        assert True
    else:
        assert False, "Expected TypeError for invalid input"


def test_profile_dataframe_generates_structural_recommendations() -> None:
    df = pd.DataFrame(
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

    report = profile_dataframe(df)

    assert len(report.structural_recommendations) > 0
    assert all(
        isinstance(rec, StructuralRecommendation)
        for rec in report.structural_recommendations
    )

    rule_ids = {rec.rule_id for rec in report.structural_recommendations}
    columns = {rec.column for rec in report.structural_recommendations}

    assert "drop_constant_column" in rule_ids
    assert "convert_to_category" in rule_ids
    assert "convert_to_datetime" in rule_ids
    assert "review_identifier_column" in rule_ids

    assert "constant" in columns
    assert "category" in columns
    assert "dates" in columns
    assert "id" in columns


def test_profile_summary_includes_recommendation_count() -> None:
    df = pd.DataFrame(
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

    report = profile_dataframe(df)
    summary = report.summary()

    assert "structural_recommendations_count" in summary
    assert summary["structural_recommendations_count"] == len(
        report.structural_recommendations
    )

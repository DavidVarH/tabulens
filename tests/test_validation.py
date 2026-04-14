import pandas as pd
import pytest

from tabulens.rules import RuleSet
from tabulens.validation import RuleResult, ValidationReport, validate_dataframe


def test_validate_dataframe_type_errors() -> None:
    rules = RuleSet().not_null("x")

    with pytest.raises(TypeError):
        validate_dataframe("not a df", rules)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        validate_dataframe(pd.DataFrame({"x": [1]}), "not rules")  # type: ignore[arg-type]


def test_not_null_pass_and_fail_with_indices() -> None:
    df = pd.DataFrame({"a": [1, None, 3]})
    rules = RuleSet().not_null("a")

    report = validate_dataframe(df, rules)
    r = report.results[0]

    assert isinstance(r, RuleResult)
    assert r.column == "a"
    assert r.affected_rows == 1
    assert r.row_indices == [1]
    assert r.status in {"near_failing", "failed"}
    assert "null" in r.message


def test_unique_ignores_nulls_and_tracks_indices() -> None:
    df = pd.DataFrame({"id": [1, 1, None, None, 2]})
    rules = RuleSet().unique("id")

    report = validate_dataframe(df, rules)
    r = report.results[0]

    assert r.affected_rows == 2
    assert sorted(r.row_indices) == [0, 1]
    assert r.status == "failed"


def test_range_rule_indices_and_status() -> None:
    df = pd.DataFrame({"x": list(range(99)) + [1000]})  # 1 invalid
    rules = RuleSet().in_range("x", min_value=0, max_value=200)

    report = validate_dataframe(df, rules)
    r = report.results[0]

    assert r.row_indices == [99]
    assert r.status == "near_failing"


def test_allowed_values_and_regex_indices() -> None:
    df = pd.DataFrame(
        {
            "estado": ["CDMX", "Jalisco", None, "INVALID"],
            "email": ["a@b.com", None, "bad", "x@y.z"],
        }
    )

    rules = (
        RuleSet()
        .allowed_values("estado", ["CDMX", "Jalisco"])
        .regex("email", r"^[^@]+@[^@]+\.[^@]+$")
    )

    report = validate_dataframe(df, rules)

    estado_result = report.results[0]
    email_result = report.results[1]

    assert estado_result.row_indices == [3]
    assert email_result.row_indices == [2]


def test_missing_column_has_all_indices() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    rules = RuleSet().not_null("missing")

    report = validate_dataframe(df, rules)
    r = report.results[0]

    assert r.status == "failed"
    assert r.row_indices == [0, 1, 2]


def test_report_helpers_and_indices_dicts() -> None:
    df = pd.DataFrame({"a": [1, None, 3]})
    rules = RuleSet().not_null("a")

    report = validate_dataframe(df, rules)

    failed = report.failed_indices()
    key = "not_null:a"

    assert key in failed
    assert failed[key] == [1]


def test_render_text_includes_indices() -> None:
    df = pd.DataFrame({"a": [1, None, 3]})
    rules = RuleSet().not_null("a")

    report = validate_dataframe(df, rules)
    text = report.render_text()

    assert "VALIDATION REPORT" in text
    assert "indices=" in text

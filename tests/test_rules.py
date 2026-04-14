import pytest

from tabulens.rules import (
    AllowedValuesRule,
    NotNullRule,
    RangeRule,
    RegexRule,
    RuleSet,
    UniqueRule,
    ValidationRule,
)


def test_ruleset_chaining_and_types() -> None:
    rules = (
        RuleSet()
        .not_null("  cliente_id  ")
        .unique("cliente_id")
        .in_range("edad", min_value=18, max_value=99)
        .allowed_values("estado", ["CDMX", "Jalisco", "Nuevo León"])
        .regex("email", r"^[^@]+@[^@]+\.[^@]+$")
    )

    lst = rules.to_list()
    assert len(lst) == 5
    assert isinstance(lst[0], NotNullRule)
    assert lst[0].column == "cliente_id"
    assert isinstance(lst[1], UniqueRule)
    assert isinstance(lst[2], RangeRule)
    assert lst[2].min_value == 18 and lst[2].max_value == 99
    assert isinstance(lst[3], AllowedValuesRule)
    assert lst[3].allowed_values == ("CDMX", "Jalisco", "Nuevo León")
    assert isinstance(lst[4], RegexRule)
    assert lst[4].pattern == r"^[^@]+@[^@]+\.[^@]+$"

    s = rules.summary()
    assert s["total_rules"] == 5
    assert s["counts_by_rule_type"] == {
        "allowed_values": 1,
        "in_range": 1,
        "not_null": 1,
        "regex": 1,
        "unique": 1,
    }


def test_in_range_partial_bounds() -> None:
    r = RuleSet().in_range("x", min_value=0.0).in_range("y", max_value=10.0).to_list()
    assert r[0].min_value == 0.0 and r[0].max_value is None
    assert r[1].min_value is None and r[1].max_value == 10.0


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RuleSet().not_null("")
    with pytest.raises(ValueError, match="non-empty"):
        RuleSet().not_null("   ")
    with pytest.raises(ValueError, match="non-empty string"):
        RuleSet().not_null(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        RuleSet().in_range("z")
    with pytest.raises(ValueError, match="non-empty list"):
        RuleSet().allowed_values("a", [])
    with pytest.raises(ValueError, match="non-empty list"):
        RuleSet().allowed_values("a", ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty pattern"):
        RuleSet().regex("e", "")
    with pytest.raises(ValueError, match="non-empty pattern"):
        RuleSet().regex("e", "   ")


def test_subclasses_inherit_validation_rule() -> None:
    r: ValidationRule = NotNullRule(rule_name="not_null", column="c")
    assert r.rule_name == "not_null" and r.column == "c"

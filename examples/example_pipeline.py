import pandas as pd

from tabulens import (
    RuleSet,
    clean_dataframe,
    generate_insights,
    keep_valid_rows,
    optimize_dataframe,
    profile_dataframe,
    validate_dataframe,
)


def main() -> None:
    # --------------------------------------------
    # 1. Example dataset
    # --------------------------------------------
    df = pd.DataFrame(
        {
            "cliente_id": [101, 102, 102, 104, None, 106],
            "edad": [25, None, 130, 40, 35, None],
            "estado": ["CDMX", "Jalisco", "Jalisco", None, "CDMX", "Puebla"],
            "email": [
                "ana@email.com",
                "bad_email",
                "bad_email",
                "luis@email.com",
                None,
                "sofia@email.com",
            ],
            "fecha_registro": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-02-10",
                None,
                "2024-03-01",
            ],
            "segmento": ["A", "A", "A", "A", "A", "A"],
            "ventas": [100.0, 150.0, 150.0, 300.0, 120.0, 500.0],
        }
    )

    print("=" * 60)
    print("ORIGINAL DATAFRAME")
    print("=" * 60)
    print(df)
    print()

    # --------------------------------------------
    # 2. Profiling
    # --------------------------------------------
    profile = profile_dataframe(df)

    print("=" * 60)
    print("PROFILE SUMMARY")
    print("=" * 60)
    print(profile.summary())
    print()

    print("=" * 60)
    print("STRUCTURAL RECOMMENDATIONS")
    print("=" * 60)
    for rec in profile.structural_recommendations:
        print(
            f"- [{rec.rule_id}] column='{rec.column}' | "
            f"action='{rec.suggested_action}' | {rec.message}"
        )
    print()

    # --------------------------------------------
    # 3. Cleaning
    # --------------------------------------------
    cleaning = clean_dataframe(
        df,
        null_strategy={
            "edad": "median",
            "estado": {"method": "fill_value", "value": "DESCONOCIDO"},
            "fecha_registro": "bfill",
        },
        duplicate_strategy={"subset": ["cliente_id"], "keep": "first"},
    )

    cleaned_df = cleaning.cleaned_df

    print("=" * 60)
    print("CLEANING REPORT")
    print("=" * 60)
    print(cleaning.render_text())
    print()

    print("=" * 60)
    print("CLEANED DATAFRAME")
    print("=" * 60)
    print(cleaned_df)
    print()

    # --------------------------------------------
    # 4. Validation
    # --------------------------------------------
    rules = (
        RuleSet()
        .not_null("cliente_id")
        .unique("cliente_id")
        .in_range("edad", min_value=18, max_value=99)
        .allowed_values("estado", ["CDMX", "Jalisco", "Puebla", "DESCONOCIDO"])
        .regex("email", r"^[^@]+@[^@]+\.[^@]+$")
    )

    validation = validate_dataframe(cleaned_df, rules)

    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(validation.render_text())
    print()

    # --------------------------------------------
    # 5. Filtering
    # --------------------------------------------
    valid_df = keep_valid_rows(cleaned_df, validation, include_near_failing=False)

    print("=" * 60)
    print("VALID ROWS ONLY")
    print("=" * 60)
    print(valid_df)
    print()

    # --------------------------------------------
    # 6. Optimization
    # --------------------------------------------
    optimized = optimize_dataframe(
        valid_df,
        profile_dataframe(valid_df),
        selected_rule_ids=["convert_to_datetime", "drop_constant_column"],
    )

    optimized_df = optimized.optimized_df

    print("=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(optimized.summary())
    print()

    print("=" * 60)
    print("OPTIMIZED DATAFRAME")
    print("=" * 60)
    print(optimized_df)
    print()

    # --------------------------------------------
    # 7. Insights
    # --------------------------------------------
    insights = generate_insights(optimized_df)

    print("=" * 60)
    print("INSIGHTS REPORT")
    print("=" * 60)
    print(insights.render_text())
    print()


if __name__ == "__main__":
    main()

import pandas as pd
from tabulens.profiling import profile_dataframe

df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "category": ["a", "a", "b", "a", "b"],
        "constant": [1, 1, 1, 1, 1],
        "dates": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
    }
)

report = profile_dataframe(df)

print(report.summary())
print(report.suggested_categorical)
print(report.suspected_identifier_columns)
print(report.suspected_datetime_columns)

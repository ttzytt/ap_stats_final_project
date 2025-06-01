from typing import Any
import polars as pl
from pprint import pprint
import os


def extract_unique_options(
    file_path: str, sheet_name: str, columns: list[str]
) -> dict[str, dict[str, Any]]:
    """
    Read an Excel file using Polars and return, for each requested column,
    its total non-null count and a list of unique values with their counts
    and percentages.

    Args:
        file_path: Path to the .xlsx file.
        sheet_name: Name of the worksheet to read.
        columns: List of column names for which to extract unique options.

    Returns:
        A dict where each key is a column name, and each value is another dict:
            {
                "total": int,           # number of non-null entries in that column
                "options": [            # list of dicts, one per unique value
                    {
                        "value": Any,         # the distinct cell value
                        "count": int,         # how many times it appears
                        "percentage": float   # count / total
                    },
                    ...
                ]
            }

    Raises:
        ValueError: If any requested column name does not exist in the DataFrame.
    """
    # Load the specified sheet into a Polars DataFrame
    df = pl.read_excel(file_path, sheet_name=sheet_name)

    result: dict[str, dict[str, Any]] = {}

    for col in columns:
        # Verify that the column exists in the DataFrame
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in sheet '{sheet_name}'.")

        # Drop nulls to consider only actual entries
        series_nonnull = df[col].drop_nulls()
        total_nonnull = series_nonnull.len()

        # Compute counts per unique value
        counts_df = (
            series_nonnull.to_frame(name=col)
            .group_by(col)
            .agg(pl.count().alias("count"))
        )

        # Build a list of dicts: [{"value": ..., "count": ..., "percentage": ...}, ...]
        options_list: list[dict[str, Any]] = []
        for row in counts_df.iter_rows(named=True):
            val = row[col]
            cnt = row["count"]
            pct = cnt / total_nonnull if total_nonnull > 0 else 0.0
            options_list.append({"value": val, "count": cnt, "percentage": pct})

        # Sort options_list by count descending (optional)
        options_list.sort(key=lambda x: x["count"], reverse=True)

        result[col] = {"total": total_nonnull, "options": options_list}

    return result


if __name__ == "__main__":
    # Example usage
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    file_path = os.path.join(data_path, "2020.xlsx")
    sheet_name = "Raw Data"
    col_names = [
        "Should you take this survey?",
        "Which best describes your situation?",
        "Race/Ethnicity",
        "Ethnicity (Is Hispanic?)",
        "Gender",
        "Sexual Orientation",
        "Religious Affiliation",
        "Political Affiliation",
        "Residence",
        "Where are you from?",
        "Which best describes your relationship with immigration?",
        "Where do you live/attend high school?",
        "What kind of school did you graduate from?",
        "How would you rank your school's competitiveness on a scale from 1 to 10?",
        "What is your unweighted high school GPA?",
        "What is your weighted high school GPA?",
        "Did you qualify for the National Merit Scholarship when you took the PSAT?",
        "Did you take the ACT?",
        "What was your composite ACT score?",
        "ACT English Score:",
        "ACT Math Score:",
        "ACT Reading Score:",
        "ACT Science Score:",
        "Did you take the SAT?",
        "What was your total SAT score?",
        "SAT Evidence-based Reading and Writing Score:",
        "SAT Math Score:",
        "Which best describes your field of study?",
        "What are your plans post-college?",
        "Do you plan to participate in the U.S. military ROTC?",
    ]

    try:
        unique_options = extract_unique_options(file_path, sheet_name, col_names)
        with open("unique_options.txt", "w") as f:
            pprint(unique_options, f)
    except ValueError as e:
        print(f"Error: {e}")

import polars as pl
from dataclasses import dataclass
from enum import Enum

@dataclass
class School:
    name: str
    rank: int

def load_survey_data(path: str, sheet_name: str) -> pl.DataFrame:
    """
    Read the raw Excel file into a Polars DataFrame.
    Automatically adds a `row_id` column via `.with_row_count()`.
    """
    return pl.read_excel(path, sheet_name=sheet_name).with_row_count("row_id")


def load_ranked_schools(path: str) -> list[School]:
    """
    Read a text file where each line is a school name (ordered by rank).
    Returns a list of School(name, rank), where rank starts at 1.
    """
    ranked: list[School] = []
    with open(path, "r", encoding="utf8") as f:
        for idx, line in enumerate(f, start=1):
            name = line.strip()
            if name:
                ranked.append(School(name=name, rank=idx))
    return ranked


class FamilyIncome(Enum):
    LESS_THAN_20K = (None, 20_000)
    RANGE_20K_45K = (20_000, 45_000)
    RANGE_45K_70K = (45_000, 70_000)
    RANGE_70K_100K = (70_000, 100_000)
    RANGE_100K_150K = (100_000, 150_000)
    RANGE_150K_200K = (150_000, 200_000)
    RANGE_200K_250K = (200_000, 250_000)
    MORE_THAN_250K = (250_000, None)

    def __init__(self, low: int | None, high: int | None):
        self.low = low
        self.high = high

    def label(self) -> str:
        if self.low is None:
            return f"Less than ${self.high:,}"
        if self.high is None:
            return f"More than ${self.low:,}"
        return f"${self.low:,} - ${self.high:,}"

    @classmethod
    def get_label_map(cls) -> dict[str, "FamilyIncome"]:
        return {fi.label(): fi for fi in cls}



def build_apply_and_decision_cols(
    df: pl.DataFrame,
    schools: list[School],
) -> tuple[list[str], list[str]]:
    prefix_apply = "Which of the following did you apply to? ["
    prefix_decision = (
        "Select your admission decision results for each school you applied to: ["
    )

    apply_cols: list[str] = []
    decision_cols: list[str] = []

    for school in schools:
        col_apply = f"{prefix_apply}{school.name}]"
        if col_apply in df.columns:
            apply_cols.append(col_apply)

        col_dec = f"{prefix_decision}{school.name}]"
        if col_dec in df.columns:
            decision_cols.append(col_dec)

    return apply_cols, decision_cols


def unpivot_applied_or_decided(
    df: pl.DataFrame,
    cols_to_melt: list[str],
    prefix: str,
    new_col_name: str,
) -> pl.DataFrame:
    """
    Melt on `cols_to_melt`, index=["row_id","income_bracket"], variable_name="__tmp_var".
    Then strip `prefix` and trailing ']' to form a bare "school" column.
    """
    return (
        df.unpivot(
            on=cols_to_melt,
            index=["row_id", "income_bracket"],
            variable_name="__tmp_var",
            value_name=new_col_name,
        )
        .with_columns(
            pl.col("__tmp_var")
            .str.replace(prefix, "", literal=True)
            .str.strip_suffix("]")
            .alias("school")
        )
        .drop("__tmp_var")
    )



def join_applied_and_decided(
    applied: pl.DataFrame,
    decided: pl.DataFrame,
) -> pl.DataFrame:
    """
    Inner join on ['row_id','income_bracket','school'], then filter out nulls.
    """
    return applied.join(
        decided,
        on=["row_id", "income_bracket", "school"],
        how="inner",
    ).filter(pl.col("applied").is_not_null() & pl.col("decision").is_not_null())


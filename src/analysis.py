# income_admission_analysis.py

import plotly
import plotly.graph_objects
import plotly.graph_objs._figure
import polars as pl
from enum import Enum
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class School:
    name: str
    rank: int


# — 1. Data loading — #


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


# — 2. FamilyIncome enum — #


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


# — 3. Column-selection & unpivot helpers — #


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


# — 4. Join & filtering logic — #


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


# — 5. Admit-rate computation — #


def compute_admit_rate_by_income(
    joined: pl.DataFrame,
    decision_col: str = "decision",
) -> pl.DataFrame:
    """
    Group by 'income_bracket' and compute percent accepted overall.
    """
    return (
        joined.with_columns((pl.col(decision_col) == "Accepted").alias("accepted"))
        .group_by("income_bracket")
        .agg((pl.sum("accepted") / pl.len() * 100).alias("admit_rate"))
        .filter(
            (pl.col("income_bracket") != "x") & (pl.col("income_bracket") != "Blank")
        )
    )


def compute_admit_rate_matrix(
    joined: pl.DataFrame,
    decision_col: str = "decision",
) -> pl.DataFrame:
    """
    Pivot so that rows = 'school', columns = 'income_bracket', cells = % accepted.

    Uses Polars’ `pivot(on=…, index=…, values=…, aggregate_function="mean")`:
      - on="income_bracket"
      - index="school"
      - values="accepted" (boolean)
      - aggregate_function="mean"
    Then multiply each cell by 100 to convert from fraction→percent.
    Finally reorder columns to match FamilyIncome order.
    """
    # 1) Create boolean `accepted` column
    temp = joined.with_columns((pl.col(decision_col) == "Accepted").alias("accepted"))

    # 2) Pivot: on="income_bracket", index="school", values="accepted", aggregate_function="mean"
    matrix = temp.pivot(
        on="income_bracket",
        index="school",
        values="accepted",
        aggregate_function="mean",
    )

    # 3) Multiply each pivoted column (except "school") by 100
    bracket_cols = [c for c in matrix.columns if c != "school"]
    matrix = matrix.with_columns([(pl.col(c) * 100).alias(c) for c in bracket_cols])

    # 4) Reorder columns: keep "school" first, then only those income labels that appear, in FamilyIncome order
    ordered_labels = [fi.label() for fi in FamilyIncome]
    present_labels = [lbl for lbl in ordered_labels if lbl in matrix.columns]
    matrix = matrix.select(["school"] + present_labels)

    return matrix


# — 6. Sorting & ordering — #


def sort_by_income_order(
    rates: pl.DataFrame,
    income_map: dict[str, FamilyIncome],
) -> pl.DataFrame:
    """
    Sort a two-column DataFrame (income_bracket, admit_rate) by FamilyIncome order.
    """
    ordered_labels = [fi.label() for fi in FamilyIncome]
    order_map = {label: idx for idx, label in enumerate(ordered_labels)}

    return rates.sort(
        pl.col("income_bracket").replace(order_map, default=None),
        descending=False,
    )

@dataclass
class GroupStats:
    income_bracket: str
    count: int
    total_applied: int
    admit_rate: float


def compute_group_admit_rate(
    joined: pl.DataFrame,
    schools: list[School],
    intervals: list[tuple[int, int]],
    min_count: int = 0,
    decision_col: str = "decision",
    cumulative: bool = False,
) -> pl.DataFrame:
    """
    Compute admit rates aggregated over arbitrary school‐rank intervals.

    Parameters:
      - joined: DataFrame with columns ['row_id','income_bracket','school','applied','decision']
      - schools: list of School(name, rank)
      - intervals: list of (low, high) tuples, inclusive rank bounds for each bucket
                   e.g. [(1,10), (11,20), (21,50)]
      - min_count: drop any (group, bracket) with fewer than min_count samples
      - decision_col: name of decision column (compared to "Accepted")
      - cumulative: if False, use each (low,high) exactly;
                    if True, for each (low,high), use all ranks ≤ high

    Returns a wide DataFrame where:
      - each row is one group (labelled either "<low>-<high>" or "1-<high>" if cumulative)
      - each income‐bracket label is a column
      - each cell is a Struct with fields {count: int, total_applied: int, admit_rate: float}

    Only includes cells where count >= min_count.
    """
    # 1) Build (school, rank) DataFrame
    school_df = pl.DataFrame(
        {
            "school": [s.name for s in schools],
            "rank": [s.rank for s in schools],
        }
    )

    # 2) Join joined with school_df to attach rank
    merged = joined.join(school_df, on="school", how="inner")

    # 3) Keep only rows where 'applied' is not null/empty
    merged = merged.filter(pl.col("applied").is_not_null() & (pl.col("applied") != ""))

    # 4) Add boolean accepted column
    merged = merged.with_columns((pl.col(decision_col) == "Accepted").alias("accepted"))

    long_records = []
    for low, high in intervals:
        if cumulative:
            cond = pl.col("rank") <= high
            label = f"1-{high}"
        else:
            cond = (pl.col("rank") >= low) & (pl.col("rank") <= high)
            label = f"{low}-{high}"

        subset = merged.filter(cond)
        if subset.is_empty():
            continue

        # 5) Group by income_bracket → count & accepted_sum
        grp = (
            subset.group_by("income_bracket")
            .agg([pl.len().alias("count"), pl.sum("accepted").alias("accepted_sum")])
            .filter(pl.col("count") >= min_count)
        )
        if grp.is_empty():
            continue

        # 6) Compute total_applied for this group
        total_applied = grp.select(pl.sum("count").alias("total_applied"))[
            "total_applied"
        ][0]

        # 7) Build a record for each bracket
        for row in grp.iter_rows(named=True):
            income_br = row["income_bracket"]
            count = row["count"]
            accepted_sum = row["accepted_sum"]
            admit_rate = (accepted_sum / count * 100) if count > 0 else 0.0

            long_records.append(
                {
                    "group_label": label,
                    "income_bracket": income_br,
                    "count": count,
                    "total_applied": total_applied,
                    "admit_rate": admit_rate,
                }
            )

    if not long_records:
        return pl.DataFrame([], schema=[])

    # 8) Build long-format DataFrame
    df_long = pl.DataFrame(long_records)

    # 9) Assemble a Struct column for count/total_applied/admit_rate
    df_with_struct = df_long.with_columns(
        pl.struct(["count", "total_applied", "admit_rate"]).alias("stats")
    )

    # 10) Pivot so each income_bracket becomes a Struct column
    df_wide = df_with_struct.pivot(
        on="income_bracket",
        index="group_label",
        values="stats",
        aggregate_function="first",
    )

    # 11) Reorder columns: group_label first, then income brackets by FamilyIncome order
    ordered_labels = [fi.label() for fi in FamilyIncome]
    present_labels = [lbl for lbl in ordered_labels if lbl in df_wide.columns]
    return df_wide.select(["group_label"] + present_labels)


# — 7. Plotting — #


def plot_admit_rate(
    rates_sorted: pl.DataFrame,
    income_order: list[str],
    title: str = "Admission Rate by Income Bracket",
) -> go.Figure:
    """
    Scatter plot of admit_rate vs. income_bracket.
    """
    fig = px.scatter(
        rates_sorted,
        x="income_bracket",
        y="admit_rate",
        title=title,
        labels={
            "income_bracket": "Income Bracket",
            "admit_rate": "Admission Rate (%)",
        },
        category_orders={"income_bracket": income_order},
    )
    return fig


def plot_admit_rate_matrix(
    matrix: pl.DataFrame,
    income_order: list[str],
    title: str = "Admission Rate by Income Bracket (Lines per School)",
) -> plotly.graph_objs._figure.Figure:
    """
    Given a Polars DataFrame `matrix` with columns:
      - 'school'
      - one column per income‐bracket label (e.g. "Less than $20,000", "$20,000 - $45,000", …)
    produce a Plotly line chart where:
      • each line corresponds to one school
      • x‐axis = income_bracket (in the specified order)
      • y‐axis = admit_rate (%)

    Steps:
      1. Melt the wide `matrix` into long form: columns = [school, income_bracket, admit_rate].
      2. Convert to pandas and call px.line(..., x='income_bracket', y='admit_rate', color='school').
      3. Use `category_orders` to enforce the given income_order on the x‐axis.
    """
    # 1) Identify all bracket columns (everything except 'school')
    bracket_cols = [c for c in matrix.columns if c != "school"]

    # 2) Melt into long form
    long_df = matrix.melt(
        id_vars="school",
        value_vars=bracket_cols,
        variable_name="income_bracket",
        value_name="admit_rate",
    )

    # 3) Convert to pandas for Plotly Express

    # 4) Build line plot
    fig = px.line(
        long_df,
        x="income_bracket",
        y="admit_rate",
        color="school",
        title=title,
        labels={
            "income_bracket": "Income Bracket",
            "admit_rate": "Admission Rate (%)",
            "school": "School",
        },
        category_orders={"income_bracket": income_order},
    )

    # 5) Optionally rotate x‐ticks for readability, if there are many brackets
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        legend_title_text="School",
    )

    return fig


def plot_grouped_admit_rate_wide(
    wide_df: pl.DataFrame,
    income_order: list[str],
    title: str = "Grouped Admission Rates by Income Bracket",
) -> go.Figure:
    """
    Given a wide Polars DataFrame `wide_df` where:
      - each row is a group_label
      - each income-bracket column is a Struct with fields {count, total_applied, admit_rate}
    this function:
      1) Renames each struct’s fields to include the bracket name as suffix
      2) Unnests each struct column individually, avoiding duplicate column names
      3) Uses Polars’ melt to pivot admit_rate columns into long form
      4) Plots a Plotly line chart directly from Polars, with:
           • x = income_bracket (ordered by `income_order`)
           • y = admit_rate
           • color = group_label
    """
    # 1) Identify all struct columns (all except "group_label")
    struct_cols = [c for c in wide_df.columns if c != "group_label"]
    if not struct_cols:
        return go.Figure()

    df = wide_df

    # 2) For each struct column, rename its fields to avoid collisions
    for bracket in struct_cols:
        new_fields = [
            f"{bracket}_count",
            f"{bracket}_total_applied",
            f"{bracket}_admit_rate",
        ]
        df = df.with_columns(
            df[bracket].struct.rename_fields(new_fields).alias(bracket)
        )

    # 3) Unnest each struct column one by one
    for bracket in struct_cols:
        df = df.unnest(bracket)

    # 4) Identify all admit_rate columns (they end with "_admit_rate")
    admit_rate_cols = [col for col in df.columns if col.endswith("_admit_rate")]

    # 5) Melt admit_rate columns into long form using Polars
    df_long = df.melt(
        id_vars=["group_label"],
        value_vars=admit_rate_cols,
        variable_name="income_bracket_field",
        value_name="admit_rate",
    )

    # 6) Extract raw income_bracket by stripping the "_admit_rate" suffix
    df_long = df_long.with_columns(
        pl.col("income_bracket_field")
        .str.replace("_admit_rate$", "", literal=False)
        .alias("income_bracket")
    )

    # 7) Cast income_bracket to categorical for plotly ordering
    df_long = df_long.with_columns(
        pl.col("income_bracket").cast(pl.Categorical).alias("income_bracket")
    )

    # 8) Plot using Plotly Express directly from Polars
    fig = px.line(
        df_long.to_pandas(),  # Plotly Express supports Polars too; if using a version that supports it, pass df_long directly instead of .to_pandas()
        x="income_bracket",
        y="admit_rate",
        color="group_label",
        title=title,
        labels={
            "income_bracket": "Income Bracket",
            "admit_rate": "Admission Rate (%)",
            "group_label": "Rank Group",
        },
        category_orders={"income_bracket": income_order},
    )

    # 9) Rotate x-axis labels for readability
    fig.update_layout(xaxis=dict(tickangle=-45), legend_title_text="Rank Group")

    return fig

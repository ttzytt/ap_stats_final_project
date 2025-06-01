import polars as pl
from dataclasses import dataclass
from organization import School, FamilyIncome


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

import polars as pl
from dataclasses import dataclass
from organization import School, FamilyIncome
from scipy.stats import chi2_contingency

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


@dataclass
class Chi2Stats:
    statistic: float
    p_value: float
    dof: int
    expected_freq: list[list[float]]


def compute_chi2_by_group(
    joined: pl.DataFrame,
    schools: list,
    intervals: list[tuple[int, int]],
    decision_col: str = "decision",
    cumulative: bool = False,
) -> pl.DataFrame:
    school_df = pl.DataFrame(
        {"school": [s.name for s in schools], "rank": [s.rank for s in schools]}
    )

    merged = joined.join(school_df, on="school", how="inner")
    merged = merged.filter(pl.col("applied").is_not_null() & (pl.col("applied") != ""))
    merged = merged.with_columns((pl.col(decision_col) == "Accepted").alias("accepted"))

    results = []

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

        contingency = (
            subset.group_by(["income_bracket", "accepted"])
            .agg(pl.len().alias("n"))
            .pivot(
                on="accepted",
                index="income_bracket",
                values="n",
                aggregate_function="first",
            )
            .fill_null(0)
        )

        print(contingency)

        for truth_val in ["true", "false"]:
            if truth_val not in contingency.columns:
                contingency = contingency.with_columns(pl.lit(0).alias(truth_val))

        observed = (
            contingency.sort("income_bracket").select(["false", "true"]).to_numpy()
        )
        chi2, p_val, dof, expected = chi2_contingency(observed)
        result = Chi2Stats(
            statistic=chi2,
            p_value=p_val,
            dof=dof,
            expected_freq=expected.tolist(),
        )

        results.append(
            {
                "group_label": label,
                "chi2": result.__dict__,
            }
        )

    return pl.DataFrame(results)


def generate_intervals_by_applicants(
    joined: pl.DataFrame, schools: list[School], first_interval: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Partition ranks into consecutive intervals so that each interval
    has (approximately) the same total number of applicants as the first_interval.

    Parameters:
      - joined: Polars DataFrame with at least columns ["school", "applied"]
                (one row per applicant‐to‐school record).
      - schools: list of School(name, rank), sorted ascending by rank.
      - first_interval: (low, high) for the first group, e.g. (1, 5).

    Returns:
      A tuple of:
        - intervals: list of (low, high) rank‐intervals (first is first_interval).
        - counts:    list of applicant counts for each corresponding interval.
    """
    # 1) Build (school → rank) mapping, then join & filter out blank 'applied'
    school_df = pl.DataFrame(
        {
            "school": [s.name for s in schools],
            "rank": [s.rank for s in schools],
        }
    )
    merged = joined.join(school_df, on="school", how="inner").filter(
        pl.col("applied").is_not_null() & (pl.col("applied") != "")
    )

    # 2) Count applicants per rank
    rank_counts_df = merged.group_by("rank").agg(pl.count().alias("count"))
    rank_counts = {
        row["rank"]: row["count"] for row in rank_counts_df.iter_rows(named=True)
    }

    # 3) Sorted list of ranks present in `schools`
    sorted_ranks = sorted(s.rank for s in schools)

    def count_for_rank(r: int) -> int:
        return rank_counts.get(r, 0)

    # 4) Compute target = total applicants in first_interval
    low0, high0 = first_interval
    target = sum(count_for_rank(r) for r in sorted_ranks if low0 <= r <= high0)

    intervals: list[tuple[int, int]] = [first_interval]
    counts: list[int] = []
    first_count = sum(count_for_rank(r) for r in sorted_ranks if low0 <= r <= high0)
    counts.append(first_count)

    # 5) Build subsequent intervals
    start = high0 + 1
    max_rank = max(sorted_ranks)

    while start <= max_rank:
        cumulative = 0
        best_diff = None
        chosen_end = start

        for r in sorted_ranks:
            if r < start:
                continue
            if r > max_rank:
                break
            cumulative += count_for_rank(r)
            diff = abs(cumulative - target)

            if best_diff is None or diff < best_diff:
                best_diff = diff
                chosen_end = r

            if cumulative >= target:
                break

        if cumulative < target:
            chosen_end = max_rank

        intervals.append((start, chosen_end))
        interval_count = sum(
            count_for_rank(r) for r in sorted_ranks if start <= r <= chosen_end
        )
        counts.append(interval_count)

        start = chosen_end + 1

    return intervals, counts

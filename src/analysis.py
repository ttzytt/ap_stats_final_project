import polars as pl
import numpy as np
from dataclasses import dataclass
from scipy.stats import chi2_contingency, pearsonr
from organization import School, FamilyIncome


# —————— Helper Functions —————— #


def _attach_rank_and_filter(
    joined: pl.DataFrame, schools: list[School]
) -> pl.DataFrame:
    """
    Join `joined` with a school→rank mapping and filter out rows where 'applied' is null or empty.
    Does NOT create the 'accepted' column—just attaches 'rank'.
    """
    school_df = pl.DataFrame(
        {
            "school": [s.name for s in schools],
            "rank": [s.rank for s in schools],
        }
    )
    return joined.join(school_df, on="school", how="inner").filter(
        pl.col("applied").is_not_null() & (pl.col("applied") != "")
    )


def _add_accepted_flag(
    df: pl.DataFrame, decision_col: str = "decision"
) -> pl.DataFrame:
    """
    Create a boolean 'accepted' column where True if `decision_col` == "Accepted".
    """
    return df.with_columns((pl.col(decision_col) == "Accepted").alias("accepted"))


def _income_bracket_counts(
    subset: pl.DataFrame, decision_col: str = "decision"
) -> pl.DataFrame:
    """
    Given a `subset` DataFrame with at least columns ['income_bracket', decision_col],
    return a DataFrame with columns:
      - 'income_bracket'
      - 'n_total'      (count of rows in that bracket)
      - 'n_accepted'   (count where decision == "Accepted")
      - 'admit_rate'   (percentage: 100 * n_accepted / n_total)

    Only includes rows where income_bracket is non-null, non-empty, and matches FamilyIncome labels.
    """
    valid_labels = [fi.label() for fi in FamilyIncome]

    return (
        subset.with_columns((pl.col(decision_col) == "Accepted").alias("accepted_flag"))
        .group_by("income_bracket")
        .agg([pl.count().alias("n_total"), pl.sum("accepted_flag").alias("n_accepted")])
        .filter(pl.col("income_bracket").is_in(valid_labels))
        .with_columns(
            (pl.col("n_accepted") / pl.col("n_total") * 100).alias("admit_rate")
        )
    )


# —————— 0. Admit-Rate by Income —————— #


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


# —————— 1. Admit-Rate Matrix by School & Income —————— #


def compute_admit_rate_matrix(
    joined: pl.DataFrame,
    decision_col: str = "decision",
) -> pl.DataFrame:
    """
    Pivot so that rows = 'school', columns = 'income_bracket', cells = % accepted.
    """
    temp = _add_accepted_flag(joined, decision_col)

    matrix = temp.pivot(
        on="income_bracket",
        index="school",
        values="accepted",
        aggregate_function="mean",
    )

    bracket_cols = [c for c in matrix.columns if c != "school"]
    matrix = matrix.with_columns([(pl.col(c) * 100).alias(c) for c in bracket_cols])

    ordered_labels = [fi.label() for fi in FamilyIncome]
    present_labels = [lbl for lbl in ordered_labels if lbl in matrix.columns]
    return matrix.select(["school"] + present_labels)


# —————— 2. Sorting by Income Order —————— #


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


# —————— 3. Grouped Admit-Rate over School-Intervals —————— #


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

    Returns a wide DataFrame where each row is one rank‐group and each income‐bracket
    is a Struct column with fields {count, total_applied, admit_rate}.
    """
    # Preprocess: attach rank & filter, then add 'accepted'
    merged = _add_accepted_flag(_attach_rank_and_filter(joined, schools), decision_col)

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

        # Group by income_bracket → count & accepted_sum
        grp = (
            subset.group_by("income_bracket")
            .agg([pl.len().alias("count"), pl.sum("accepted").alias("accepted_sum")])
            .filter(pl.col("count") >= min_count)
        )
        if grp.is_empty():
            continue

        total_applied = grp.select(pl.sum("count").alias("total_applied"))[
            "total_applied"
        ][0]

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

    df_long = pl.DataFrame(long_records)
    df_with_struct = df_long.with_columns(
        pl.struct(["count", "total_applied", "admit_rate"]).alias("stats")
    )
    df_wide = df_with_struct.pivot(
        on="income_bracket",
        index="group_label",
        values="stats",
        aggregate_function="first",
    )

    ordered_labels = [fi.label() for fi in FamilyIncome]
    present_labels = [lbl for lbl in ordered_labels if lbl in df_wide.columns]
    return df_wide.select(["group_label"] + present_labels)


# —————— 4. Chi-Square by Group —————— #


@dataclass
class Chi2Stats:
    statistic: float
    p_value: float
    dof: int
    expected_freq: list[list[float]]


def compute_chi2_by_group(
    joined: pl.DataFrame,
    schools: list[School],
    intervals: list[tuple[int, int]],
    decision_col: str = "decision",
    cumulative: bool = False,
) -> pl.DataFrame:
    """
    For each rank‐group interval, perform a chi‐square test of independence between
    admission outcome and income bracket. Returns DataFrame with columns:
      - group_label: str
      - chi2: struct {statistic, p_value, dof, expected_freq}
    """
    merged = _add_accepted_flag(_attach_rank_and_filter(joined, schools), decision_col)

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

        for truth_val in ["true", "false"]:
            if truth_val not in contingency.columns:
                contingency = contingency.with_columns(pl.lit(0).alias(truth_val))

        observed = (
            contingency.sort("income_bracket").select(["false", "true"]).to_numpy()
        )
        chi2, p_val, dof, expected = chi2_contingency(observed)

        stats = Chi2Stats(
            statistic=chi2,
            p_value=p_val,
            dof=dof,
            expected_freq=expected.tolist(),
        )
        results.append({"group_label": label, "chi2": stats.__dict__})

    return pl.DataFrame(results)


# —————— 5. Generate Intervals by Applicants —————— #


def generate_intervals_by_applicants(
    joined: pl.DataFrame, schools: list[School], first_interval: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Partition ranks into consecutive intervals so that each interval
    has (approximately) the same total number of applicants as the first_interval.
    Returns (intervals, counts).
    """
    merged = _attach_rank_and_filter(joined, schools)

    rank_counts_df = merged.group_by("rank").agg(pl.count().alias("count"))
    rank_counts = {
        row["rank"]: row["count"] for row in rank_counts_df.iter_rows(named=True)
    }

    sorted_ranks = sorted(s.rank for s in schools)

    def count_for_rank(r: int) -> int:
        return rank_counts.get(r, 0)

    low0, high0 = first_interval
    target = sum(count_for_rank(r) for r in sorted_ranks if low0 <= r <= high0)

    intervals: list[tuple[int, int]] = [first_interval]
    counts: list[int] = []
    first_count = sum(count_for_rank(r) for r in sorted_ranks if low0 <= r <= high0)
    counts.append(first_count)

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


# —————— 6. Income-Admit Correlation by Group —————— #


@dataclass
class CorrelationStats:
    r_value: float
    p_value: float


def compute_income_admit_corr_by_group(
    joined: pl.DataFrame,
    schools: list[School],
    intervals: list[tuple[int, int]],
    decision_col: str = "decision",
    cumulative: bool = False,
) -> pl.DataFrame:
    """
    For each rank‐group interval, compute Pearson r between:
      • X = bracket_index (0,1,2,…)
      • Y = admit_rate (%) in that bracket = (accepted_count / total_count) * 100
    Returns DataFrame with columns:
      - group_label: str
      - corr: struct { r_value, p_value }
    """
    merged = _attach_rank_and_filter(joined, schools)

    income_order = [fi.label() for fi in FamilyIncome]
    results: list[dict] = []

    for low, high in intervals:
        if cumulative:
            label = f"1-{high}"
            cond = pl.col("rank") <= high
        else:
            label = f"{low}-{high}"
            cond = (pl.col("rank") >= low) & (pl.col("rank") <= high)

        subset = merged.filter(cond)
        if subset.is_empty():
            continue

        bracket_counts = _income_bracket_counts(subset, decision_col)
        if bracket_counts.is_empty():
            continue

        pdf = bracket_counts.to_pandas()
        x_list, y_list = [], []
        for idx, label_str in enumerate(income_order):
            sub = pdf[pdf["income_bracket"] == label_str]
            if not sub.empty:
                x_list.append(idx)
                y_list.append(sub["admit_rate"].iloc[0])

        if len(x_list) < 2:
            continue

        x_arr = np.array(x_list)
        y_arr = np.array(y_list)
        try:
            r_val, p_val = pearsonr(x_arr, y_arr)
        except Exception:
            r_val, p_val = 0.0, 1.0

        stats = CorrelationStats(r_value=float(r_val), p_value=float(p_val))
        results.append({"group_label": label, "corr": stats.__dict__})

    return pl.DataFrame(results)

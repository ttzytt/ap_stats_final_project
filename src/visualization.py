import plotly.graph_objects as go
import plotly.express as px
import polars as pl
import numpy as np 
from organization import School, FamilyIncome
import pandas as pd

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
) -> go.Figure:
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
      - each income-bracket column is a Struct with fields:
          {count, total_applied, admit_rate, lower_ci, upper_ci, conf_level}

    This function:
      1) Renames each struct’s fields to include the bracket name as suffix
      2) Unnests each struct column one by one
      3) Gathers admit_rate, lower_ci, upper_ci into long form
      4) Plots a Plotly line chart with error bars (CI) for each rank group
    """
    # 1) Identify all struct columns (all except "group_label")
    struct_cols = [c for c in wide_df.columns if c != "group_label"]
    if not struct_cols:
        return go.Figure()

    df = wide_df

    # 2) Rename each struct’s fields to avoid collisions
    for bracket in struct_cols:
        new_fields = [
            f"{bracket}_count",
            f"{bracket}_total_applied",
            f"{bracket}_admit_rate",
            f"{bracket}_lower_ci",
            f"{bracket}_upper_ci",
            f"{bracket}_conf_level",
        ]
        df = df.with_columns(
            df[bracket].struct.rename_fields(new_fields).alias(bracket)
        )

    # 3) Unnest each struct column one by one
    for bracket in struct_cols:
        df = df.unnest(bracket)

    # 4) Identify all admit_rate, lower_ci, upper_ci columns
    admit_rate_cols = [col for col in df.columns if col.endswith("_admit_rate")]
    lower_cols = [col for col in df.columns if col.endswith("_lower_ci")]
    upper_cols = [col for col in df.columns if col.endswith("_upper_ci")]

    # 5) Build a long DataFrame with fields: group_label, income_bracket, admit_rate, lower_ci, upper_ci
    #    We know each bracket name is the prefix before "_admit_rate", so we can extract bracket.
    records = []
    for row in df.iter_rows(named=True):
        grp = row["group_label"]
        for rate_col, low_col, high_col in zip(admit_rate_cols, lower_cols, upper_cols):
            bracket = rate_col.replace("_admit_rate", "")
            records.append(
                {
                    "group_label": grp,
                    "income_bracket": bracket,
                    "admit_rate": row[rate_col],
                    "lower_ci": row[low_col],
                    "upper_ci": row[high_col],
                }
            )

    df_long = pl.DataFrame(records)

    # 6) Cast income_bracket to categorical for ordering
    df_long = df_long.with_columns(
        pl.col("income_bracket").cast(pl.Categorical).alias("income_bracket")
    )

    # 7) Convert to pandas for Plotly
    pdf = df_long.to_pandas()

    # 8) Plot: for each group_label, one trace with line + markers + error bars
    fig = go.Figure()
    for grp in pdf["group_label"].unique():
        sub = pdf[pdf["group_label"] == grp]
        # Ensure order by income_order
        sub = sub.set_index("income_bracket").loc[income_order].reset_index()
        hovertemplate = """
        Group: %{customdata[0]}<br>
        Income Bracket: %{x}<br>
        Admission Rate: %{y:.3f}%<br>
        CI: (%{customdata[1]:.3f}, %{customdata[2]:.3f}), range=%{customdata[3]:.3f}<br>
        """
        fig.add_trace(
            go.Scatter(
                x=sub["income_bracket"],
                y=sub["admit_rate"],
                customdata=np.column_stack(
                    (np.repeat(grp, len(sub)), sub["lower_ci"], sub["upper_ci"], 
                     sub["upper_ci"] - sub["lower_ci"])
                ),
                mode="lines+markers",
                name=grp,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=sub["upper_ci"] - sub["admit_rate"],
                    arrayminus=sub["admit_rate"] - sub["lower_ci"],
                    thickness=1,
                    width=3,
                ),
                hovertemplate=hovertemplate,
            )
        )

    # 9) Layout tweaks
    fig.update_layout(
        title=title,
        xaxis_title="Income Bracket",
        yaxis_title="Admission Rate (%)",
        xaxis=dict(tickangle=-45),
        legend_title_text="Rank Group",
    )

    return fig


def plot_chi2_pvalues(
    df: pl.DataFrame,
    title: str = "Chi-Square Test p-values by School Group",
) -> go.Figure:
    """
    Given a Polars DataFrame with columns:
      - 'group_label': str
      - 'chi2': struct with fields including 'p_value'

    Returns a Plotly line chart where:
      - x-axis = group_label (ordered numerically)
      - y-axis = p_value
    """
    # Extract 'p_value' field
    df_extracted = df.with_columns(pl.col("chi2").struct.field("p_value"))

    # Extract order from group_label: assume format "low-high"
    def parse_low(group_label: str) -> int:
        return int(group_label.split("-")[0])

    # Sort and get correct x-axis order
    ordered_labels = sorted(df_extracted["group_label"].to_list(), key=parse_low)

    # Convert to pandas for plotting
    pdf = df_extracted.to_pandas()

    # Plot with Plotly Express
    fig = px.line(
        pdf,
        x="group_label",
        y="p_value",
        title=title,
        labels={
            "group_label": "Rank Group",
            "p_value": "p-value",
        },
        category_orders={"group_label": ordered_labels},
        markers=True,
    )

    fig.update_layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="p-value"),
        legend_title_text="",
    )

    return fig


def plot_applicant_counts(
    joined: pl.DataFrame,
    schools: list[School],
    intervals: list[tuple[int, int]],
    plot_type: str = "line",
    fit_type: str = "linear"
) -> go.Figure:
    """
    Plots number of applicants per rank‐group interval. Offers:
      - plot_type: "line" or "scatter"
      - fit_type: "linear" or "exponential" (applies only if plot_type == "scatter")

    If fit_type == "exponential", fits y = a * exp(b * x) on numeric index.
    """
    # 1) Attach rank
    school_df = pl.DataFrame({
        "school": [s.name for s in schools],
        "rank":   [s.rank for s in schools],
    })
    merged = (
        joined
        .join(school_df, on="school", how="inner")
        .filter(pl.col("applied").is_not_null() & (pl.col("applied") != ""))
    )

    # 2) Count applicants per interval
    labels: list[str] = []
    counts: list[int] = []
    for (low, high) in intervals:
        label = f"{low}-{high}"
        subset = merged.filter((pl.col("rank") >= low) & (pl.col("rank") <= high))
        cnt = subset.height
        labels.append(label)
        counts.append(cnt)

    x_idx = np.arange(len(labels))
    y = np.array(counts, dtype=float)

    fig = go.Figure()

    if plot_type == "line":
        fig.add_trace(go.Scatter(
            x=labels, y=counts,
            mode="lines+markers",
            name="Applicants"
        ))
    else:  # scatter + fit
        fig.add_trace(go.Scatter(
            x=labels, y=counts,
            mode="markers",
            name="Applicants"
        ))

        if fit_type == "linear":
            # Linear fit: y = m x + b
            m, b = np.polyfit(x_idx, y, 1)
            y_pred = m * x_idx + b
            # R²
            corr_mat = np.corrcoef(y, y_pred)
            r2 = corr_mat[0,1]**2 if corr_mat.size > 1 else 0.0
            fig.add_trace(go.Scatter(
                x=labels, y=y_pred,
                mode="lines", name="Linear fit"
            ))
            eq_text = f"y = {m:.2f}·x + {b:.2f} (R² = {r2:.3f})"
        else:
            # Exponential fit: y = a * exp(b x) → ln(y) = ln(a) + b x
            # Filter out zero or negative y to safely take log
            mask = y > 0
            x_pos = x_idx[mask]
            y_pos = y[mask]
            ln_y = np.log(y_pos)
            b, ln_a = np.polyfit(x_pos, ln_y, 1)
            a = np.exp(ln_a)
            y_pred_full = a * np.exp(b * x_idx)
            # Compute R² on original y (for positive region or overall? We'll use positive only)
            y_pred_pos = a * np.exp(b * x_pos)
            corr_mat = np.corrcoef(y_pos, y_pred_pos)
            r2 = corr_mat[0,1]**2 if corr_mat.size > 1 else 0.0
            fig.add_trace(go.Scatter(
                x=labels, y=y_pred_full,
                mode="lines", name="Exponential fit"
            ))
            eq_text = f"y = {a:.2f}·e^({b:.2f}·x) (R² = {r2:.3f})"

        fig.add_annotation(
            x=0.05, y=0.95, xref="paper", yref="paper",
            text=eq_text, showarrow=False, font=dict(size=12)
        )

    fig.update_layout(
        title="Number of Applicants by Rank Group",
        xaxis_title="Rank Group",
        yaxis_title="Number of Applicants",
        xaxis=dict(tickangle=-45)
    )
    return fig


def plot_applicant_bar_scaled(
    intervals: list[tuple[int, int]],
    counts: list[int],
    title: str = "Applicants by Rank Group (Bar Width ∝ Group Size)",
) -> go.Figure:
    """
    Draws a bar chart where:
      - each bar’s height = applicant count for that interval,
      - each bar’s width  ∝ (high - low + 1) = number of ranks in the interval.

    The x-axis is continuous; tick labels are the interval strings.
    """
    # Compute midpoints and widths
    mids: list[float] = []
    widths: list[int] = []
    labels: list[str] = []

    for (low, high), cnt in zip(intervals, counts):
        mid = (low + high) / 2
        w = high - low + 1
        label = f"{low}-{high}"
        mids.append(mid)
        widths.append(w)
        labels.append(label)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=mids, y=counts, width=widths, marker_color="steelblue", name="Applicants"
        )
    )

    # Set tick positions at mids with text = labels
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode="array", tickvals=mids, ticktext=labels, title="Rank Group"
        ),
        yaxis=dict(title="Number of Applicants"),
    )

    return fig


def plot_correlation_by_group(
    df: pl.DataFrame,
    title: str = "Income-Admit Correlation by School Group",
) -> go.Figure:
    """
    Given a Polars DataFrame with columns:
      - 'group_label': str
      - 'corr': struct with fields 'r_value' and 'p_value'

    Returns a Plotly chart of 'r_value' vs 'group_label', with points colored
    by significance (p_value < 0.05). The connecting line follows numeric
    rank‐group order even without Polars categorical ordering.
    """
    # 1) Extract r_value and p_value from the Struct column 'corr'
    df2 = df.with_columns(
        [
            pl.col("corr").struct.field("r_value"),
            pl.col("corr").struct.field("p_value"),
        ]
    )

    # 2) Build and add a numeric key 'low_rank' = integer part before the dash in group_label
    def parse_low(label: str) -> int:
        return int(label.split("-")[0])

    df2 = df2.with_columns(pl.col("group_label").map_elements(parse_low, return_dtype=pl.Int64).alias("low_rank"))

    # 3) Sort by that numeric key
    df2 = df2.sort("low_rank")

    # 4) Convert to pandas for Plotly
    pdf = df2.to_pandas()

    # 5) Determine the exact ordered list of group_label
    ordered_labels = pdf["group_label"].tolist()

    # 6) Mark significance: p_value < 0.05 → True, else False
    pdf["significant"] = pdf["p_value"] < 0.05
    print(pdf['p_value'])
    # 7) Create scatter, passing category_orders to force x‐axis in the correct sequence
    fig = px.scatter(
        pdf,
        x="group_label",
        y="r_value",
        color="significant",
        color_discrete_map={True: "firebrick", False: "steelblue"},
        category_orders={"group_label": ordered_labels},
        title=title,
        labels={"group_label": "Rank Group", "r_value": "r", "p_value": "p"},
        hover_data=["p_value"],
        # markers=True,
    )

    # 8) Add a connecting dashed line in the same order
    fig.add_trace(
        go.Scatter(
            x=ordered_labels,
            y=pdf["r_value"].tolist(),
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )
    )

    # 9) Annotate each point with its rounded r-value
    for _, row in pdf.iterrows():
        fig.add_annotation(
            x=row["group_label"],
            y=row["r_value"],
            text=f"{row['r_value']:.2f}",

            showarrow=False,
            yshift=10,
            font=dict(size=10),
        )

    # 10) Final layout adjustments
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="Pearson r"),
        legend_title_text="p < 0.05",
    )

    return fig

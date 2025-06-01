import plotly.graph_objects as go
import plotly.express as px
import polars as pl
import numpy as np 
from organization import School, FamilyIncome

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
) -> go.Figure:
    """
    Plots the number of applicants per rank‐group interval.

    Inputs:
      - joined: Polars DataFrame with columns ["school","applied",…]
      - schools: list of School(name, rank)
      - intervals: list of (low, high) rank intervals
      - plot_type: either "line" or "scatter"

    Behavior:
      1. Attach `rank` to each row, filter out null/empty `applied`.
      2. For each (low,high) in `intervals`, count how many rows have
         rank ∈ [low, high].
      3. Build a small DataFrame with columns ["group_label", "count"].
      4. If plot_type == "line": draw a lines+markers plot of `count`
         vs. `group_label`.
         If plot_type == "scatter":
           • draw a scatter of `count` vs. `group_label`
           • fit a simple linear regression on the numeric index (0,1,2,…)
             vs. `count`
           • overlay the regression line, compute R², annotate equation & R².
    """
    # 1) Attach rank → same as in generate_intervals_by_applicants
    school_df = pl.DataFrame(
        {
            "school": [s.name for s in schools],
            "rank": [s.rank for s in schools],
        }
    )
    merged = joined.join(school_df, on="school", how="inner").filter(
        pl.col("applied").is_not_null() & (pl.col("applied") != "")
    )

    # 2) Count applicants per interval
    labels: list[str] = []
    counts: list[int] = []
    for low, high in intervals:
        label = f"{low}-{high}"
        subset = merged.filter((pl.col("rank") >= low) & (pl.col("rank") <= high))
        cnt = subset.height
        labels.append(label)
        counts.append(cnt)

    # 3) Build a small Polars DataFrame (if needed later)
    df_counts = pl.DataFrame({"group_label": labels, "count": counts})

    # 4) Prepare X‐axis numeric indices for regression
    x_indices = np.arange(len(labels))
    y_values = np.array(counts, dtype=float)

    # 5) Build the Plotly Figure
    fig = go.Figure()

    if plot_type == "line":
        # Simple line+marker plot
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=counts,
                mode="lines+markers",
                name="Number of Applicants",
            )
        )
    else:
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=counts,
                mode="markers",
                name="Applicants",
            )
        )

        # Compute best‐fit line: y = m·x + b, on numeric indices
        slope, intercept = np.polyfit(x_indices, y_values, 1)
        y_pred = slope * x_indices + intercept

        # Compute R²
        r_matrix = np.corrcoef(y_values, y_pred)
        r2 = r_matrix[0, 1] ** 2 if r_matrix.size > 1 else 0.0

        # Overlay the best‐fit line
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=y_pred,
                mode="lines",
                name="Best-fit Line",
            )
        )

        # Annotate equation & R² in the corner
        eq_text = f"y = {slope:.2f}·x + {intercept:.2f}  (R-sq = {r2:.3f})"
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=eq_text,
            showarrow=False,
            font=dict(size=12),
        )

    # 6) Layout tweaks
    fig.update_layout(
        title="Number of Applicants by Rank Group",
        xaxis_title="Rank Group",
        yaxis_title="Number of Applicants",
        xaxis=dict(tickangle=-45),
    )

    return fig

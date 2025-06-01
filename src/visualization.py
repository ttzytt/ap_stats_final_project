import plotly.graph_objects as go
import plotly.express as px
import polars as pl


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

import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import logging
    import marimo as mo
    from icecream import ic
    logging.getLogger("fastexcel.types.dtype").setLevel(logging.ERROR)
    return ic, mo, pl


@app.cell
def _(pl):
    df = pl.read_excel("./data/2020.xlsx", sheet_name="Raw Data").with_row_count("row_id")
    df
    return (df,)


@app.cell
def _(df):
    print(df.head())
    return


@app.cell
def _(mo):
    mo.md(r"Tries to find the relationship between income and college admissions rate")
    return


@app.cell
def _():
    ranked_schools : list[str] = []
    # read school_rank.txt, each line is a school name, schools placed by rank
    with open("./data/school_rank.txt", "r", encoding='utf8') as f:
        for line in f:
            ranked_schools.append(line.strip())
    return (ranked_schools,)


@app.cell
def _():
    from enum import Enum
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
    return (FamilyIncome,)


@app.cell
def _(FamilyIncome, df, ic, pl, ranked_schools):
    top20 = ranked_schools[:50]

    # 2) build column names
    apply_prefix    = "Which of the following did you apply to? ["
    decision_prefix = "Select your admission decision results for each school you applied to: ["

    apply_cols = [
        f"{apply_prefix}{s}]" 
        for s in top20 
        if f"{apply_prefix}{s}]" in df.columns
    ]
    decision_cols = [
        f"{decision_prefix}{s}]" 
        for s in top20 
        if f"{decision_prefix}{s}]" in df.columns
    ]

    ic(apply_cols) 
    ic(decision_cols)
    # 3) map raw income answers into your enum
    income_col = "Approximately how much is your family's total yearly income?"
    income_map = { b.label(): b for b in FamilyIncome }
    ic(income_map)
    sel_df = df.with_columns(
        pl.col(income_col)
        .alias("income_bracket")
    ).filter(pl.col("income_bracket").is_not_null())
    return apply_cols, apply_prefix, decision_cols, decision_prefix, sel_df


@app.cell
def _(apply_cols, apply_prefix, decision_cols, decision_prefix, pl, sel_df):
    # 4) unpivot the "applied" columns
    applied = sel_df.unpivot(
        on=apply_cols,
        index=["row_id", "income_bracket"],
        variable_name="school",
        value_name="applied",
    ).with_columns(
        pl.col("school")
        .str.replace(apply_prefix, "", literal=True)  
        .str.strip_suffix("]")
        .alias("school")
    )

    # 5) unpivot the "decision" columns
    decided = sel_df.unpivot(
        on=decision_cols,
        index=["row_id", "income_bracket"],
        variable_name="school",
        value_name="decision",
    ).with_columns(
        pl.col("school")
        .str.replace(decision_prefix, "", literal=True) 
        .str.strip_suffix("]")
        .alias("school")
    )   
    applied, decided
    return applied, decided


@app.cell
def _(applied, decided, ic, pl):
    # 6) join & filter
    joined = (
        applied
        .join(decided, on=["row_id", "school", "income_bracket"], how='inner')
        .filter(pl.col("applied").is_not_null() & pl.col("decision").is_not_null())
    )

    ic(applied.shape)
    ic(decided.shape)
    ic(joined.shape)
    joined 
    return (joined,)


@app.cell
def _(joined, pl):
    rates = (
        joined.with_columns((pl.col("decision") == "Accepted").alias("accepted"))
        .group_by("income_bracket")
        .agg((pl.sum("accepted") / pl.len() * 100).alias("admit_rate"))
        .filter(
            (pl.col("income_bracket") != "x") & (pl.col("income_bracket") != "Blank")
        )
    )
    rates
    return (rates,)


@app.cell
def _(FamilyIncome, ic, pl, rates):
    income_order = [fi.label() for fi in FamilyIncome]
    order_map = {label: idx for idx, label in enumerate(income_order)}
    ic(order_map)

    rates_sorted = (
        rates.sort(
            pl.col("income_bracket").replace(order_map, default=None), 
            descending=False
        )
    )
    rates_sorted
    return income_order, rates_sorted


@app.cell
def _(ic, income_order, mo, rates_sorted):
    import plotly.express as px
    ic(income_order)
    fig = px.scatter(
        rates_sorted,
        x="income_bracket",
        y="admit_rate",
        title="Admission Rate by Income Bracket",
        labels={
            "income_bracket": "Income Bracket",
            "admit_rate": "Admission Rate (%)",
        },
        category_orders={"income_bracket": income_order},
    )

    plot = mo.ui.plotly(fig)
    plot
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo 
    import polars as pl
    import sys
    import os
    # Ensure the current directory is in the Python path
    sys.path.append(".")
    from analysis import (
        compute_admit_rate_matrix,
        compute_group_admit_rate, 
        compute_chi2_by_group,
    )

    from organization import  (
        School, 
        load_survey_data, 
        load_ranked_schools, 
        FamilyIncome, 
        build_apply_and_decision_cols,
        unpivot_applied_or_decided,
        join_applied_and_decided
    )

    from visualization import (
        plot_admit_rate, 
        plot_admit_rate_matrix, 
        plot_grouped_admit_rate_wide,
        plot_chi2_pvalues
    )

    return (
        FamilyIncome,
        build_apply_and_decision_cols,
        compute_admit_rate_matrix,
        compute_chi2_by_group,
        compute_group_admit_rate,
        join_applied_and_decided,
        load_ranked_schools,
        load_survey_data,
        mo,
        os,
        pl,
        plot_admit_rate_matrix,
        plot_grouped_admit_rate_wide,
        unpivot_applied_or_decided,
    )


@app.cell
def _(load_ranked_schools, load_survey_data, os):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    schools = load_ranked_schools(os.path.join(data_path, 'school_rank.txt'))
    survey_df = load_survey_data(os.path.join(data_path, '2020.xlsx'), 'Raw Data')
    return schools, survey_df


@app.cell
def _(schools, survey_df):
    survey_df, schools
    return


@app.cell
def _(FamilyIncome, pl, survey_df):
    income_col = "Approximately how much is your family's total yearly income?"
    income_labels = [fi.label() for fi in FamilyIncome]
    sel_df = (
        survey_df
        .with_columns(
            pl.col(income_col).alias("income_bracket")
        )
        .filter(pl.col("income_bracket").is_in(income_labels))
    )
    sel_df
    return income_labels, sel_df


@app.cell
def _(build_apply_and_decision_cols, schools, sel_df):
    apply_cols, decision_cols = build_apply_and_decision_cols(sel_df, schools)
    apply_cols, decision_cols
    return apply_cols, decision_cols


@app.cell
def _(apply_cols, sel_df, unpivot_applied_or_decided):
    apply_prefix = "Which of the following did you apply to? ["
    applied = unpivot_applied_or_decided(
        sel_df,
        cols_to_melt=apply_cols,
        prefix=apply_prefix,
        new_col_name="applied"
    )
    applied
    return (applied,)


@app.cell
def _(decision_cols, sel_df, unpivot_applied_or_decided):
    decision_prefix = (
        "Select your admission decision results for each school you applied to: ["
    )
    decided = unpivot_applied_or_decided(
        sel_df, cols_to_melt=decision_cols, prefix=decision_prefix, new_col_name="decision"
    )
    decided
    return (decided,)


@app.cell
def _(applied, decided, join_applied_and_decided):
    joined = join_applied_and_decided(applied, decided)
    joined
    return (joined,)


@app.cell
def _(compute_admit_rate_matrix, joined):
    matrix = compute_admit_rate_matrix(joined)
    matrix
    return (matrix,)


@app.cell
def _(income_labels, matrix, mo, plot_admit_rate_matrix):
    fig = plot_admit_rate_matrix(matrix, income_labels)
    fig.update_traces(visible='legendonly')
    mo.ui.plotly(fig)
    return


@app.cell
def _(compute_group_admit_rate, joined, schools):
    school_grp_interv = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 40), (41, 60), (61, 100)]
    group_admit_rate = compute_group_admit_rate(joined, schools, school_grp_interv, 20)
    group_admit_rate
    return group_admit_rate, school_grp_interv


@app.cell
def _(group_admit_rate, income_labels, mo, plot_grouped_admit_rate_wide):
    fig2 = plot_grouped_admit_rate_wide(group_admit_rate, income_labels)
    # fig2.update_traces(visible='legendonly')
    fig2.write_html("grouped_admit_rate.html")
    mo.ui.plotly(fig2)
    return


@app.cell
def _(compute_chi2_by_group, joined, school_grp_interv, schools):
    chi_sq = compute_chi2_by_group(joined, schools, school_grp_interv)
    chi_sq
    return


@app.cell
def _(
    compute_group_admit_rate,
    income_labels,
    joined,
    mo,
    plot_grouped_admit_rate_wide,
    schools,
):
    cum_group_admit_rate = compute_group_admit_rate(joined, schools, [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 70), (71, 100)], 20, cumulative = True)

    fig3 = plot_grouped_admit_rate_wide(cum_group_admit_rate, income_labels)
    # fig3.update_traces(visible='legendonly')
    mo.ui.plotly(fig3)
    return


if __name__ == "__main__":
    app.run()

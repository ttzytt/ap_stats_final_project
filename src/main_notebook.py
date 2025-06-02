import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import sys
    import os
    from plotly.subplots import make_subplots
    # Ensure the current directory is in the Python path
    sys.path.append(".")
    from analysis import (
        compute_admit_rate_matrix,
        compute_group_admit_rate,
        compute_chi2_by_group,
        generate_intervals_by_applicants,
        compute_income_admit_corr_by_group,
    )

    from organization import (
        School,
        load_survey_data,
        load_ranked_schools,
        FamilyIncome,
        build_apply_and_decision_cols,
        unpivot_applied_or_decided,
        join_applied_and_decided,
    )

    from visualization import (
        plot_admit_rate,
        plot_admit_rate_matrix,
        plot_grouped_admit_rate_wide,
        plot_chi2_pvalues,
        plot_applicant_counts,
        plot_applicant_bar_scaled,
        plot_correlation_by_group,
        show_traces,
        hide_traces,
        add_traces_to_subplot
    )
    return (
        FamilyIncome,
        add_traces_to_subplot,
        build_apply_and_decision_cols,
        compute_admit_rate_matrix,
        compute_chi2_by_group,
        compute_group_admit_rate,
        compute_income_admit_corr_by_group,
        generate_intervals_by_applicants,
        join_applied_and_decided,
        load_ranked_schools,
        load_survey_data,
        make_subplots,
        mo,
        os,
        pl,
        plot_admit_rate_matrix,
        plot_applicant_bar_scaled,
        plot_applicant_counts,
        plot_chi2_pvalues,
        plot_correlation_by_group,
        plot_grouped_admit_rate_wide,
        show_traces,
        unpivot_applied_or_decided,
    )


@app.cell
def _(load_ranked_schools, load_survey_data, os):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    schools = load_ranked_schools(os.path.join(data_path, "school_rank.txt"))
    survey_df = load_survey_data(os.path.join(data_path, "2020.xlsx"), "Raw Data")
    figs = []
    return figs, schools, survey_df


@app.cell
def _():
    import plotly.io as pio
    import plotly.graph_objects as go
    font_template = pio.templates['plotly']
    font_template.layout.font.size = 18
    pio.templates['large_font'] = font_template
    pio.templates.default = 'large_font'
    return


@app.cell
def _(schools, survey_df):
    survey_df, schools
    return


@app.cell
def _(FamilyIncome, pl, survey_df):
    income_col = "Approximately how much is your family's total yearly income?"
    income_labels = [fi.label() for fi in FamilyIncome]
    sel_df = survey_df.with_columns(
        pl.col(income_col).alias("income_bracket")
    ).filter(pl.col("income_bracket").is_in(income_labels))
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
        new_col_name="applied",
    )
    applied
    return (applied,)


@app.cell
def _(decision_cols, sel_df, unpivot_applied_or_decided):
    decision_prefix = (
        "Select your admission decision results for each school you applied to: ["
    )
    decided = unpivot_applied_or_decided(
        sel_df,
        cols_to_melt=decision_cols,
        prefix=decision_prefix,
        new_col_name="decision",
    )
    decided
    return (decided,)


@app.cell
def _(applied, decided, join_applied_and_decided):
    joined = join_applied_and_decided(applied, decided)
    joined
    return (joined,)


@app.cell
def _(figs, joined, mo, plot_applicant_counts, schools):
    fixed_interv_sz = 10
    fixed_intervs = [
        (i, i + fixed_interv_sz - 1)
        for i in range(1, len(schools) + 1, fixed_interv_sz)
    ]
    figs.append(plot_applicant_counts(joined, schools, fixed_intervs, "scatter"))
    figs[-1].write_html("./output/applicant_counts_fixed_intervals_linear_fit.html", include_plotlyjs="cdn")
    mo.ui.plotly(figs[-1])
    return (fixed_intervs,)


@app.cell
def _(figs, fixed_intervs, joined, mo, plot_applicant_counts, schools):
    figs.append(
        plot_applicant_counts(
            joined, schools, fixed_intervs, "scatter", "exponential"
        )
    )
    figs[-1].write_html("./output/applicant_counts_fixed_intervals_exp_fit.html", include_plotlyjs="cdn")
    mo.ui.plotly(figs[-1])
    return


@app.cell
def _(generate_intervals_by_applicants, joined, schools):
    sc_dyn_intervs, sc_dyn_interv_cnts = generate_intervals_by_applicants(
        joined, schools, (1, 5)
    )
    sc_dyn_intervs, sc_dyn_interv_cnts
    return sc_dyn_interv_cnts, sc_dyn_intervs


@app.cell
def _(figs, mo, plot_applicant_bar_scaled, sc_dyn_interv_cnts, sc_dyn_intervs):
    figs.append(plot_applicant_bar_scaled(sc_dyn_intervs, sc_dyn_interv_cnts))
    figs[-1].write_html("./output/scaled_applicant_bar.html", include_plotlyjs="cdn")
    mo.ui.plotly(figs[-1])
    return


@app.cell
def _(compute_admit_rate_matrix, joined):
    matrix = compute_admit_rate_matrix(joined)
    matrix
    return (matrix,)


@app.cell
def _(figs, income_labels, matrix, mo, plot_admit_rate_matrix):
    figs.append(plot_admit_rate_matrix(matrix, income_labels))
    figs[-1].update_traces(visible="legendonly")
    mo.ui.plotly(figs[-1])
    return


@app.cell
def _(compute_group_admit_rate, joined, sc_dyn_intervs, schools):
    # school_grp_interv = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 40), (41, 60), (61, 100)]
    group_admit_rate = compute_group_admit_rate(
        joined, schools, sc_dyn_intervs, 20, ci_method="normal"
    )
    group_admit_rate
    return (group_admit_rate,)


@app.cell
def _(
    add_traces_to_subplot,
    figs,
    group_admit_rate,
    income_labels,
    make_subplots,
    mo,
    plot_grouped_admit_rate_wide,
    show_traces,
):
    combined_fig = make_subplots(2, 2, subplot_titles=['1-9', '10-30', '31-73', '74-200'])

    figs.append(plot_grouped_admit_rate_wide(group_admit_rate, income_labels))
    figs[-1].write_html("./output/grouped_admit_rate_combined.html", include_plotlyjs="cdn")

    show_traces(figs[-1], ['1-5', '6-9'], invis_type=False)
    figs[-1].write_html("./output/grouped_admit_rate_1-9.html", include_plotlyjs="cdn")
    add_traces_to_subplot(combined_fig, figs[-1].data, row=1, col=1)

    figs.append(plot_grouped_admit_rate_wide(group_admit_rate, income_labels))
    show_traces(figs[-1], ['10-15', '16-24', '25-30'], invis_type=False)
    figs[-1].write_html("./output/grouped_admit_rate_10-30.html", include_plotlyjs="cdn")
    add_traces_to_subplot(combined_fig, figs[-1].data, row=1, col=2)

    figs.append(plot_grouped_admit_rate_wide(group_admit_rate, income_labels))
    show_traces(figs[-1], ['31-37', '38-56', '57-73'], invis_type=False)
    figs[-1].write_html("./output/grouped_admit_rate_31-73.html", include_plotlyjs="cdn")
    add_traces_to_subplot(combined_fig, figs[-1].data, row=2, col=1)

    figs.append(plot_grouped_admit_rate_wide(group_admit_rate, income_labels))
    show_traces(figs[-1], ['74-109', '110-200'], invis_type=False)
    figs[-1].write_html("./output/grouped_admit_rate_74-200.html", include_plotlyjs="cdn")
    add_traces_to_subplot(combined_fig, figs[-1].data, row=2, col=2)

    figs.append(combined_fig)
    figs[-1].write_html("./output/grouped_admit_rate_subplot.html", include_plotlyjs="cdn")
    mo.ui.plotly(combined_fig)
    return


@app.cell
def _(
    compute_group_admit_rate,
    figs,
    income_labels,
    joined,
    mo,
    plot_grouped_admit_rate_wide,
    schools,
):
    lg_group_admit_rate = compute_group_admit_rate(joined, schools, [(1, 200)], 20)
    figs.append(plot_grouped_admit_rate_wide(lg_group_admit_rate, income_labels))
    figs[-1].write_html("./output/lg_grouped_admit_rate.html", include_plotlyjs="cdn")
    mo.ui.plotly(figs[-1])
    return


@app.cell
def _(compute_chi2_by_group, joined, sc_dyn_intervs, schools):
    chi_sq = compute_chi2_by_group(joined, schools, sc_dyn_intervs)
    chi_sq
    return (chi_sq,)


@app.cell
def _(chi_sq, figs, mo, plot_chi2_pvalues):
    figs.append(plot_chi2_pvalues(chi_sq))
    figs[-1].write_html("./output/chi2_pvalues.html", include_plotlyjs="cdn")
    mo.ui.plotly(figs[-1])
    return


@app.cell
def _(compute_income_admit_corr_by_group, joined, sc_dyn_intervs, schools):
    correlation = compute_income_admit_corr_by_group(
        joined, schools, sc_dyn_intervs
    )
    correlation
    return (correlation,)


@app.cell
def _(correlation, figs, mo, plot_correlation_by_group):
    figs.append(plot_correlation_by_group(correlation))
    figs[-1].write_html("./output/correlation_by_group.html", include_plotlyjs="cdn")
    mo.ui.plotly(figs[-1])
    return


@app.cell
def _(
    compute_group_admit_rate,
    figs,
    income_labels,
    joined,
    mo,
    plot_grouped_admit_rate_wide,
    schools,
):
    cum_group_admit_rate = compute_group_admit_rate(
        joined,
        schools,
        [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 70), (71, 100)],
        20,
        cumulative=True,
    )

    figs.append(plot_grouped_admit_rate_wide(cum_group_admit_rate, income_labels))
    # fig3.update_traces(visible='legendonly')
    mo.ui.plotly(figs[-1])
    return


if __name__ == "__main__":
    app.run()

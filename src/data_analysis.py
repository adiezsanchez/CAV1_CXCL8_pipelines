import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import re
    from pathlib import Path

    import pandas as pd
    import plotly.express as px
    import seaborn as sns

    return Path, pd, re, sns


@app.cell
def _(Path):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = PROJECT_ROOT / "results"
    PROCESSED_RESULTS_DIR = PROJECT_ROOT / "processed_results"
    COMBINED_CSV_PATH = PROCESSED_RESULTS_DIR / "combined_features.csv"
    INTENSITY_COLS = ["GFP_CAFs_mean_int", "AF647_CXCL8_mean_int"]
    GENOTYPE_PALETTE = {
        "scr": "#2ca02c",     # green
        "shCAV1": "#ff7f0e",  # orange
    }
    GENOTYPE_ORDER = ["scr", "shCAV1"]
    return (
        COMBINED_CSV_PATH,
        GENOTYPE_ORDER,
        GENOTYPE_PALETTE,
        INTENSITY_COLS,
        PROCESSED_RESULTS_DIR,
        RESULTS_DIR,
    )


@app.cell
def _(COMBINED_CSV_PATH, PROCESSED_RESULTS_DIR, RESULTS_DIR, pd, re):
    def assign_genotype(image_name: str) -> str | None:
        name = str(image_name)
        name_lower = name.lower()

        if "knockdown" in name_lower or re.search(r"sh", name, re.IGNORECASE):
            return "shCAV1"
        if (
            re.search(r"wt", name, re.IGNORECASE)
            or "wildtype" in name_lower
            or re.search(r"scr", name, re.IGNORECASE)
        ):
            return "scr"
        return None

    def add_genotype_column(df: pd.DataFrame) -> pd.DataFrame:
        annotated_df = df.copy()
        genotype = annotated_df["lif_image_name"].map(assign_genotype)
        insert_at = annotated_df.columns.get_loc("lif_image_name") + 1
        annotated_df.insert(insert_at, "genotype", genotype)
        return annotated_df

    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{RESULTS_DIR}'. "
            "Run `pixi run loop` to generate per-image feature tables first."
        )

    combined_df = pd.concat(
        (pd.read_csv(csv_path) for csv_path in csv_files),
        ignore_index=True,
    )
    combined_df = add_genotype_column(combined_df)

    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(COMBINED_CSV_PATH, index=False)

    genotype_counts = combined_df["genotype"].value_counts(dropna=False)
    print(
        f"Combined {len(csv_files)} file(s) into {COMBINED_CSV_PATH} "
        f"({len(combined_df)} rows)"
    )
    print(f"Genotype counts:\n{genotype_counts.to_string()}")
    return


@app.cell
def _(COMBINED_CSV_PATH, mo, pd):
    features_df = pd.read_csv(COMBINED_CSV_PATH)

    container_options = ["All"] + sorted(
        features_df["lif_container_id"].astype(str).unique().tolist()
    )
    lif_container_radio = mo.ui.radio(
        options=container_options,
        value=container_options[0],
        label="Filter by LIF container",
    )

    lif_container_radio
    return features_df, lif_container_radio


@app.cell
def _(INTENSITY_COLS, features_df, lif_container_radio):
    selected_container = lif_container_radio.value
    if selected_container == "All":
        filtered_df = features_df.copy()
    else:
        filtered_df = features_df[
            features_df["lif_container_id"].astype(str) == selected_container
        ].copy()

    missing_cols = [col for col in INTENSITY_COLS if col not in filtered_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required intensity columns in combined data: {missing_cols}"
        )

    print(f"Plotting {len(filtered_df)} nuclei (filter: {selected_container})")
    return filtered_df, selected_container


@app.cell
def _(
    GENOTYPE_ORDER,
    GENOTYPE_PALETTE,
    INTENSITY_COLS,
    filtered_df,
    selected_container,
    sns,
):
    jointplot_grid = sns.jointplot(
        data=filtered_df,
        x=INTENSITY_COLS[0],
        y=INTENSITY_COLS[1],
        kind="scatter",
        hue="genotype",
        hue_order=GENOTYPE_ORDER,
        palette=GENOTYPE_PALETTE,
        marginal_ticks=True,
        joint_kws={"alpha": 0.6, "s": 20},
    )
    jointplot_grid.set_axis_labels(
        xlabel="GFP_CAFs mean intensity",      # hardcoded x-axis title
        ylabel=f"AF647_{selected_container} mean intensity",   # hardcoded y-axis title
    )
    jointplot_grid.figure.suptitle(
        f"GFP_CAFs vs AF647_{selected_container} mean intensity",
        y=1.02,
    )
    jointplot_grid
    return


@app.cell
def _(mo):
    gfp_min_slider = mo.ui.slider(
        start=0,
        stop=255,
        value=0,
        step=1,
        label="Min GFP_CAFs mean intensity",
        show_value=True,
    )
    return (gfp_min_slider,)


@app.cell
def _(
    GENOTYPE_ORDER,
    GENOTYPE_PALETTE,
    INTENSITY_COLS,
    filtered_df,
    gfp_min_slider,
    mo,
    selected_container,
    sns,
):
    int_filtered_df = filtered_df[
        filtered_df["GFP_CAFs_mean_int"] > gfp_min_slider.value
    ].copy()

    print(
        f"Intensity filter: GFP_CAFs_mean_int > {gfp_min_slider.value} "
        f"({len(int_filtered_df)} / {len(filtered_df)} nuclei)")

    new_jointplot_grid = sns.jointplot(
        data=int_filtered_df,          # use int_filtered_df instead of filtered_df
        x=INTENSITY_COLS[0],
        y=INTENSITY_COLS[1],
        kind="scatter",
        hue="genotype",
        hue_order=GENOTYPE_ORDER,
        palette=GENOTYPE_PALETTE,
        marginal_ticks=True,
        joint_kws={"alpha": 0.6, "s": 20},
    )
    new_jointplot_grid.set_axis_labels(
        xlabel="GFP_CAFs mean intensity",
        ylabel=f"AF647_{selected_container} mean intensity",
    )
    new_jointplot_grid.figure.suptitle(
        f"GFP_CAFs vs AF647_{selected_container} mean intensity",
        y=1.02,
    )

    mo.vstack([gfp_min_slider, new_jointplot_grid])
    return


if __name__ == "__main__":
    app.run()

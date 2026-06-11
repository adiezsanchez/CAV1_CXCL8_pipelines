<h1>CAV1_CXCL8_pipelines — 3D nuclei segmentation and marker quantification in CAV1/CXCL8 microscopy datasets</h1>

Microscopy image analysis pipelines used in the CAV1_CXCL8 manuscript. The workflow segments nuclei with CellposeSAM, simulates cytoplasm by dilation, extracts per-nucleus morphology and marker intensities from `.lif` containers, and supports batch processing plus downstream genotype comparisons.

<h2>How to install this tool? (Environment setup)</h2>

> [!TIP]
> In order to run these Marimo apps and batch scripts you will need to familiarize yourself with Python virtual environments, IDEs and Git. If you are not familiar with those concepts, watch [Before you start (Python, IDE and Git on Windows)](https://youtu.be/tzdFuxF2E3U).
>
> TL;DR You are busy in the wet lab, skip to the Pixi section below.

1. Clone this repository:

   <code>git clone https://github.com/adiezsanchez/CAV1_CXCL8_pipelines</code>

2. If you do not have git installed, download the code as a `.zip` from the green **Code** button on GitHub.

3. Install [Pixi](https://pixi.sh/latest/installation/) and create the environment from the project root:

   <code>cd CAV1_CXCL8_pipelines && pixi install</code>

> [!TIP]
> [Pixi](https://pixi.sh/latest/installation/) provides reproducible environments across `linux-64`, `win-64`, and `osx-arm64`. GPU PyTorch is enabled on Linux and Windows when NVIDIA drivers are available.

<h2>Running the pipelines</h2>

**Image analysis (single-image QC in Napari)** — `src/app.py`

- Edit interactively: <code>pixi run marimo_edit</code>
- Run read-only: <code>pixi run marimo_run</code>

Loads `.lif` containers from `data/`, predicts or reuses 3D nuclei labels, simulates cytoplasm, extracts per-marker features, and visualizes GFP and CXCL8 intensities in Napari.

**Data analysis (combine CSVs and plot)** — `src/data_analysis.py`

- Edit interactively: <code>pixi run analysis_edit</code>
- Run read-only: <code>pixi run analysis_run</code>

Concatenates per-image CSVs from `results/`, assigns genotype labels from image names (`scr` vs `shCAV1`), and plots marker intensity relationships.

**Batch processing (all images in a folder)**

<code>pixi run loop</code>

Processes every image in the `.lif` containers listed under `raw_data_directory` in `config.yaml` and writes one CSV per image to `results_directory`. Edit `config.yaml` to point at your data, marker channels, and nuclei volume filters before running.

<h2>Workflow summary</h2>

1. Place `.lif` files in `./data/` (or update `raw_data_directory` in `config.yaml`).
2. Explore segmentation and feature extraction with `pixi run marimo_edit`.
3. Batch-process the dataset with `pixi run loop`.
4. Combine results and compare genotypes with `pixi run analysis_edit`.

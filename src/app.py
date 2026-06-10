import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    import tifffile

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from utils.data_viz import map_df_column_to_labels, plot_prop_to_3d_centroids
    from utils.feature_extraction import extract_nuclei_features_per_marker
    from utils.io import (
        calculate_rescale_factor,
        ensure_output_dir,
        explore_lif_container,
        list_containers,
        load_lif_image,
        load_precomputed_results_if_available,
    )
    from utils.napari_marimo import add_or_update_labels_layer
    from utils.segmentation import (
        predict_nuclei_labels,
        simulate_cytoplasm)

    return (
        Path,
        add_or_update_labels_layer,
        calculate_rescale_factor,
        ensure_output_dir,
        explore_lif_container,
        extract_nuclei_features_per_marker,
        list_containers,
        load_lif_image,
        load_precomputed_results_if_available,
        map_df_column_to_labels,
        predict_nuclei_labels,
        simulate_cytoplasm,
        tifffile,
    )


@app.cell
def _():
    # Pump Qt events so Napari stays interactive inside Marimo's asyncio kernel.
    import asyncio
    import sys as _sys

    from qtpy.QtWidgets import QApplication

    qapp = QApplication.instance() or QApplication(_sys.argv)

    async def _qt_pump_coro():
        while True:
            qapp.processEvents()
            await asyncio.sleep(0.01)

    qt_pump_task = asyncio.create_task(_qt_pump_coro())
    import napari

    return (napari,)


@app.cell
def _(Path):
    # Copy the path where your .lif containers are stored, you can use absolute or relative paths to point at other disk locations
    RAW_DATA_DIRECTORY = str(Path(__file__).resolve().parent.parent / "data")

    # Channel index used for CellposeSAM-based 3D nuclei segmentation
    NUCLEI_CHANNEL = 0

    # Minimum and maximum nuclei label volume to use for filtering predicted nuclei labels
    MIN_MAX_NUCLEI_VOLUME = (1500, 25000)

    # Channel info
    MARKERS = (("DAPI", 0), ("GFP_CAFs", 1), ("AF568_TUB", 2), ("AF647_CXCL8", 3))
    return MARKERS, MIN_MAX_NUCLEI_VOLUME, NUCLEI_CHANNEL, RAW_DATA_DIRECTORY


@app.cell
def _(RAW_DATA_DIRECTORY, list_containers, mo):
    # Iterate through the .lif container files in the directory
    lif_containers = list_containers(RAW_DATA_DIRECTORY, file_format="lif")

    if not lif_containers:
        raise FileNotFoundError(
            f"No .lif containers found in '{RAW_DATA_DIRECTORY}'. "
            "Check RAW_DATA_DIRECTORY and file extension."
        )

    _max_container_index = len(lif_containers) - 1
    lif_container_slider = mo.ui.slider(
        start=0,
        stop=_max_container_index,
        value=min(1, _max_container_index),
        label="LIF container index",
        show_value=True,
    )
    return lif_container_slider, lif_containers


@app.cell
def _(explore_lif_container, lif_container_slider, lif_containers, mo):
    # Explore different .lif files (0 defines the first file in the directory)
    LIF_CONTAINER_INDEX = lif_container_slider.value
    lif_path = lif_containers[LIF_CONTAINER_INDEX]

    # Explore the contents of a single .lif container
    nr_imgs, lif_container_id = explore_lif_container(file_path=lif_path, display=True)

    _max_image_index = max(0, nr_imgs - 1)
    lif_image_slider = mo.ui.slider(
        start=0,
        stop=_max_image_index,
        value=min(4, _max_image_index),
        label="LIF image index",
        show_value=True,
    )

    mo.vstack([lif_container_slider, lif_image_slider])
    return lif_container_id, lif_image_slider, lif_path


@app.cell
def _(lif_image_slider, lif_path, load_lif_image):
    # Load a single image from a .lif container
    LIF_IMAGE_INDEX = lif_image_slider.value
    lif_image, lif_image_name, xml_metadata = load_lif_image(
        file_path=lif_path, image_index=LIF_IMAGE_INDEX
    )

    print(f"Image loaded: {lif_image_name}")
    return lif_image, lif_image_name, xml_metadata


@app.cell
def _(MARKERS, lif_image, napari):
    viewer = napari.current_viewer() or napari.Viewer(ndisplay=2)
    viewer.layers.clear()
    viewer.add_image(
        lif_image,
        channel_axis=0,
        colormap=["cyan", "yellow", "white", "magenta"],
        name=[marker_name for marker_name, _ in MARKERS],
    )
    viewer.reset_view()
    return (viewer,)


@app.cell
def _(
    MIN_MAX_NUCLEI_VOLUME,
    NUCLEI_CHANNEL,
    RAW_DATA_DIRECTORY,
    add_or_update_labels_layer,
    calculate_rescale_factor,
    ensure_output_dir,
    lif_container_id,
    lif_image,
    lif_image_name,
    load_precomputed_results_if_available,
    predict_nuclei_labels,
    tifffile,
    viewer,
    xml_metadata,
):
    # Ensure output directory for this container's nuclei labels
    nuclei_labels_dir = ensure_output_dir(
        RAW_DATA_DIRECTORY, lif_container_id, results_type="nuclei_labels"
    )
    print(f"Nuclei labels directory: {nuclei_labels_dir}")

    # Calculate anisotropy CellposeSAM parameter to rescale across the Z-axis (ratio of Z-resolution to XY-resolution)
    rescale_factor = calculate_rescale_factor(xml_metadata, display=True)

    # Load precomputed labels when available; otherwise predict and store them
    nuclei_labels = load_precomputed_results_if_available(
        nuclei_labels_dir, lif_image_name, results_type="nuclei_labels"
    )

    if nuclei_labels is not None:
        print(f"Predictions already calculated for: {lif_image_name} ...loading")
        add_or_update_labels_layer(viewer, nuclei_labels, name="nuclei_labels")

    else:
        # Predict nuclei labels using CellposeSAM using anisotropy correction
        nuclei_labels = predict_nuclei_labels(
            lif_image,
            rescale_factor,
            NUCLEI_CHANNEL,
            MIN_MAX_NUCLEI_VOLUME,
            visualize=True,
            viewer=viewer,
        )
        # Create path for nuclei labels (used only when saving a newly computed prediction)
        nuclei_labels_path = nuclei_labels_dir / f"{lif_image_name}_nuclei_labels.tif"
        # Save the prediction
        tifffile.imwrite(nuclei_labels_path, nuclei_labels)
    return (nuclei_labels,)


@app.cell
def _(nuclei_labels, simulate_cytoplasm, viewer):
    cytoplasm_labels = simulate_cytoplasm(nuclei_labels, dilation_radius=2)
    viewer.add_labels(cytoplasm_labels)
    return (cytoplasm_labels,)


@app.cell
def _(
    MARKERS,
    cytoplasm_labels,
    extract_nuclei_features_per_marker,
    lif_container_id,
    lif_image,
    lif_image_name,
):
    # Create a dictionary containing all image descriptors
    descriptor_dict = {
        "lif_container_id": lif_container_id,
        "lif_image_name": lif_image_name,
    }

    # Extract morphological and intensity features per marker (from cytoplasm)
    props_df = extract_nuclei_features_per_marker(
        cytoplasm_labels, lif_image, MARKERS, descriptor_dict
    )
    return (props_df,)


@app.cell
def _(cytoplasm_labels, map_df_column_to_labels, props_df, viewer):
    # Visualize GFP mean intensity
    map_df_column_to_labels(
        cytoplasm_labels,
        props_df,
        value_column="GFP_CAFs_mean_int",
        colormap="inferno",
        visualize=True,
        viewer=viewer,
    )
    return


@app.cell
def _(cytoplasm_labels, map_df_column_to_labels, props_df, viewer):
    # Visualize AF647 mean intensity (CXCL8 or CAV1)
    map_df_column_to_labels(
        cytoplasm_labels,
        props_df,
        value_column="AF647_CXCL8_mean_int",
        colormap="inferno",
        colormap_vmin=0,
        colormap_vmax=60,
        visualize=True,
        viewer=viewer,
    )
    return


if __name__ == "__main__":
    app.run()

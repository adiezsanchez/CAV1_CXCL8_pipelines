from cellpose import models, core
import numpy as np
from skimage.segmentation import relabel_sequential
import pyclesperanto_prototype as cle

_CELLPOSE_MODEL = None


def _get_cellpose_model(require_gpu: bool = True):
    """
    Lazily initialize and cache the Cellpose model.

    This keeps module import lightweight and allows workflows that only load
    precomputed results to run on CPU-only environments.
    """
    global _CELLPOSE_MODEL

    if _CELLPOSE_MODEL is not None:
        return _CELLPOSE_MODEL

    has_gpu = core.use_gpu()
    if require_gpu and not has_gpu:
        raise RuntimeError(
            "Cellpose nuclei prediction requires GPU, but no GPU was detected. "
            "You can still run workflows that load precomputed nuclei labels."
        )

    _CELLPOSE_MODEL = models.CellposeModel(gpu=has_gpu)
    return _CELLPOSE_MODEL


def _resolve_napari_viewer(viewer):
    """
    Return a napari Viewer for optional visualization.

    If ``viewer`` is passed, it is used. Otherwise tries ``napari.current_viewer()``;
    if none exists, creates ``napari.Viewer()``. Import is deferred until visualization runs.
    """
    if viewer is not None:
        return viewer
    import napari

    v = napari.current_viewer()
    if v is not None:
        return v
    return napari.Viewer()

def _remove_labels_touching_xy(labels: np.ndarray) -> np.ndarray:
    """
    Remove connected-component labels that touch either extreme face of the
    x or y in-plane axes in a 3D label volume.

    Args:
        labels (np.ndarray): 3D labeled array (shape: (z, y, x)), where each unique
            integer (>0) identifies an object and 0 is background.

    Returns:
        np.ndarray: Labeled array of same shape as input, with labels touching the
            extreme faces along x or y axes set to 0.
            Label IDs for remaining components are preserved (no relabeling).
    """
    if labels.ndim != 3:
        raise ValueError("Input must be a 3D array with shape (z, y, x).")

    # Unpack the shape to retrieve dimensions
    _, y_dim, x_dim = labels.shape

    # Extract the four extreme faces along x and y axes
    x0_face = labels[:, :, 0]          # x=0
    x1_face = labels[:, :, -1]         # x=max
    y0_face = labels[:, 0, :]          # y=0
    y1_face = labels[:, -1, :]         # y=max

    # Find all unique label values present on any extreme face (excluding background 0)
    labels_to_remove = np.unique(
        np.concatenate([
            x0_face.ravel(),
            x1_face.ravel(),
            y0_face.ravel(),
            y1_face.ravel(),
        ])
    )
    labels_to_remove = labels_to_remove[labels_to_remove != 0]

    if labels_to_remove.size == 0:
        # No labels to remove; return a copy
        return labels.copy()

    # Remove detected labels by setting them to 0 everywhere
    cleaned = labels.copy()
    cleaned[np.isin(cleaned, labels_to_remove)] = 0

    return cleaned

def _keep_objects_in_size_range(labels: np.ndarray, min_max_size: tuple[int, int]) -> np.ndarray:
    """
    Keep only labeled objects whose voxel count is within a min/max range.

    Args:
        labels (np.ndarray): Labeled image where 0 is background.
        min_max_size (tuple[int, int]): Inclusive size range as (min_size, max_size).

    Returns:
        np.ndarray: Filtered labels, relabeled sequentially from 1..N.
    """
    min_size, max_size = min_max_size
    counts = np.bincount(labels.ravel())
    keep = (counts >= max(min_size, 0)) & (counts <= max_size)
    keep[0] = False  # keep background as 0

    filtered = labels.copy()
    filtered[~keep[labels]] = 0
    filtered, _, _ = relabel_sequential(filtered)
    return filtered

def predict_nuclei_labels(image: np.ndarray, rescale_factor: float, nuclei_channel: int, min_max_nuclei_volume: tuple[int, int] = (0, 1000000), visualize=False, viewer=None) -> np.ndarray:
    """
    Predict nuclei labels using CellposeSAM using anisotropy correction.

    Args:
        image (np.ndarray): Image to predict nuclei labels from.
        rescale_factor (float): Rescale factor to apply to the Z-axis for isotropic scaling (z_um / mean([x_um, y_um])).
        nuclei_channel (int): Channel index of the nuclei channel in the image.
        min_max_nuclei_volume (tuple[int, int], optional): Inclusive min/max nuclei volume
            used to filter predicted labels. Defaults to (0, 1000000).
        visualize (bool, optional): If True, display the predicted nuclei labels in Napari.
        viewer (optional): Napari ``Viewer`` instance. If ``visualize`` is True and this is omitted,
            the current viewer (if any) is used, otherwise a new ``napari.Viewer()`` is created.

    Returns:
        np.ndarray: Nuclei labels.
    """
    model = _get_cellpose_model(require_gpu=True)

    # Predict nuclei labels
    nuclei_labels, _ , _ = model.eval(image[nuclei_channel], do_3D=True, anisotropy=rescale_factor, z_axis=0, niter=1000)
    # Remove labels touching the longest axis extremes
    nuclei_labels = _remove_labels_touching_xy(nuclei_labels)
    # Filter nuclei labels to keep only those within the specified size range
    nuclei_labels = _keep_objects_in_size_range(nuclei_labels, min_max_nuclei_volume)

    # Display the resulting nuclei labels in Napari if requested.
    if visualize:
        from utils.napari_marimo import add_or_update_labels_layer

        v = _resolve_napari_viewer(viewer)
        add_or_update_labels_layer(v, nuclei_labels, name="nuclei_labels")

    return nuclei_labels

def simulate_cytoplasm(nuclei_labels, dilation_radius=2, erosion_radius=0):

    if erosion_radius >= 1:

        # Erode nuclei_labels to maintain a closed cytoplasmic region when labels are touching (if needed)
        eroded_nuclei_labels = cle.erode_labels(nuclei_labels, radius=erosion_radius)
        eroded_nuclei_labels = cle.pull(eroded_nuclei_labels)
        nuclei_labels = eroded_nuclei_labels

    # Dilate nuclei labels to simulate the surrounding cytoplasm
    cyto_nuclei_labels = cle.dilate_labels(nuclei_labels, radius=dilation_radius)
    cytoplasm = cle.pull(cyto_nuclei_labels)

    # Create a binary mask of the nuclei
    nuclei_mask = nuclei_labels > 0

    # Set the corresponding values in the cyto_nuclei_labels array to zero
    cytoplasm[nuclei_mask] = 0

    return cytoplasm


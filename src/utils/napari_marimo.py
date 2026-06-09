"""Helpers for using Napari inside Marimo's asyncio kernel."""


def add_or_update_image_layer(viewer, data, name: str, **kwargs):
    """Add an image layer or replace data on an existing layer of the same name."""
    if name in viewer.layers:
        viewer.layers[name].data = data
    else:
        viewer.add_image(data, name=name, **kwargs)


def add_or_update_labels_layer(viewer, data, name: str = "nuclei_labels", **kwargs):
    """Add a labels layer or replace data on an existing layer of the same name."""
    if name in viewer.layers:
        viewer.layers[name].data = data
    else:
        viewer.add_labels(data, name=name, **kwargs)

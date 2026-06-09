"""Batch-process LIF containers and save per-image feature tables as CSV."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import tifffile
import yaml

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.feature_extraction import extract_nuclei_features_per_marker
from utils.io import (
    calculate_rescale_factor,
    ensure_output_dir,
    explore_lif_container,
    list_containers,
    load_lif_image,
    load_precomputed_results_if_available,
)
from utils.segmentation import predict_nuclei_labels, simulate_cytoplasm

logger = logging.getLogger(__name__)

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _resolve_path(path_value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return config


def setup_logging(config: dict) -> None:
    logging_config = config.get("logging", {})
    level_name = str(logging_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    if logging_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root_logger.addHandler(console_handler)

    log_file = logging_config.get("log_file")
    if log_file:
        log_path = _resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root_logger.addHandler(file_handler)


def sanitize_filename_component(value: str) -> str:
    sanitized = _INVALID_FILENAME_CHARS.sub("_", value.strip())
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" .")
    return sanitized or "unnamed"


def output_csv_path(
    results_dir: Path,
    lif_container_id: str,
    lif_image_name: str,
) -> Path:
    container_part = sanitize_filename_component(lif_container_id)
    image_part = sanitize_filename_component(lif_image_name)
    return results_dir / f"{container_part}_{image_part}.csv"


def process_image(
    lif_path: str,
    image_index: int,
    lif_container_id: str,
    config: dict,
    raw_data_dir: Path,
    results_dir: Path,
) -> Path | None:
    lif_image, lif_image_name, xml_metadata = load_lif_image(
        file_path=lif_path,
        image_index=image_index,
    )
    csv_path = output_csv_path(results_dir, lif_container_id, lif_image_name)

    if config.get("skip_existing_csv", True) and csv_path.is_file():
        logger.info(
            "Skipping image %d (%s): CSV already exists at %s",
            image_index,
            lif_image_name,
            csv_path,
        )
        return None

    logger.info("Processing image %d: %s", image_index, lif_image_name)

    nuclei_labels_dir = ensure_output_dir(
        raw_data_dir,
        lif_container_id,
        results_type="nuclei_labels",
    )
    rescale_factor = calculate_rescale_factor(xml_metadata, display=False)
    logger.debug(
        "Rescale factor for %s: %.4f (labels dir: %s)",
        lif_image_name,
        rescale_factor,
        nuclei_labels_dir,
    )

    nuclei_labels = load_precomputed_results_if_available(
        nuclei_labels_dir,
        lif_image_name,
        results_type="nuclei_labels",
    )

    if nuclei_labels is not None:
        logger.info("Loaded precomputed nuclei labels for %s", lif_image_name)
    else:
        logger.info("Predicting nuclei labels for %s", lif_image_name)
        min_max_volume = tuple(config["min_max_nuclei_volume"])
        nuclei_labels = predict_nuclei_labels(
            lif_image,
            rescale_factor,
            config["nuclei_channel"],
            min_max_volume,
            visualize=False,
        )
        nuclei_labels_path = nuclei_labels_dir / f"{lif_image_name}_nuclei_labels.tif"
        tifffile.imwrite(nuclei_labels_path, nuclei_labels)
        logger.info("Saved nuclei labels to %s", nuclei_labels_path)

    cytoplasm_labels = simulate_cytoplasm(
        nuclei_labels,
        dilation_radius=config.get("cytoplasm_dilation_radius", 2),
        erosion_radius=config.get("cytoplasm_erosion_radius", 0),
    )

    markers = [tuple(marker) for marker in config["markers"]]
    descriptor_dict = {
        "lif_container_id": lif_container_id,
        "lif_image_name": lif_image_name,
    }
    props_df = extract_nuclei_features_per_marker(
        cytoplasm_labels,
        lif_image,
        markers,
        descriptor_dict,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    props_df.to_csv(csv_path, index=False)
    logger.info(
        "Saved %d nuclei features for %s to %s",
        len(props_df),
        lif_image_name,
        csv_path,
    )
    return csv_path


def process_container(
    lif_path: str,
    config: dict,
    raw_data_dir: Path,
    results_dir: Path,
) -> tuple[int, int]:
    container_name = Path(lif_path).name
    logger.info("Starting container: %s", container_name)

    nr_imgs, lif_container_id = explore_lif_container(file_path=lif_path, display=False)
    logger.info("Container %s contains %d image(s)", lif_container_id, nr_imgs)

    processed = 0
    skipped = 0
    for image_index in range(nr_imgs):
        started_at = time.perf_counter()
        try:
            csv_path = process_image(
                lif_path=lif_path,
                image_index=image_index,
                lif_container_id=lif_container_id,
                config=config,
                raw_data_dir=raw_data_dir,
                results_dir=results_dir,
            )
        except Exception:
            logger.exception(
                "Failed processing image %d in container %s",
                image_index,
                lif_container_id,
            )
            raise

        elapsed = time.perf_counter() - started_at
        if csv_path is None:
            skipped += 1
            logger.debug("Image %d finished in %.1fs (skipped)", image_index, elapsed)
        else:
            processed += 1
            logger.info("Image %d finished in %.1fs", image_index, elapsed)

    logger.info(
        "Finished container %s: %d saved, %d skipped",
        lif_container_id,
        processed,
        skipped,
    )
    return processed, skipped


def run_loop(config: dict) -> None:
    raw_data_dir = _resolve_path(config["raw_data_directory"])
    results_dir = _resolve_path(config["results_directory"])

    lif_containers = list_containers(str(raw_data_dir), file_format="lif")
    if not lif_containers:
        raise FileNotFoundError(
            f"No .lif containers found in '{raw_data_dir}'. "
            "Check raw_data_directory in config.yaml."
        )

    logger.info("Found %d LIF container(s) in %s", len(lif_containers), raw_data_dir)
    logger.info("Writing CSV results to %s", results_dir)

    total_processed = 0
    total_skipped = 0
    loop_started_at = time.perf_counter()

    for container_index, lif_path in enumerate(lif_containers):
        logger.info(
            "Container %d/%d: %s",
            container_index + 1,
            len(lif_containers),
            Path(lif_path).name,
        )
        processed, skipped = process_container(
            lif_path=lif_path,
            config=config,
            raw_data_dir=raw_data_dir,
            results_dir=results_dir,
        )
        total_processed += processed
        total_skipped += skipped

    elapsed = time.perf_counter() - loop_started_at
    logger.info(
        "Loop complete in %.1fs: %d CSV(s) written, %d skipped across %d container(s)",
        elapsed,
        total_processed,
        total_skipped,
        len(lif_containers),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process LIF files and export per-image feature CSVs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to YAML config file (default: config.yaml in project root)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    setup_logging(config)
    logger.info("Loaded config from %s", config_path)
    run_loop(config)


if __name__ == "__main__":
    main()

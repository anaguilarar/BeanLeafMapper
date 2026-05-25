"""Project-wide constants and defaults."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "photos_iaf_20260328"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class ArucoConfig:
    """Physical size (cm) of the ArUco/AprilTag marker side and detector dictionary."""

    marker_size_cm: float = 9.0
    dictionary: str = "DICT_APRILTAG_16h5"


@dataclass
class GridConfig:
    """Graph-paper grid spacing on lab photos (small square side length, cm)."""

    square_size_cm: float = 0.5


@dataclass
class DetectionConfig:
    """SAM3 prompts and thresholds."""

    leaf_prompt: str = "leaf"
    leaf_confidence: float = 0.45
    image_downscale: int = 2
    # Main-plant filter:
    #   if `main_plant_cluster_eps_cm` is set, DBSCAN-cluster leaves and keep the
    #   cluster closest to the image centre.
    #   else fall back to `main_plant_max_distance_cm` radius filter.
    main_plant_cluster_eps_cm: float | None = 8.0
    main_plant_cluster_min_samples: int = 3
    main_plant_max_distance_cm: float = 30.0
    min_area_quantile: float = 0.75


@dataclass
class ModelConfig:
    """Model backend + per-backend knobs.

    `backend` selects which detector to build. Only "sam3_image" is shipped today;
    other entries are placeholders for future backends (see README for the menu).
    """

    backend: str = "sam3_image"
    # SAM3 image-specific knobs:
    checkpoint_path: str | None = None   # local .pt file; else download from HF
    device: str | None = None            # "cuda", "cpu", "cuda:0", ...  None = auto
    compile: bool = False                # torch.compile (~2x on Ampere+); needs sm_80+


@dataclass
class PipelineConfig:
    photos_dir: Path = DEFAULT_PHOTOS_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    aruco: ArucoConfig = field(default_factory=ArucoConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated, Literal
from pathlib import Path

dictconfig = "DICT_APRILTAG_16h5"
class ArucoConfig(BaseModel):
    """Physical size (cm) of the ArUco/AprilTag marker side and detector dictionary."""
    marker_size: float = Field(default=9.0, description="marker size in cm")
    dictionary: str = Field(default="DICT_APRILTAG_16h5", description="aruco dict configuration")

class GridConfig(BaseModel):
    """Graph-paper grid spacing on lab photos (small square side length, cm)."""

    square_size_cm: float = 0.5

class GeneralInfoConfig(BaseModel):
    """General information about settings"""
    photos_dir: Annotated[Path,  Field(description = 'input path')]
    output_dir: Annotated[Path,  Field(description = 'ouput path')]
    

class DetectionConfig(BaseModel):
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


class ModelConfig(BaseModel):
    """Model backend + per-backend knobs.

    `backend` selects which detector to build. Only "sam3_image" is shipped today;
    other entries are placeholders for future backends (see README for the menu).
    """

    backend: str = "sam3_image"
    # SAM3 image-specific knobs:
    checkpoint_path: str | None = None   # local .pt file; else download from HF
    device: str | None = None            # "cuda", "cpu", "cuda:0", ...  None = auto
    compile: bool = False     
    
    
class PipelineConfig(BaseModel):
    GENERAL_INFO: Annotated[GeneralInfoConfig, Field(description= "top-level settings")]
    ARUCO:  Annotated[ ArucoConfig , Field(description = ' ArUco/AprilTag marker side')]
    LAB_GRID: Annotated[ GridConfig  , Field(description = 'Graph-paper grid spacing on lab photos (small square side length, cm).')]
    DETECTION: Annotated[ DetectionConfig, Field(description = 'SAM3 prompts and thresholds')]
    MODEL: Annotated[ModelConfig, Field(description = 'Model backend')]



from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, MISSING, OmegaConf


DEFAULT_VLM_SYSTEM_PROMPT = (
    "You identify interactable portable objects in multi-view 3D scene renderings. "
    "When multiple objects share the same label, distinguish them using their relative position "
    "in the scene or image, such as left, right, center, front, back, near window, near table, or "
    "similar spatial cues. "
    "Return only valid JSON."
)

DEFAULT_VLM_USER_PROMPT = (
    "Inspect these multi-view renderings of one scene. Return a JSON array of portable, "
    "manipulable object instances suitable for object-level decomposition. Each item must have "
    'keys: "name", "category", "portable", "recognizability", "text_prompt". '
    "If there are multiple instances of the same object category or label, make each name unique by "
    "including a clear position hint in the name and text_prompt, for example: "
    '"chair_left", "chair_right", "chair_near_window", "chair_center_table". '
    'Use recognizability values "precise", "subtle", or "unrecognizable". '
    "Return at most {max_objects} items. Do not include walls, floor, ceiling, fixed cabinets, "
    "or large architecture."
)


@dataclass
class DeviceConfig:
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class VLMPromptConfig:
    system: str = DEFAULT_VLM_SYSTEM_PROMPT
    user: str = DEFAULT_VLM_USER_PROMPT


@dataclass
class VLMConfig:
    model: str = "gpt-5.5"
    api_base: str = "${oc.env:OPENAI_API_BASE,''}"
    api_key: str = "${oc.env:OPENAI_API_KEY,''}"
    num_keyframes: int = 8
    max_objects: int = 20
    max_retries: int = 3
    timeout: float = 90.0
    image_max_side: int = 768
    prompts: VLMPromptConfig = field(default_factory=VLMPromptConfig)


@dataclass
class SAM3ComponentFilterConfig:
    enabled: bool = False
    min_area_ratio: float = 0.0002
    max_area_ratio: float = 0.18
    min_component_area_px: int = 32
    reject_multi_component_union: bool = True


@dataclass
class SAM3Config:
    source_dir: str = "third_party/sam3"
    checkpoint_path: str = "sam3/sam3.1_multiplex.pt"
    device: str = "${device.device}"
    confidence_threshold: float = 0.45
    max_candidates_per_view: int = 5
    min_candidate_quality: float = 0.0
    instance_strategy: str = "largest"  # largest | best_score | best_quality | union
    mode: str = "image"
    component_filter: SAM3ComponentFilterConfig = field(default_factory=SAM3ComponentFilterConfig)


@dataclass
class InstanceSeparationConfig:
    enabled: bool = False
    core_erode_px: int = 2
    ignore_dilate_px: int = 4
    min_core_area_ratio: float = 0.15
    core_weight: float = 1.5
    exclusion_weight: float = 0.75
    exclusion_dilate_px: int = 6
    exclusion_min_overlap: float = 0.001
    exclusion_max_bbox_gap_px: int = 16
    adaptive_threshold: bool = False
    adaptive_threshold_max_delta: float = 0.08
    adaptive_threshold_quantile: float = 0.75
    adaptive_threshold_min_active_ratio: float = 0.002


@dataclass
class SoftAssignmentConfig:
    lambda_fp: float = 1.0
    threshold: float = 0.5
    num_iters: int = 500
    lr: float = 0.01
    log_interval: int = 50
    render_chunk: int = 3
    depth_gate_enabled: bool = False
    depth_tolerance: float = 0.08
    min_alpha: float = 0.05
    use_mask_reliability_weights: bool = True
    instance_separation: InstanceSeparationConfig = field(default_factory=InstanceSeparationConfig)


@dataclass
class CandidateSupportConfig:
    enabled: bool = True
    depth_gate_enabled: bool = True
    depth_tolerance: float = 0.08
    min_alpha: float = 0.05
    max_candidates_per_view: int = 5
    min_candidate_quality: float = 0.0
    component_count_max_points: int = 4096
    support_duplicate_iou: float = 0.45
    support_duplicate_containment: float = 0.70
    identity_suppression_enabled: bool = False
    identity_suppression_min_reliability_gap: float = 0.02


@dataclass
class FeatureFieldConfig:
    feature_dim: int = 16
    num_iters: int = 1000
    lr: float = 0.01
    temperature: float = 0.07
    threshold: float = 0.75
    render_chunk: int = 8
    log_interval: int = 50
    lambda_mask: float = 1.0
    lambda_fp: float = 2.0
    lambda_exclusion: float = 0.75
    use_background_query: bool = True
    save_feature_field: bool = True


@dataclass
class AssignmentConfig:
    method: str = "soft"  # soft | feature_field
    feature_field: FeatureFieldConfig = field(default_factory=FeatureFieldConfig)


@dataclass
class PrefilterConfig:
    enabled: bool = False
    core_erode_px: int = 2
    min_core_area_ratio: float = 0.15
    min_view_hits: int = 2
    min_hit_ratio: float = 0.25
    depth_gate_enabled: bool = False
    depth_tolerance: float = 0.08
    min_alpha: float = 0.05


@dataclass
class ObjectCleanupConfig:
    enabled: bool = False
    keep_largest_component: bool = True
    radius: float = 1.2
    min_component_gaussians: int = 128
    min_component_ratio: float = 0.08


@dataclass
class ObjectDedupConfig:
    enabled: bool = False
    suppress: bool = False
    mode: str = "adaptive"  # adaptive | legacy
    duplicate_score_threshold: float = 0.62
    fragment_containment_threshold: float = 0.85
    fragment_max_count_ratio: float = 0.70
    max_centroid_distance: float = 1.0
    min_horizontal_iou: float = 0.45
    min_3d_iou: float = 0.35
    min_containment: float = 0.60
    prefer_larger: bool = True
    up_axis: Optional[str] = None


@dataclass
class ViewPropagationConfig:
    enabled: bool = False
    max_hops: int = 2
    neighbor_top_k: int = 4
    max_pose_distance: float = 1.5
    max_angle_deg: float = 45.0
    min_view_overlap: float = 0.15
    anchor_min_score: float = 0.5
    anchor_min_area: float = 0.005
    anchor_max_area_ratio: float = 0.35
    local_sam_weight: float = 1.0
    propagated_weight: float = 0.75
    geometry_weight: float = 0.75
    association_enabled: bool = True
    candidate_unique_assignment: bool = True
    fusion_mode: str = "intersection"  # intersection | propagated | union
    min_assignment_iou: float = 0.6
    min_assignment_visibility: float = 0.25
    local_mask_dilate_px: int = 3
    min_fused_area_ratio: float = 0.25
    min_reprojection_iou: float = 0.2
    max_depth_error: float = 0.05
    track_merge_iou: float = 0.25
    min_track_views: int = 2
    conflict_min_views: int = 1
    conflict_suppression: bool = True
    fallback_to_sam3: bool = False
    fallback_to_raw_on_empty_graph: bool = True
    fallback_to_raw_on_no_association: bool = True
    save_debug: bool = True


@dataclass
class ChunkingConfig:
    enabled: bool = False
    method: str = "xy_grid"
    up_axis: Optional[str] = None
    prompt: Optional[str] = None
    roi_center: Optional[list[float]] = None
    roi_radius: Optional[float] = None
    roi_margin: float = 0.0
    min_roi_overlap: float = 0.15
    keep_top_k: int = 2
    bounds_quantile: list[float] = field(default_factory=lambda: [0.01, 0.99])
    target_gaussians_per_chunk: int = 120000
    min_chunk_size_xy: float = 3.5
    max_chunk_size_xy: float = 5.0
    overlap_xy: float = 0.8
    min_gaussians_per_chunk: int = 10000
    max_chunks: int = 16
    merge_iou_threshold: float = 0.20
    merge_centroid_distance: float = 1.0


@dataclass
class RingSpec:
    name: str = "ring"
    num_cameras: int = 8
    orbit_radius: Optional[float] = None
    orbit_radius_scale: Optional[float] = None
    height_offset: float = 0.0
    pitch_deg: Optional[float] = None
    look_at_center: bool = True
    look_at_offset_xy: list[float] = field(default_factory=lambda: [0.0, 0.0])
    look_at_offset_scale: float = 0.0


@dataclass
class CameraGenConfig:
    trajectory_mode: str = "single_ring"  # single_ring | multi_ring | hybrid
    auto_center: bool = False
    up_axis: str = "z"  # z | y | x | -z | -y | -x
    keyframe_strategy: str = "uniform"  # uniform | diverse
    keyframe_min_angle_deg: float = 20.0
    two_pass: bool = False  # reserved for Phase 2 object-centric anchor views
    num_cameras: int = 16
    image_width: int = 512
    image_height: int = 512
    fov_deg: float = 90.0
    pitch_deg: float = 15.0
    radius_jitter: float = 0.5
    orbit_radius: Optional[float] = None
    look_at_center: bool = False
    height_offset: float = 0.0
    center: Optional[list[float]] = None
    rings: list[RingSpec] = field(default_factory=list)


@dataclass
class DecompositionConfig:
    manual_prompts: list[str] = field(default_factory=list)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    candidate_support: CandidateSupportConfig = field(default_factory=CandidateSupportConfig)
    view_propagation: ViewPropagationConfig = field(default_factory=ViewPropagationConfig)
    prefilter: PrefilterConfig = field(default_factory=PrefilterConfig)
    soft_assignment: SoftAssignmentConfig = field(default_factory=SoftAssignmentConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    object_cleanup: ObjectCleanupConfig = field(default_factory=ObjectCleanupConfig)
    object_dedup: ObjectDedupConfig = field(default_factory=ObjectDedupConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    camera_gen: CameraGenConfig = field(default_factory=CameraGenConfig)


@dataclass
class PipelineConfig:
    output_dir: str = MISSING
    device: DeviceConfig = field(default_factory=DeviceConfig)
    stage1: DecompositionConfig = field(default_factory=DecompositionConfig)


def load_config(config_path: Path | str, overrides: list[str] | None = None) -> DictConfig:
    base = OmegaConf.structured(PipelineConfig)
    file_cfg = OmegaConf.load(str(config_path))
    cfg = OmegaConf.merge(base, file_cfg)
    if overrides:
        _apply_overrides(cfg, overrides)
    OmegaConf.resolve(cfg)
    return cfg


def default_config(overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.structured(PipelineConfig)
    if overrides:
        _apply_overrides(cfg, overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _apply_overrides(cfg: DictConfig, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Config override must be KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        parsed = OmegaConf.from_dotlist([f"value={value}"]).value
        OmegaConf.update(cfg, key, parsed, merge=True)

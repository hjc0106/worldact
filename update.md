# WorldAct Stage 2: Object Refinement - Update Log

## Bug Fixes (Critical)

### 1. merge.py: Fix `_generated_keep_mask` logic reversal
**Problem**: The original `low_alpha` filter kept generated points only where the refined scene had LOW alpha (background regions), discarding generated points that fell on the actual object surface. This caused the "completion" result to be pure background noise.

**Fix**: Replaced `keep & in_mask & low_alpha` with `keep & in_mask & (at_surface | in_gap)`:
- `at_surface`: Points where refined scene has high alpha AND projected depth matches rendered depth (within proportional tolerance) — these are points on/near the object surface
- `in_gap`: Points where refined scene has low alpha BUT the depth is within the object's depth range (with 30% margin) — these fill genuine holes in the object
- Added per-view proportional depth tolerance: `depth_tolerance * |rendered_depth|` instead of absolute tolerance

### 2. utils.py: Fix `depth_gate` absolute vs proportional tolerance
**Problem**: `depth_tolerance=0.08` was used as absolute tolerance in meters, which is too strict for distant objects and too loose for nearby ones.

**Fix**: Changed to proportional: `tolerance = depth_tolerance * |rendered_depth|` (minimum 0.1), so a tolerance of 0.08 means 8% of rendered depth.

### 3. prune.py: Add depth consistency filtering
**Problem**: Background noise Gaussians could pass mask support filtering because they project into the 2D mask region, even though their depth is far from the object body.

**Fix**: Added `_depth_consistency_keep()` which:
- Computes per-view depth distribution (10th/90th percentile) of mask-interior points
- Builds an object depth range `[z_lo - margin, z_hi + margin]` with 30% margin
- Rejects points whose depth falls outside this range in >=50% of views
- New config fields: `use_depth_consistency`, `depth_consistency_percentile`, `depth_consistency_margin`

## New Modules (Generation Pipeline)

### 4. align.py: ICP + Procrustes Coordinate Alignment
**Purpose**: Compute 4x4 transform to align generated 3DGS (canonical space) to world coordinates.

- `compute_alignment_transform()`: Coarse alignment (PCA + center + scale) → Fine alignment (ICP)
- `_coarse_alignment()`: Centers objects, normalizes scale by 90th percentile extent, aligns principal axes via PCA+Singular Value Decomposition
- `_icp_alignment()`: Iterative closest point with Procrustes per step, max correspondence distance culling
- `write_alignment_metadata()`: Writes transform to metadata.json (consumed by merge.py's `_apply_metadata_transform`)

### 5. inpaint.py: Multi-view Rendering + Inpainting
**Purpose**: Render prune scene from multiple views, composite with source scene, and inpaint missing regions.

- `render_multiview_inpaint()`: Orchestrates multi-view rendering pipeline
- `_select_views()`: Selects best view + evenly spaced views from available mask views
- `_render_and_inpaint_view()`: Per-view rendering → mask composition → inpainting
- `_inpaint_lama()`: LaMa integration for image inpainting (with graceful fallback)

### 6. generate.py: Multi-view Generation (SV3D / Zero123++)
**Purpose**: Generate multi-view consistent images from a single crop image.

- `generate_multiview_images()`: Main entry point supporting `sv3d` and `zero123plus` models
- `_generate_sv3d()`: Uses StableVideoDiffusionPipeline from diffusers for SV3D inference
- `_generate_zero123plus()`: Uses Zero123++ v2 for 6-view orbital generation
- `_generate_orbit_cameras()`: Generates orbital camera parameters (extrinsic/intrinsic matrices)
- `_estimate_elevation_from_scene()`: Estimates camera elevation from best view's extrinsic matrix
- Fallback: If model weights unavailable, copies crop image as placeholder views

### 7. reconstruct.py: 3DGS Reconstruction from Multi-view Images
**Purpose**: Reconstruct complete 3DGS point cloud from multi-view images.

- `reconstruct_3dgs()`: Main entry supporting `gs_lrm`, `instant_splat`, and `fallback` modes
- `_run_gs_lrm_inference()`: Direct GS-LRM model inference for 3DGS reconstruction
- `_run_instant_splat_inference()`: InstantSplat reconstruction via subprocess
- `_fallback_point_cloud()`: Simple point cloud generation using multi-view triangulation (no learned model needed)
- Output: `generated.ply` + `metadata.json` (with identity transform placeholder)

### 8. sv3d_backend.py: End-to-End SV3D Backend
**Purpose**: `FeedForwardBackend` subclass that chains all 4 stages.

Pipeline: `Inpaint → Generate → Reconstruct → Align`
1. Load seed PLY from request JSON
2. Render multi-view inpainted images
3. Generate multi-view consistent images (SV3D/Zero123++)
4. Reconstruct 3DGS (GS-LRM/InstantSplat)
5. Compute alignment transform (ICP + Procrustes)
6. Write `generated.ply` + `metadata.json`

Registered as `"sv3d"` backend in `make_backend()`.

## Enhanced Modules

### 9. merge.py: Overlap Region Blending
**Added**: `_blend_overlap_region()` function for smooth merging at the seed/generated boundary.

- Computes distance from each generated Gaussian to nearest seed Gaussian
- Applies distance-weighted opacity reduction and scale shrinking in the overlap zone (`blend_distance`)
- Within `blend_distance`: opacity decreases by up to 50%, scale shrinks by up to 30%
- Eliminates hard visual seams between original and generated Gaussians

### 10. io.py: Multi-view Asset Writing
**Added**: `_write_multiview_assets()` function integrated into `write_object_package()`.

- Renders source scene from multiple views with object masks
- Saves RGB + mask per view to `work_dir/multiview/`
- Writes per-view camera parameters (extrinsic, intrinsic) to request JSON
- `_select_multiview_indices()`: Deterministic view sampling (seed=42)

### 11. common.py: Extended Configuration
New config fields:
- `blend_distance: float = 0.05` — overlap blend distance for merge
- `num_generation_views: int = 6` — number of views for generation
- `generation_resolution: int = 512` — output resolution for generation
- `generation_model: str = "sv3d"` — generation model (`sv3d` | `zero123plus`)
- `reconstruction_model: str = "gs_lrm"` — reconstruction model (`gs_lrm` | `instant_splat`)
- `inpaint_model: str = "lama"` — inpainting model (`lama` | `none`)
- `icp_iterations: int = 50` — ICP iteration count for alignment

### 12. backend.py: Extended Backend Registry
- `make_backend()` now accepts `"sv3d"` kind and `**kwargs` for SV3DBackend configuration
- SV3DBackend lazy-imported to avoid requiring heavy dependencies at module level

### 13. refine_objects.py: Extended CLI
New arguments:
- `--blend-distance` (default 0.05)
- `--num-generation-views` (default 6)
- `--generation-resolution` (default 512)
- `--generation-model {sv3d,zero123plus}` (default sv3d)
- `--reconstruction-model {gs_lrm,instant_splat}` (default gs_lrm)
- `--inpaint-model {lama,none}` (default lama)
- `--icp-iterations` (default 50)
- `--backend {none,external,sv3d}` (expanded with sv3d)

## Usage Examples

### Prune only (no generation, previous bug-fixed behavior):
```bash
python scripts/refine_objects.py \
  --ply outputs/building/aerial_regions_parking_bev_crop/regions/region_000/chunk.ply \
  --stage-dir outputs/building/parking_region_000_instance_seg_vp_conflictmin1_adaptive_dedupdiag/stage1
```

### Full pipeline with SV3D + GS-LRM:
```bash
python scripts/refine_objects.py \
  --ply outputs/building/aerial_regions_parking_bev_crop/regions/region_000/chunk.ply \
  --stage-dir outputs/building/parking_region_000_instance_seg_vp_conflictmin1_adaptive_dedupdiag/stage1 \
  --backend sv3d \
  --generation-model sv3d \
  --reconstruction-model gs_lrm \
  --icp-iterations 50
```

### External backend with custom script:
```bash
python scripts/refine_objects.py \
  --ply ... --stage-dir ... \
  --backend external \
  --backend-command "python scripts/my_gen.py {request}"
```

## File Change Summary

| File | Status | Description |
|------|--------|-------------|
| `worldact/stage2_object_refinement/align.py` | NEW | ICP + Procrustes coordinate alignment |
| `worldact/stage2_object_refinement/inpaint.py` | NEW | Multi-view render + LaMa inpainting |
| `worldact/stage2_object_refinement/generate.py` | NEW | SV3D/Zero123++ multi-view generation |
| `worldact/stage2_object_refinement/reconstruct.py` | NEW | GS-LRM/InstantSplat 3DGS reconstruction |
| `worldact/stage2_object_refinement/sv3d_backend.py` | NEW | End-to-end SV3D backend |
| `worldact/stage2_object_refinement/merge.py` | MODIFIED | Fix keep_mask logic; add overlap blending |
| `worldact/stage2_object_refinement/prune.py` | MODIFIED | Add depth consistency filtering |
| `worldact/stage2_object_refinement/utils.py` | MODIFIED | Fix depth_gate to proportional tolerance |
| `worldact/stage2_object_refinement/common.py` | MODIFIED | Add generation config fields |
| `worldact/stage2_object_refinement/backend.py` | MODIFIED | Add sv3d backend support |
| `worldact/stage2_object_refinement/io.py` | MODIFIED | Add multi-view asset writing |
| `scripts/refine_objects.py` | MODIFIED | Add new CLI arguments |
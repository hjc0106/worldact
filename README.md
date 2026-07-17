# WorldAct: Activating Monolithic 3D Worlds into Interactive-Ready Object-Centric Scenes

<p align="center">
  <img src="assets/logo.jpg" alt="WorldAct logo" width="260">
</p>

[Jichen Hu](https://github.com/Abuuu122), [Jiawei Guo](https://github.com/GuoCalix), [Jiazhong Cen](https://github.com/Jumpat), [Chen Yang](), [Sikuang Li](), [Wei Shen]()

"WorldAct: Activating Monolithic 3D Worlds into Interactive-Ready Object-Centric Scenes"
<p align="center">
  <a href=https://sjtu-deepvisionlab.github.io/WorldAct><img alt="Official Site" src="https://img.shields.io/badge/Official%20Site-333399?logo=homeassistant&logoColor=white"></a>
  <a href="https://github.com/SJTU-DeepVisionLab/WorldAct/releases"><img alt="GitHub downloads" src="https://img.shields.io/github/downloads/SJTU-DeepVisionLab/WorldAct/total?color=green"></a>
  <a href="https://github.com/SJTU-DeepVisionLab/WorldAct"><img alt="Visitors" src="https://visitor-badge.laobi.icu/badge?page_id=SJTU-DeepVisionLab/WorldAct"></a>
  <a href="https://arxiv.org/abs/2605.15843"><img alt="arXiv" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="#"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-TBD-2ea44f?logo=databricks&logoColor=white"></a>
  <a href="https://github.com/SJTU-DeepVisionLab/WorldAct/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/SJTU-DeepVisionLab/WorldAct"></a>
</p>


#### 🔥🔥🔥News

- **2026-05-14:** This repository is released.

---

> Recent 3D world modeling systems based on generative scene synthesis, such as Marble, can create coherent and explorable 3D environments, yet their outputs are typically static monolithic assets with limited editability and physical interaction. This restricts their use in immersive content creation and embodied simulation, where generated worlds must be actively modified and manipulated. To tackle this challenge, we present WorldAct, a framework that converts static generated 3D worlds into editable and interaction-ready scenes. WorldAct uses a multimodal agent to guide scene decomposition, identify actionable objects, reconstruct geometrically aligned object-level meshes for interaction, and restore the residual background via 3D inpainting. The resulting scenes support object-level editing, collision-aware manipulation, and embodied task execution while preserving global scene coherence. Experiments show that WorldAct enables richer interaction scenarios than the original generated scenes, suggesting a practical path toward editable and interactive 3D world models.

![overview](assets/image.png)

---

### Pipeline

![pipeline overview](assets/pipline_o.png)

![pipeline detail](assets/pipline_d.png)

---

## TODO

- [ ] Release training code.
- [ ] Publish datasets.

## Contents

1. [Testing](#testing)
2. [Stage2 Object Refinement](#stage2-object-refinement)
3. [Results](#results)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)

## <a name="testing"></a>Testing

TBD

## <a name="stage2-object-refinement"></a>Stage2 Object Refinement

The staged object refinement workflow can be run incrementally. The examples below use the parking-lot region used during local validation; adjust paths and `--only-object-id` for other scenes.

Set a stable CUDA host compiler before running commands that render 3DGS views:

```bash
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
```

### Phase 1: Background Separation And Seed Cleanup

Run only object/background separation and seed cleanup. This does not invoke SV3D.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/hongjiacheng/miniconda3/envs/worldact/bin/python scripts/refine_objects.py \
  --ply outputs/building/aerial_regions_parking_bev_crop/regions/region_000/chunk.ply \
  --stage-dir outputs/building/parking_region_000_instance_seg_balanced_recall_identitydiag_nearcam_sixrings_allviews/stage1 \
  --output-dir outputs/building/test_stage2_object_refinement_cleanup_adaptive \
  --device cuda \
  --backend none \
  --refinement-stage cleanup_only \
  --only-object-id 1
```

Important artifacts:

- `object_refined_seed.ply`: cleaned object seed used by later stages.
- `object_core.ply`: high-confidence object points.
- `uncertain_shell.ply`: retained but lower-confidence points.
- `background_reject.ply`: rejected background/noise points.
- `metrics.json` and `separation_metrics.json`: cleanup diagnostics.

Adaptive cleanup is enabled by default. To reproduce a fixed-threshold run, add `--no-adaptive-cleanup` and optionally set:

```bash
--max-off-mask-ratio 0.55 \
--robust-core-bbox-quantile 0.04 \
--robust-core-bbox-expand 0.18
```

### Phase 2: Clean View Selection And SV3D Pseudo Views

Run cleanup, foregroundized view selection, and SV3D pseudo-view generation, but skip final merge.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/hongjiacheng/miniconda3/envs/worldact/bin/python scripts/refine_objects.py \
  --ply outputs/building/aerial_regions_parking_bev_crop/regions/region_000/chunk.ply \
  --stage-dir outputs/building/parking_region_000_instance_seg_balanced_recall_identitydiag_nearcam_sixrings_allviews/stage1 \
  --output-dir outputs/building/test_stage2_object_refinement_generate_only \
  --device cuda \
  --backend sv3d \
  --refinement-stage generate_only \
  --generation-model sv3d \
  --generation-model-path /home/hongjiacheng/codes/gs_lab/worldact/sv3d \
  --generation-resolution 512 \
  --num-generation-views 6 \
  --only-object-id 1
```

Important artifacts:

- `clean_input.png` and `mask.png`: foregroundized single-view SV3D condition.
- `view_scores.json`: clean-view selection scores.
- `multiview_generate/view_*.png`: SV3D pseudo views.
- `pseudo_view_metrics.json`: pseudo-view foreground statistics.

### Phase 3: Local GS Refinement And Merge

Run cleanup, pseudo-view generation, and local object-only GS painting/refinement.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/hongjiacheng/miniconda3/envs/worldact/bin/python scripts/refine_objects.py \
  --ply outputs/building/aerial_regions_parking_bev_crop/regions/region_000/chunk.ply \
  --stage-dir outputs/building/parking_region_000_instance_seg_balanced_recall_identitydiag_nearcam_sixrings_allviews/stage1 \
  --output-dir outputs/building/test_stage2_object_refinement_local_refine \
  --device cuda \
  --backend sv3d \
  --refinement-stage local_refine \
  --generation-model sv3d \
  --generation-model-path /home/hongjiacheng/codes/gs_lab/worldact/sv3d \
  --generation-resolution 512 \
  --num-generation-views 6 \
  --local-refine-iterations 80 \
  --local-refine-lr 0.03 \
  --only-object-id 1
```

Important artifacts:

- `paint_metrics.json`: local refinement diagnostics.
- `object_<id>_<name>_final.ply`: final refined object PLY.

### Legacy Full SV3D Reconstruction

To run the original full generation/reconstruction/merge path, use `--refinement-stage legacy`:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/hongjiacheng/miniconda3/envs/worldact/bin/python scripts/refine_objects.py \
  --ply outputs/building/aerial_regions_parking_bev_crop/regions/region_000/chunk.ply \
  --stage-dir outputs/building/parking_region_000_instance_seg_balanced_recall_identitydiag_nearcam_sixrings_allviews/stage1 \
  --output-dir outputs/building/test_stage2_object_refinement_legacy_sv3d \
  --device cuda \
  --backend sv3d \
  --refinement-stage legacy \
  --generation-model sv3d \
  --generation-model-path /home/hongjiacheng/codes/gs_lab/worldact/sv3d \
  --reconstruction-model instant_splat \
  --generation-resolution 512 \
  --num-generation-views 6 \
  --only-object-id 1
```

## <a name="results"></a>Results

We present the performance of our proposed WorldAct framework.

<details open>
<summary>Quantitative Results (click to expand)</summary>

- Results in Tab. 1 and Tab. 2 of the main paper

  ![table](assets/Table.png)

</details>

<details open>
<summary>Qualitative Results (click to expand)</summary>

- Results in Fig. 4 of the main paper

  ![fig4](assets/fig4.png)

- Results in Fig. 5 of the main paper

  ![fig5](assets/fig5.png)

</details>

## <a name="citation"></a>Citation

If you find our model or code helpful in your research or work, please cite the following paper.

```bibtex
@article{hu2026worldact,
      title={WorldAct: Activating Monolithic 3D Worlds into Interactive-Ready Object-Centric Scenes},
      author={Jichen Hu and Jiawei Guo and Jiazhong Cen and Chen Yang and Sikuang Li and Wei Shen},
      journal={arXiv preprint arXiv:2605.15843},
      year={2026}
}
```

## <a name="acknowledgements"></a>Acknowledgements

This project builds on numerous model repositories. We thank [Marble](https://marble.worldlabs.ai/) and [SAM3D](https://ai.meta.com/research/sam3d/) for the research ideas that inspired this work.

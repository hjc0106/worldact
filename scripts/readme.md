# Scripts 使用说明

本文档按当前推荐工作流组织：先做 **Stage 1 区域级场景分割**，得到区域级 `chunk.ply` 和该区域的推荐相机配置；再做 **Stage 2 实例级场景分割**，在区域 chunk 内执行细粒度目标分割。

所有命令默认从项目根目录执行：

```bash
cd /home/hongjiacheng/Codes/gs_lab/WorldAct_Codex
source "/home/hongjiacheng/miniconda3/etc/profile.d/conda.sh"
conda activate worldact
```

## 总体流程

推荐两阶段：

```text
完整 scene.ply
  -> Stage 1: segment_regions_aerial.py
     航拍图 + SAM3 prompt + BEV/direct bbox
     输出区域级 regions/region_*/chunk.ply
     同时输出 regions/region_*/region_camera_config.yaml

  -> Stage 2A: run_decomposition.py 或 run_seg.sh
     直接使用 Stage 1 的 region_camera_config.yaml
     对整个区域 chunk 做实例级分割

  -> Stage 2B: localize_roi_center.py，可选
     如果只想聚焦区域内更小目标，再用 prompt 重定位中心
     输出更细粒度的 roi_camera_config.yaml
```

关键原则：

- Stage 1 只负责“区域级场景划分”，目标是导出覆盖目标区域的 `chunk.ply`，并给出整个 chunk 的推荐相机中心和环视半径。
- Stage 2 才做“实例级目标分割”，不要在全场景里直接细分目标。
- 如果不想在 Stage 2 再提供中心定位 prompt，直接使用 Stage 1 生成的 `region_camera_config.yaml`。
- 如果区域 chunk 仍然很大，或者只想聚焦 chunk 内某个子区域，再运行 `localize_roi_center.py` 做二次定位。
- 大规模室外场景建议显式指定 `--up-axis=-y`，不要依赖 auto。
- Stage 1 默认 `height_mode=full`，只按水平 bbox 裁剪，不限制高度，减少漏掉区域内物体的风险。

## 脚本概览

| 脚本 | 作用 |
|------|------|
| `segment_regions_aerial.py` | Stage 1 区域级场景分割：航拍图 + SAM3 prompt + BEV/direct bbox，导出区域 `chunk.ply` |
| `render_aerial_views.py` | 只保存密集航拍图，用于检查大场景俯视覆盖情况 |
| `localize_roi_center.py` | Stage 2 可选前置：在区域 chunk 内进一步定位更小 ROI，生成 `roi_camera_config.yaml` |
| `run_decomposition.py` | Stage 2 核心入口：手动 prompt/VLM + SAM3 + 3DGS assignment |
| `run_seg.sh` | `run_decomposition.py` 的封装脚本，可用 `OBJECTS=...` 走手动 prompt |
| `run_seg_vp.sh` | 带 view propagation 默认参数的封装脚本 |
| `visualize_cameras.py` | 检查相机轨迹和 keyframe render，不跑 SAM3/VLM |
| `render_chunks.py` | 旧版几何网格 chunk 预览和 prompt-guided chunk 筛选，主要作对比/调试 |
| `convert_gaussianeditor_dataset.py` | 从 GaussianEditor 数据集中导出训练后的标准 3DGS PLY |

OmegaConf 覆盖项可以追加在命令末尾：

```bash
stage1.camera_gen.up_axis=-y stage1.view_propagation.enabled=false
```

## Stage 1: 区域级场景分割

Stage 1 输入完整场景 PLY，输出一个或多个区域级 PLY：

```text
outputs/.../regions/region_000/chunk.ply
outputs/.../regions/region_000/bbox.json
outputs/.../regions/region_000/region_camera.json
outputs/.../regions/region_000/region_camera_config.yaml
outputs/.../regions/region_000/region_center_visualization.png
outputs/.../region_manifest.json
```

### 1.1 Building 停车区域

BEV 融合版本：

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/segment_regions_aerial.py \
  --ply demo/mill-19/building.ply \
  --config configs/scenes/outdoor.yaml \
  --output-dir outputs/building/aerial_regions_parking_bev \
  --device cuda \
  --up-axis=-y \
  --prompt "parking lot" \
  --prompt "black car" \
  --prompt "white car" \
  --max-views 120 \
  --bbox-margin 1.5 \
  --keep-top-k 3 \
  --region-mode bev
```

BEV 裁剪版本（`aerial_regions_parking_bev_crop`，用于更严格的高度裁剪）：

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/segment_regions_aerial.py \
  --ply demo/mill-19/building.ply \
  --config configs/scenes/outdoor.yaml \
  --output-dir outputs/building/aerial_regions_parking_bev_crop \
  --device cuda \
  --up-axis=-y \
  --prompt "parking lot" \
  --prompt "black car" \
  --prompt "white car" \
  --max-views 120 \
  --bbox-margin 1.5 \
  --keep-top-k 3 \
  --region-mode bev \
  --height-mode quantile \
  --height-quantile 0.0 0.995 \
  --height-margin 0.0 \
  --max-scale 3.0
```

非 BEV baseline，用于对比：

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/segment_regions_aerial.py \
  --ply demo/mill-19/building.ply \
  --config configs/scenes/outdoor.yaml \
  --output-dir outputs/building/aerial_regions_parking_direct \
  --device cuda \
  --up-axis=-y \
  --prompt "parking lot" \
  --prompt "black car" \
  --prompt "white car" \
  --max-views 120 \
  --bbox-margin 1.5 \
  --region-mode direct_bbox
```

### 1.2 Garden 圆桌区域

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/segment_regions_aerial.py \
  --ply demo/garden-7k.ply \
  --config configs/scenes/outdoor.yaml \
  --output-dir outputs/garden/aerial_regions_table_bev \
  --device cuda \
  --up-axis=-y \
  --prompt "round table" \
  --prompt "vase on table" \
  --max-views 80 \
  --bbox-margin 1.0 \
  --keep-top-k 2 \
  --region-mode bev
```

### 1.3 Stage 1 输出怎么看

主要文件：

```text
outputs/.../aerial_views/                 每个航拍视角的 RGB 图
outputs/.../aerial_contact_sheet.png      航拍图总览
outputs/.../aerial_plan.png               航拍相机采样点俯视示意
outputs/.../aerial_masks/                 SAM3 mask 和 overlay
outputs/.../aerial_masks.json             每个 prompt/view 的 candidate 记录
outputs/.../occupancy.png                 BEV 模式的 occupancy 和 region bbox
outputs/.../region_manifest.json          最终区域列表
outputs/.../regions/region_000/chunk.ply  区域级场景 PLY
outputs/.../regions/region_000/bbox.json  区域 bbox 和参数记录
outputs/.../regions/region_000/region_camera.json                 推荐相机中心、bbox center、robust center 和半径
outputs/.../regions/region_000/region_camera_config.yaml           可直接用于 Stage 2 的相机配置
outputs/.../regions/region_000/region_center_visualization.png     中心点质量检查图
outputs/.../regions/region_000/recommended_camera_overrides.txt    不使用 YAML 时的覆盖项
```

`--region-mode bev` 会生成 `occupancy.png`；`--region-mode direct_bbox` 不生成 BEV occupancy，而是直接把所有有效 mask 反投影点合成一个水平 bbox。

常用调参：

```bash
--max-views 160              # 更多航拍视角
--grid-cols 12 --grid-rows 8 # 固定航拍网格
--bbox-margin 2.0            # 扩大导出的区域
--occupancy-threshold 0.2    # BEV 区域更宽松
--keep-top-k 5               # 保留更多候选区域
--candidate-mode best        # 每个 prompt/view 只取最高分 mask
--candidate-mode union       # 每个 prompt/view 合并多个候选 mask
--region-camera-purpose coverage     # 默认：面向整个 region chunk 的覆盖轨迹
--region-camera-coverage-mode cylinder # 单中心覆盖圆柱/覆盖球第一阶段使用同一水平 footprint
--region-camera-standoff-scale 0.35  # orbit_radius = coverage_radius + standoff
--region-camera-min-standoff 1.0     # 最小退让距离，避免相机贴近区域边界
--region-camera-height-mode geometric # 默认：垂直中心为 robust 几何高度范围中点
--region-camera-floor-quantile 0.02  # 用区域高斯高度 2% 分位估计 floor
--region-camera-height-quantile 0.98 # 用区域高斯高度 98% 分位估计 top
--region-camera-height-mode adaptive  # 可选：按比例从 floor 抬高
--region-camera-height-scale 0.35    # adaptive_height = 0.35 * 主体高度
--region-camera-height-min 0.5       # adaptive 高度下限
--region-camera-height-max 2.0       # adaptive 高度上限
--region-camera-height-mode fixed --region-camera-height 1.0 # 复现固定 floor + 1.0
--region-camera-radius-scale 0.65    # 按区域水平 bbox 对角线估计环视半径
--no-region-camera-config            # 不导出 Stage 2 相机配置和中心可视化
```

`coverage` 模式下，水平中心使用 region bbox center，半径使用：

```text
coverage_radius = 0.5 * sqrt(span0^2 + span1^2)
orbit_radius = coverage_radius + max(min_standoff, standoff_scale * coverage_radius)
center_up = 0.5 * (height_q02 + height_q98)
```

如果需要回到旧的中心估计方式用于对比：

```bash
--region-camera-purpose object \
--region-camera-center-source volume # 或 robust/bbox/hybrid
```

## Stage 2: 实例级场景分割

Stage 2 输入 Stage 1 的 `chunk.ply`。默认推荐先直接使用 Stage 1 生成的 `region_camera_config.yaml`；如果区域内还需要更细粒度聚焦，再运行 `localize_roi_center.py`。

### 2.1 检查 Stage 1 区域相机配置

以 Stage 1 输出的 `region_000/chunk.ply` 为例：

```text
outputs/garden/aerial_regions_table_bev/regions/region_000/region_camera_config.yaml
outputs/garden/aerial_regions_table_bev/regions/region_000/region_center_visualization.png
outputs/garden/aerial_regions_table_bev/regions/region_000/region_camera.json
```

`region_camera_config.yaml` 可以直接作为 Stage 2 的 `--config`。先看 `region_center_visualization.png`，确认红色 `camera center` 是否位于整个区域 bbox 的覆盖中心附近，蓝色 `coverage radius` 是否覆盖目标区域。垂直坐标默认不是高斯 median，也不是固定 1.0，而是区域 robust 几何高度范围的中点，减少地面高斯密度对中心的影响。

如果 `camera center` 明显偏离目标区域，可以重跑 Stage 1 并切换：

```bash
--region-camera-height-mode adaptive
--region-camera-height-mode fixed --region-camera-height 1.0
```

### 2.2 可选：在区域 chunk 内进一步定位中心点

当 Stage 1 的区域 chunk 仍然包含较多背景，或者你只想围绕 chunk 内更小目标采样时，再运行：

```bash
python scripts/localize_roi_center.py \
  --ply outputs/garden/aerial_regions_table_bev/regions/region_000/chunk.ply \
  --output-dir outputs/garden/region_000_roi_probe \
  --device cuda \
  --up-axis=-y \
  --num-views 16 \
  --prompt "round table with a vase" \
  --base-config configs/scenes/outdoor.yaml
```

查看：

```text
outputs/garden/region_000_roi_probe/probe_contact_sheet.png
outputs/garden/region_000_roi_probe/roi_masks/best_view_*_roi_overlay.png
outputs/garden/region_000_roi_probe/roi_topdown_render_closeup.png
outputs/garden/region_000_roi_probe/roi_center.json
outputs/garden/region_000_roi_probe/roi_camera_config.yaml
```

如果只想先看探索图，不跑 SAM3 prompt，可以去掉 `--prompt`：

```bash
python scripts/localize_roi_center.py \
  --ply outputs/garden/aerial_regions_table_bev/regions/region_000/chunk.ply \
  --output-dir outputs/garden/region_000_roi_probe_preview \
  --device cuda \
  --up-axis=-y \
  --num-views 16
```

### 2.3 执行实例级分割：手动 prompt

推荐直接调用 `run_decomposition.py`，不依赖 VLM credentials：

```bash
python scripts/run_decomposition.py \
  --ply outputs/garden/aerial_regions_table_bev/regions/region_000/chunk.ply \
  --config outputs/garden/aerial_regions_table_bev/regions/region_000/region_camera_config.yaml \
  --objects "vase with flowers on the round table" "round table" \
  --save-keyframes \
  --save-sam3-masks \
  --output-dir outputs/garden/region_000_instance_seg \
  --device cuda \
  stage1.chunking.enabled=false \
  stage1.camera_gen.up_axis=-y \
  stage1.view_propagation.enabled=false
```

输出：

```text
outputs/garden/region_000_instance_seg/stage1/keyframes/
outputs/garden/region_000_instance_seg/stage1/objects.json
outputs/garden/region_000_instance_seg/stage1/object_*.ply
```

如果你已经运行了 `localize_roi_center.py` 做二次定位，就把 `--config` 换成对应的 `roi_camera_config.yaml`。

### 2.4 执行实例级分割：使用 `run_seg.sh`

如果只分割一个手动 prompt，可以用封装脚本：

```bash
PLY_PATH=outputs/garden/aerial_regions_table_bev/regions/region_000/chunk.ply \
CONFIG_PATH=outputs/garden/aerial_regions_table_bev/regions/region_000/region_camera_config.yaml \
OUTPUT_DIR=outputs/garden/region_000_instance_seg_runseg \
DEVICE=cuda \
OBJECTS="vase with flowers on the round table" \
bash scripts/run_seg.sh \
  stage1.chunking.enabled=false \
  stage1.camera_gen.up_axis=-y \
  stage1.view_propagation.enabled=false
```

如果不设置 `OBJECTS`，`run_seg.sh` 会走 VLM 自动发现对象，需要 `openai.env` 或环境变量中包含：

```bash
OPENAI_API_BASE=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5
```

### 2.5 View Propagation 可选版本

对跨视角漏检明显的区域，可以开启传播。对重复实例较多的场景，优先使用 identity-safe 设置：

```bash
python scripts/run_decomposition.py \
  --ply outputs/garden/aerial_regions_table_bev/regions/region_000/chunk.ply \
  --config outputs/garden/aerial_regions_table_bev/regions/region_000/region_camera_config.yaml \
  --objects "wood chair with green cushion" \
  --save-keyframes \
  --save-sam3-masks \
  --output-dir outputs/garden/region_000_chair_prop \
  --device cuda \
  stage1.chunking.enabled=false \
  stage1.camera_gen.up_axis=-y \
  stage1.view_propagation.enabled=true \
  stage1.view_propagation.save_debug=true \
  stage1.view_propagation.fallback_to_sam3=false \
  stage1.view_propagation.candidate_unique_assignment=true \
  stage1.view_propagation.fusion_mode=intersection \
  stage1.view_propagation.max_hops=1 \
  stage1.view_propagation.max_angle_deg=45 \
  stage1.view_propagation.min_view_overlap=0.2 \
  stage1.view_propagation.min_assignment_iou=0.6 \
  stage1.view_propagation.min_assignment_visibility=0.25
```

传播调试输出：

```text
outputs/.../stage1/view_propagation/
outputs/.../stage1/tracks.json
outputs/.../stage1/propagation_stats.json
```

## Stage 3: 前景目标级 3DGS 补全与细化

Stage 3 输入 Stage 2 的 `stage1/` 输出，只处理前景 `object_*.ply` / `objects.json` / `masks.pt` 对应的目标级 3DGS，不训练原始 scene 3DGS，也不处理背景 inpainting。

默认先跑无生成后端版本，用于诊断、剪枝去背景干扰，并导出后续前馈网络需要的 crop package：

```bash
python scripts/refine_objects.py \
  --ply outputs/building/aerial_regions_parking_bev/regions/region_000/chunk.ply \
  --stage-dir outputs/building/parking_region_000_instance_seg_depthaware_cleanup/stage1 \
  --output-dir outputs/building/parking_region_000_object_refine \
  --device cuda \
  --backend none \
  --opacity-min 0.02 \
  --component-radius 1.2 \
  --min-mask-support 1 \
  --depth-tolerance 0.08
```

主要输出：

```text
outputs/.../object_refine/object_metrics.json
outputs/.../object_refine/object_<id>_<name>/object_refined_seed.ply
outputs/.../object_refine/object_<id>_<name>/object_<id>_<name>_final.ply
outputs/.../object_refine/object_<id>_<name>/crop.png
outputs/.../object_refine/object_<id>_<name>/mask.png
outputs/.../object_refine/object_<id>_<name>/object_request.json
```

其中 `object_refined_seed.ply` 是经过 opacity、scale、主连通体、mask/depth consistency 剪枝后的目标 GS；`object_*_final.ply` 在 `--backend none` 时等同于 refined seed。

如果要接入 LGM、CompleteSplat、Splatter Image 等外部前馈后端，使用 `--backend external`。命令会收到 `object_request.json` 路径，并需要在同一个 object 工作目录写出：

```text
generated.ply       # 外部前馈网络生成的目标级 3DGS
metadata.json       # 可选；可包含 {"transform": [[... 4x4 ...]]}
```

示例：

```bash
python scripts/refine_objects.py \
  --ply outputs/building/aerial_regions_parking_bev/regions/region_000/chunk.ply \
  --stage-dir outputs/building/parking_region_000_instance_seg_depthaware_cleanup/stage1 \
  --output-dir outputs/building/parking_region_000_object_refine_lgm \
  --device cuda \
  --backend external \
  --backend-command 'python /path/to/feedforward_backend.py --request {request}' \
  --backend-timeout 900
```

外部后端未写出 `generated.ply` 时，流程不会凭空补几何，会保留 deterministic refined object；写出 `generated.ply` 时，只会在目标 mask 内且 refined object 渲染 alpha 较低的区域保守合并 generated splats。

## 辅助调试脚本

### 只保存航拍图

用于在 Stage 1 前检查航拍相机是否覆盖目标区域：

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/render_aerial_views.py \
  --ply demo/mill-19/building.ply \
  --output-dir outputs/building/aerial_views \
  --device cuda \
  --up-axis=-y \
  --max-views 120 \
  --image-width 640 \
  --image-height 640 \
  --fov-deg 70 \
  --overlap 0.35
```

### 只预览实例级相机轨迹

```bash
python scripts/visualize_cameras.py \
  --ply outputs/garden/aerial_regions_table_bev/regions/region_000/chunk.ply \
  --config outputs/garden/region_000_roi_probe/roi_camera_config.yaml \
  --output-dir outputs/garden/region_000_camera_vis \
  --device cuda \
  --num-keyframes 40 \
  --save-renders
```

### 旧版几何 chunk 预览

`render_chunks.py` 是旧版几何网格 chunk 方案，当前主要用于对比和调试：

```bash
python scripts/render_chunks.py \
  --ply demo/garden-7k.ply \
  --config configs/scenes/outdoor.yaml \
  --output-dir outputs/garden_chunk_prompt_vase_table \
  --device cuda \
  --export-selected-ply \
  --render-selected-only \
  --save-probe-views \
  stage1.chunking.enabled=true \
  stage1.chunking.prompt="round table with a vase" \
  stage1.chunking.min_roi_overlap=0.15 \
  stage1.chunking.keep_top_k=2 \
  stage1.chunking.roi_margin=0.5 \
  stage1.camera_gen.up_axis=-y
```

## 常见检查项

### Stage 1 没有输出 region

先看：

```text
aerial_contact_sheet.png
aerial_masks/
aerial_masks.json
occupancy.png
aerial_region_summary.json
```

常见处理：

- prompt 改短、更对象化，例如 `"parking lot"`、`"black car"`、`"round table"`。
- 增大 `--max-views` 或固定 `--grid-cols/--grid-rows`。
- 降低 `--occupancy-threshold`，例如 `0.25 -> 0.15`。
- 降低 `--min-component-area`。
- 对比 `--region-mode direct_bbox` 判断是不是 BEV threshold 太严格。
- 确认 `--up-axis=-y` 是否正确；方向不对时 BEV 平面会错。

### Stage 1 导出的区域太大或太小

- 太小：增大 `--bbox-margin`，或降低 `--occupancy-threshold`。
- 太大：减小 `--bbox-margin`，用 `--candidate-mode best`，或提高 `--occupancy-threshold`。
- 多个区域混在一起：优先用 `--region-mode bev`，并适当提高 `--occupancy-threshold`。

### Stage 2 相机中心不对

先只跑 `localize_roi_center.py` 不带 `--prompt`，查看 `probe_contact_sheet.png`。确认目标在探索图中可见后，再带 prompt 自动定位。负向 up axis 要写成：

```bash
--up-axis=-y
```

不要写成：

```bash
--up-axis -y
```

### Stage 2 分割结果异常

- 先关闭传播：`stage1.view_propagation.enabled=false`。
- 看 `stage1/keyframes/*_sam3_overlay.png`，确认 SAM3 单视角 mask 是否合理。
- 再开启 identity-safe propagation，对比 `stage1/view_propagation/*_propagated.png`。
- 重复实例混入时，优先设置 `stage1.view_propagation.fallback_to_sam3=false`。

## GaussianEditor 数据集导出

GaussianEditor 数据目录里常见两类 PLY：

- `input.ply`：COLMAP/初始化点云，不能作为本项目 3DGS 输入。
- `point_cloud/iteration_*/point_cloud.ply`：训练后的 Gaussian splat，可以作为本项目输入。

检查将要导出的文件：

```bash
python scripts/convert_gaussianeditor_dataset.py \
  --dataset-root /home/hongjiacheng/Codes/gs_lab/GaussianEditor/dataset \
  --output-dir demo/gaussianeditor \
  --scenes garden bonsai \
  --iteration latest \
  --dry-run
```

确认无误后导出，默认建议软链接：

```bash
python scripts/convert_gaussianeditor_dataset.py \
  --dataset-root /home/hongjiacheng/Codes/gs_lab/GaussianEditor/dataset \
  --output-dir demo/gaussianeditor \
  --iteration latest \
  --mode symlink \
  --overwrite
```

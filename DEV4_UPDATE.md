# dev-4 更新说明

创建日期：2026-10-20。

## 分支基线与本次变更

- 基于 `dev-3` 的 `961ea1dfc40875d2e0ea9bfbfe66ea19e05abfa3` 创建 `dev-4`。
- 本次新增本文档，整理已有实验支持能力、代码位置及使用边界，没有新增算法实现。
- 下文列出的源码功能均已包含在 `dev-3`，由 `dev-4` 继承，不能作为本次新增功能统计。
- ArtiFixer 主仓库中的实验脚本、数据和蒸馏改动不属于本 WorldAct 独立仓库的提交范围。

## 从 dev-3 继承的主要实验支持

### 1. SAM3 分割输入与候选选择

文件：`worldact/stage1_decomposition/sam3_segmenter.py`、`worldact/config.py`。

- 支持 `best_quality` 候选选择策略，实际启用取决于实验配置；默认策略没有因此自动改变。
- 图像转换支持 HWC、CHW、单张 batch，以及浮点、布尔和 uint8 等输入，统一转为 RGB。
- 用于避免输入布局错误，并允许按质量选择目标 mask。这些修改没有重写 Stage 1 的高斯归属优化核心。

### 2. 高斯渲染与贡献归因诊断

文件：`worldact/rendering/rasterizer.py`。

- 增加高斯到像素的可见贡献索引、alpha 与前向合成权重查询。
- 增加可见命中数、有效贡献数及投影分辨率等统计。
- 自动寻找当前 Python 环境中的 Ninja，支持 gsplat 扩展编译。
- 为残影来源、弱贡献点和遮挡分析提供接口；诊断接口本身不自动清理、删除或优化场景高斯。

### 3. 上方向估计与物体环绕轨迹

文件：`worldact/stage2_artifixer/up_axis.py`、`trajectory.py`，以及 `scripts/regenerate_object_trajectory.py`。

- 支持手动上方向、场景地面、物体几何及相机提示等来源，并记录估计来源与置信度。
- 支持物体直立环绕、俯仰角、观察中心、相机距离和边界控制。
- 用于减少绕错轴、车辆倾斜或出框。估计结果仍需检查，不能保证所有场景自动对齐。

### 4. 锚点相机、稀疏 mask 和精修 PLY 输入

文件：`worldact/stage2_artifixer/anchors.py`、`inputs.py`，以及 `scripts/run_artifixer_refinement.py`。

- 适配远景中占比很小的目标 mask，改善锚点筛选与错误提示。
- 支持物体居中的锚点相机，并处理源图裁剪后的主点及画面重排。
- 导出黑底物体锚点时去除白底合成项，并保留软 alpha 边缘。
- 输入准备支持指定精修物体 PLY，减少后续误用旧物体版本的问题。
- 无 mask 时的锚点覆盖回退使用投影支持区域，而不是直接把整幅图作为物体。

### 5. 修复任务提交、结果导入与运行稳定性

文件：`scripts/submit_artifixer_package.py`、`scripts/run_artifixer_refinement.py`，以及 `worldact/stage2_artifixer/remote_client.py`、`runner.py`、`result_import.py`。

- 完善远程修复服务的输入包提交、任务结果接收和导入。
- 增加多 GPU 环境默认值及退出时资源清理等处理。
- 主要影响任务运行和输入输出管理，不代表修复模型画质得到提升。

### 6. 实验配置与检查

文件：`configs/artifixer/car_front_views.yaml`、`car_front_upright.yaml`、`car_preview.yaml` 及相关测试。

- 提供车辆实验的视角、分辨率、视场角、环绕和 mask 筛选参数。
- 包含针对 SAM3 输入和 Stage 2 数据包等功能的测试。本文档更新没有重新运行整套功能测试。

## 使用边界

- 启用 ROI 配置不等于实际 ROI 有效；由异常 alpha 生成的全屏 mask 仍可能失去车辆外保护。
- 相机坐标对应正确，不保证独立重建间的局部道路高度和车辆接地位置一致。
- 贡献归因只能说明可见性与遮挡关系，不能单独证明某个高斯属于车辆。
- `scripts/regenerate_object_trajectory.py.orig` 是当前工作区未跟踪的备份文件，不纳入本次文档提交。

## 本次提交范围

只提交 `DEV4_UPDATE.md`。本次只需检查文档差异；没有运行训练、生成新修复结果或修改模型参数。

# 红外—卫星检索头训练与 Windows 离线交付

GitHub 仓库中，本训练工程位于 `training/`。以下源码相对路径和训练命令以该目录为基准；使用已安装的离线部署包时，工程目录仍是部署目标下的 `project/`。源码仓库不包含环境或离线安装材料。

本工程默认冻结 DINOv2 ViT-L/14，训练两个独立 MLP 检索头；另提供在线微调 DINO 的 LoRA 备选方案。适用于四组单通道 IR（DS/DV/NS/NV）与单通道卫星候选图。当前交付是可运行工程、依赖和离线恢复工具；真实训练数据、DSM、安装标定和比赛机器最新修改版未随交接包提供。目标驱动后续由用户提供为 527.27，尚未在目标机实测。**没有已训练好的真实业务权重，也没有真实精度结论。**

先看 `docs/验证状态.md`，它区分已执行检查和待服务器/真实数据核验的项目。`docs/方法与边界.md` 说明几何、数据、损失；`docs/接入说明.md` 说明实际源码接入点和 .so/Orin 边界。

## 最新完整包：一次安装全部功能

**2026-11-05 整合版 `offline_rebuild_B.zip` 已直接包含所有最新代码。新安装不需要再安装任何增补包或依次打补丁。** 包含几何检查、调试/HTML/CSV/候选图输出，以及以下六个训练入口；原单卡基线仍保留。

| 方案 | PowerShell 入口 | 默认输出目录 |
|---|---|---|
| MLP 单卡 | `scripts/train.ps1` | `artifacts/real/training` |
| MLP 双卡 DDP | `scripts/train_mlp_ddp.ps1` | `artifacts/real/mlp_ddp` |
| LoRA 单卡 | `scripts/train_lora.ps1` | `artifacts/real/lora` |
| LoRA 双卡 DDP | `scripts/train_lora_ddp.ps1` | `artifacts/real/lora_ddp` |
| LoRA 外观增强，单卡 | `scripts/train_lora_aug.ps1` | `artifacts/real/lora_aug` |
| LoRA 外观增强，双卡 DDP | `scripts/train_lora_aug_ddp.ps1` | `artifacts/real/lora_aug_ddp` |

下载这个完整 ZIP 并解压到 `D:\SCC_bundle`，安装到新的空目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\SCC_bundle\offline_install.ps1 -BundleRoot D:\SCC_bundle -Destination D:\SCC_Retrieval
```

安装器默认自动选择：仅下载完整 ZIP 时使用离线 B 重建；如果解压目录另有 A 环境归档，则使用 A。可以继续显式指定 `-Method A` 或 `-Method B`。环境保持 Python3.10.16 / PyTorch2.5.1+cu118，不新增依赖，不包含 DINO 权重，安装后配置服务器已有权重和真实数据。

完成数据准备后，在 `D:\SCC_Retrieval\project` 中选择上述入口并传 `-EnvDir D:\SCC_Retrieval\env`。MLP 使用已提取的冻结特征缓存，LoRA 在线读取原图；数据增强本次仅覆盖 LoRA。默认弱增强保留30%原图，只改变灰度外观；挖掘、验证、测试与推理均不增强。

详细说明：[LoRA 双卡](docs/双卡备选方案.md)、[MLP 双卡](docs/MLP双卡备选方案.md)、[数据增强](docs/数据增强.md)。`release_manifest.json` 记录完整功能及源文件 SHA256，`validation/` 包含各阶段和完整包恢复验证证据。历史增补包只供旧部署单独更新；新完整包无需使用它们。

## 交付物

准备机训练工作区：`C:\Users\taizun\Documents\ChatGPT\IR_Retrieval\training`。

大文件交付目录：`D:\IR_Retrieval_delivery`，不包含真实数据。**按服务器已有权重的部署方式，当前交付包已移除 DINO 基座权重；安装后先配置服务器已有的 DINO 权重路径，再运行自检和训练。**

| 文件/目录 | 用途 |
|---|---|
| source.zip | 工程、中文文档、脚本、测试和原交接参考附件 |
| environment_win64_py310_cu118.zip | conda-pack 环境，包含 Python；Windows x64 专用 |
| offline_rebuild_B.zip | B 方式完整代码与离线环境安装材料，内含独立哈希清单；DINO 权重由服务器提供，不含 A 环境归档 |
| model_and_source.zip | 固定 commit 的 DINOv2 源码及许可证；沿用原文件名，已移除 DINO 权重 |
| wheelhouse/、requirements.lock.txt | CPython 3.10 Windows/通用 wheels，精确版本和 SHA256 |
| conda_packages/、conda_explicit.json | Python 及基础运行库的本地 conda 包，离线恢复无需求解下载 |
| installers/ | 固定版 Miniconda x64、已验签 VC++ 运行库安装器 |
| asset_sources.json、wheel_sources.json | 来源、版本与哈希 |
| SHA256SUMS.txt | 整包文件校验清单 |
| release_manifest.json | 已集成的全部功能、环境版本、工程源文件哈希 |
| validation/ | 实际测试报告与合成诊断图，不包含合成训练权重 |

PyTorch2.5.1+cu118 / torchvision0.20.1+cu118 是本准备机已验证组合，目标服务器仍需验收。采用 CUDA11.8 的保守配套 Windows 驱动基线 522.06；较旧驱动的兼容情况需另验证，脚本不擅自安装或更换 NVIDIA 驱动。不要把 nvidia-smi 显示的最高 CUDA 版本理解为必须安装相同版本的 Toolkit。

## 0. 精简 MLP 流程（推荐）

只需编辑 `configs/mlp.yaml`（数据/卫星/DSM 路径、序列划分、DINO 权重、确认标志），其余参数继承 `configs/train.yaml`。先小样本检查，再全量：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mlp.ps1 -EnvDir D:\SCC_Retrieval\env -Limit 12 -Steps prepare
powershell -ExecutionPolicy Bypass -File scripts\run_mlp.ps1 -EnvDir D:\SCC_Retrieval\env
```

候选块由推理原代码生成（地图网格、搜索窗、旋转、块尺寸、重叠、黑边过滤都与部署相同），`inference` 段照抄部署 naviParams 的 infrared 段即可。搜索中心为真值加 ≤`center_offset_m` 随机偏移（训练窗口 `train_window_m`，验证/测试窗口 `eval_window_m` = 推理捕获窗口），训练集只保留全部正例＋`train_negatives` 个负例，验证/测试按 `eval_query_stride` 抽帧。`preparation_report.json` 中的 `candidates_per_split`、`covered_queries`、`block_side_m`、`satellite_intensity` 可直接核对。说明见 [训练推理对齐](docs/训练推理对齐.md) 末节。

## 1. 离线服务器恢复

复制整个交付目录到服务器，例如 `D:\SCC_bundle`。PowerShell 默认脚本策略若禁止运行，可仅对本次进程使用 `powershell -ExecutionPolicy Bypass -File ...`。不需要 Docker、WSL、NCCL、Triton 或编译 xFormers。

方式 A（已另行取得 A 环境归档时可选，独立最终目录）：

```powershell
powershell -ExecutionPolicy Bypass -File D:\SCC_bundle\offline_install.ps1 `
  -BundleRoot D:\SCC_bundle -Destination D:\SCC_Retrieval_A -Method A
```

脚本首先校验所有哈希，解压至 `env`，设置 Windows DLL 搜索路径并运行 `Scripts\conda-unpack.exe`。目标目录必须为空。**完成重定位后不要移动环境**，如需换路径应从原包重新解压。VC++ 缺失时增加 `-InstallVCRuntime`，安装可能需要管理员权限；不默认触碰宿主运行库。

方式 B（完整 ZIP 的默认方式，从完整本地包重建）：

```powershell
powershell -ExecutionPolicy Bypass -File D:\SCC_bundle\offline_install.ps1 `
  -BundleRoot D:\SCC_bundle -Destination D:\SCC_Retrieval_B -Method B
```

可用 `-CondaExe C:\已有Miniconda\Scripts\conda.exe`；否则脚本使用包内 Miniconda 安装器创建新的私有目录，不注册系统 Python。然后通过本地 @EXPLICIT 清单 `conda create --offline`，以及 `pip --no-index --require-hashes` 安装完整依赖。恢复绝不在线 pip/conda 求解。

若本机安装器只留下 `_conda.exe` 独立引导程序，脚本验证其版本后使用它重建目标环境，不依赖引导目录成为完整 Python 环境。也可以单独解压 `offline_rebuild_B.zip` 并把解压目录传给 `-BundleRoot`，仅使用 `-Method B`。

**先接入服务器已有 DINO 权重，再验收。** 编辑安装后 `project/configs/train.yaml` 中 `model` 下的这两项，其他设置保持原样：

```yaml
model:
  checkpoint: 'D:/SCC_Loc/实际权重目录/dinov2_vitl14_pretrain.pth'
  checkpoint_sha256: '服务器实际DINO权重文件的64位SHA256'
```

路径为示例，填写服务器实际位置；两种方案共用这份配置，无需再次下载或复制权重到部署包。可用以下命令读取文件 SHA256：

```powershell
(Get-FileHash -LiteralPath 'D:\SCC_Loc\实际权重目录\dinov2_vitl14_pretrain.pth' -Algorithm SHA256).Hash.ToLower()
```

这里需要非 register DINOv2 ViT-L/14 的骨干 state_dict，不能填写 `minima_roma.pth` 等整套匹配网络权重。官方参考 SHA256 为 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`；如果不同，先确认确为 SCC 实际使用的 DINO，并通过严格结构加载及部署一致性检查。不要清空哈希校验或仅凭文件存在将 `deployed_backbone_confirmed` 置 true。模型实现和预处理仍需对齐。

验收（默认要求两卡；准备机/单卡回退用 `-ExpectedGPUs 1`）：

```powershell
cd D:\SCC_Retrieval_A\project
powershell -ExecutionPolicy Bypass -File scripts\offline_check.ps1 `
  -EnvDir D:\SCC_Retrieval_A\env -BundleRoot D:\SCC_bundle `
  -ExpectedGPUs 2 -RequireA4000
```

成功标志：依赖和 DLL、GPU 独立前反向、离线网络保护、本地 DINO、合成 TIFF/PRJ/TFW/不同 CRS DSM、准备/提取/训练/评估/导出重载全部 PASS，pytest 全通过，退出码 0。报告在 `artifacts\offline_check.json`。Python 自检阻止 socket 联网；目标服务器本来无网络。准备机的验收只证明报告记录的网络隔离范围，不声称已禁用整个操作系统网卡。

## 2. 真实数据接入与小样本校准

**先验证几何，再确认标志。** 新增 `scripts/validate_geometry.ps1 -EnvDir ...` 可在确认标志为 false 时直接读取图像/TXT，检查参数、绘制卫星预测足迹，并用可选同名点 CSV 统计米级误差。无需 DINO 或训练 manifest，不会自动设定标定通过。详见 [几何验证](docs/几何验证.md)。

1. 编辑 `configs/train.yaml` 的数据路径、四组 records、文件命名正则及 sequence 映射；未明确格式时先 inspect。不要复制 bice 示例给全序列。
2. 配置卫星/DSM、各自地理元数据权威来源、强度范围、正常高/DSM 基准转换、局部正常高→椭球高偏移。uint16/float 未配置强度区间会报错。
3. 人工核对 EO/非 EO 安装像素轴、完整姿态链和每帧参考旋转角。四个 `*_confirmed` 必须有证据才能置 true。
4. 在卫星米制 CRS 下提供地理分区 GeoJSON，属性 split=train/val/test；留足完整候选搜索窗和缓冲空间。缺少独立地区时不要强造三分区指标。
5. 根据实际部署设定候选尺度/搜索协议。高度尺度扰动需填写 reference_agl_m。源快照没有完整 backends 字段转换，需按接入说明核对或将部署迁移到共享协议。

```powershell
$py = 'D:\SCC_Retrieval_A\env\python.exe'
& $py inspect_dataset.py --config configs/train.yaml
& $py prepare_dataset.py --config configs/train.yaml --limit 12
& $py diagnostics.py --config configs/train.yaml --manifest artifacts/real/manifest.json
```

检查 mapping_report、preparation_report 和 contact_sheet，确认真实映射、足迹和方向后，再去掉 `--limit` 生成完整 manifest。图块按需窗口读取，不持久化数百 GB 重复图。inspect 输出磁盘上界估算；正式 prepare 会按候选上限做磁盘预检。实际缓存大小约 `(查询数+候选数)×1024×4` 字节加元数据。

## 3. 两进程提特征、单卡训练与评估

```powershell
powershell -ExecutionPolicy Bypass -File scripts\extract_dual_gpu.ps1 -EnvDir D:\SCC_Retrieval_A\env
powershell -ExecutionPolicy Bypass -File scripts\train.ps1 -EnvDir D:\SCC_Retrieval_A\env
powershell -ExecutionPolicy Bypass -File scripts\evaluate.ps1 -EnvDir D:\SCC_Retrieval_A\env -Split val
# 固定配置和 best 后，测试集只作最终评估
powershell -ExecutionPolicy Bypass -File scripts\evaluate.ps1 -EnvDir D:\SCC_Retrieval_A\env -Split test
```

两张卡分别启动一个提取进程，各加载一个冻结 backbone，分片互不重复；失败后同命令重跑自动跳过已验证缓存。`-GPUs 1` 单卡回退。小测自动选择能够运行的 batch，不能把两张 16GB 当作一张 32GB。MLP 默认只在 cuda:0 训练，另一卡可独立做另一个输出目录的实验。参数改变后需新 manifest/缓存，指纹不匹配会拒绝复用。

需要同步双卡训练时，改用已内置的 `scripts/train_mlp_ddp.ps1`；它复用同一 manifest/缓存，保持原全局 batch size 和学习率，输出到独立目录。默认要求两张不同 CUDA 设备，训练时使用标准 DDP/Gloo。

结果同时报告候选覆盖率、所有合法查询 Recall@1/5/10/20、有正样本查询的条件 Recall、首正例名次、组/sequence/尺度统计和耗时。候选中无正样本计未召回。数据解析、几何、读图失败另列全量原因。检索指标不等于最终米级定位精度。

正式导出：`artifacts\real\scc_retrieval_head.pth`。只有两个头和白名单推理元数据，不含 GT 或 backbone 副本。实际集成详见 `docs/接入说明.md`；旧 .so 不会因为复制 pth 自动加载。

## 4. LoRA 备选方案

新增最后 4 层 Q/V、rank=8 的共享 LoRA，131,072 个可训练参数；固定 GeM 输出 1024 维。沿用原 manifest，从图像在线训练，不能复用冻结特征缓存。不增加环境依赖，使用 `scripts/train_lora.ps1` / `scripts/evaluate_lora.ps1`，与 MLP 分开保存结果。

最新完整包还内置 `scripts/train_lora_ddp.ps1`、`scripts/train_lora_aug.ps1` 和 `scripts/train_lora_aug_ddp.ps1`。它们分别提供双卡训练、单卡增强和双卡增强，详细参数及断点说明见开头链接。

训练、断点恢复、公平比较和 SCC 接入详见 [LoRA 备选方案](docs/LoRA备选方案.md)。LoRA 检索使用独立 DINO 实例；原始匹配模型和 SGVA 的 dense/CLS 保持原样，部署增加的显存和计算需实测。

## 5. 联网重建与常见错误

`scripts/online_prepare.ps1` 为准备机创建独立环境，下载官方 Windows wheels、本地源码和测试使用的权重；运行检查后 `scripts/pack_environment.ps1` 生成两种恢复材料及哈希。打包脚本排除 DINO 权重，服务器使用已有文件。包内精确 lock 是本次实际安装结果；requirements.in 是顶层约束，未来重新在线解析的传递依赖可能不同，必须重新测试并出新版本。

- `Formal labels blocked`：真实数据/标定/基准/部署尚未确认；先完成检查，不要绕过标志。
- `sequence missing`：给多帧 TXT 增加显式 sequence，或 `sequence_by_file`，不能只按 number 跨序列匹配。
- `Geo metadata conflict`：对照报告中的两套仿射和 CRS，明确权威来源，不要删除 TFW 掩盖问题。
- `Ray above/near horizon` / `NoData`：检查姿态、DSM、基准与边界；无可信地面高程不可默认海拔 0。
- `Stale cache` / fingerprint mismatch：配置、输入、模型或代码已变，使用新输出目录重新准备/提取。
- CUDA/DLL 错误：先看驱动、卡数、位数及 VC++，只使用包内 Windows CPython3.10 wheels；目标机不临时联网安装。
- conda-pack 缺 `pkg_resources`：本工程锁定 setuptools75.8.0，与 conda-pack0.8.1 配套；不要任意升级。

## 6. 调试与输出更新（2026-11-05）

MLP 与 LoRA 的评估命令自动生成离线 HTML、各方案候选图、逐查询 `queries.csv`、分组 `metrics.csv` 和完整 `candidates.csv`。控制台打印 HTML 路径；评估 JSON 的 `outputs.html` 为相对路径。CSV 使用 UTF-8 BOM，方便 Windows Excel 读取。候选同时保留原始索引、实际排名、相似度、标签、覆盖率、尺度和有效比例；相似度不是概率。

每次评估在 `<评估文件名>_reports/<UTC时间_随机编号>/` 创建独立目录，最后才更新主 JSON。`<评估文件>.attempt.json` 记录最近一次命令的 running/complete/failed 和错误栈；失败时可能保留上次成功 JSON，先检查 attempt 状态，不能把旧结果当成本次成功。图片读取失败会写入报告及 `status.json` 的 `complete_with_visualization_errors`，不静默丢失查询或改变指标分母。完整候选 CSV 可能较大，HTML只展示最多100条失败优先明细。

候选图支持 `gem1024`、`heads`、`mlp512`、`lora_gem1024`，只展示当前评估 split 的查询，每组优先选最差查询；各方案可能选中不同查询，完整比较以逐查询 CSV 为准。独立运行 `diagnostics.py --mode lora_gem1024 ...` 可指定方案。未传评估文件时只显示清单顺序的候选预览，明确标为 NOT ranking，不再把覆盖率排序当成模型排名。绿色多边形是当前几何预测足迹。

训练逐轮输出 `history.json`、`history.csv`、`history.png`（loss与验证R@1，epoch从0开始）。可离线重绘旧结果：

```powershell
& $py outputs.py --history artifacts/real/training/history.json
& $py outputs.py --evaluation artifacts/real/evaluation_val.json --manifest artifacts/real/manifest.json --output artifacts/real/review_val.json
```

旧评估未保存相似度时，重绘的分数字段留空。几何预检增加 `frames.csv` 和 `controls.csv`，逐帧标记控制点超差，损坏的控制点CSV也生成失败报告。独立几何/候选图诊断和新训练要求空输出目录；重复试验请换新目录。LoRA合法断点恢复仍用 `--resume`；代码更新后应新建训练运行，旧代码的恢复指纹会被拒绝。原清单生成代码、配置、数据和冻结特征缓存格式未修改。

两类推理权重先写临时文件并重载核对张量，再原子替换目标文件；保存中断不会覆盖原有完整权重。`.pth.json` 提供SHA256，权重及侧车文件应一起复制并核对哈希。

来源核对：[PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)、[DINOv2 官方源码](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)、[CUDA11.8 官方发布说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html)、[conda-pack 官方文档](https://conda.github.io/conda-pack/)、[Miniconda 官方安装器与哈希](https://repo.anaconda.com/miniconda/)。

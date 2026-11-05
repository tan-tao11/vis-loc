# vis-loc / training

红外—卫星视觉定位的训练工程。本说明中的相对路径和命令以仓库的 `training/` 目录为基准。

当前已收录训练端的 MLP、LoRA、单卡 / 双卡 DDP、LoRA 数据增强、几何诊断、结果可视化和推理接入适配器。**实际服务器上的完整推理源码尚未同步**；`reference/scc_snapshot/` 是历史参考快照，不能代表当前推理实现。

两端 AI 请先阅读仓库根目录的 [AGENTS.md](https://github.com/tan-tao11/vis-loc/blob/main/AGENTS.md) 和 [训练推理对齐清单](docs/训练推理对齐.md)。训练目标是服务实际推理；目前仍有候选生成、几何标定及真实 DINO 权重身份等待推理端核验。

## 仓库结构

训练源码、配置、脚本、测试和文档统一位于仓库的 `training/`，内部导入方式和相对路径保持一致。实际推理源码位于并列的 `inference/`，不放在训练目录内。

| 路径 | 内容 |
|---|---|
| `train*.py`、`configs/` | MLP / LoRA 训练、DDP 和增强配置 |
| `records.py`、`geometry.py`、`preprocess.py`、`prepare_dataset.py` | TXT 解析、几何、图像预处理及候选生成 |
| `models.py`、`lora.py` | 冻结 DINO + 双域 MLP、检索专用 LoRA |
| `evaluate*.py`、`export*.py` | 评估及推理权重导出 |
| `integration/` | MLP / LoRA 加载器、针对历史快照的示例补丁 |
| `tests/`、`scripts/` | 回归测试、Windows 启动与离线交付脚本 |
| `docs/`、`reference/` | 方法说明、诊断文档、接口与历史参考附件 |

仓库只保存源码、示例配置、依赖版本清单和文档。环境、安装器、依赖 wheel、模型权重、真实影像 / DSM、特征缓存和训练输出均不上传。参考目录中的两张小图是几何说明图。

## 使用已有环境

```powershell
git clone https://github.com/tan-tao11/vis-loc.git
cd vis-loc/training
```

复用服务器现有训练环境，通过每个启动脚本的 `-EnvDir` 指定其绝对路径。已验证环境为 Python 3.10.16 / PyTorch 2.5.1+cu118；`requirements.lock.txt` 记录 Windows 离线包的依赖和哈希，**它不是跨平台通用安装清单**。

GitHub 克隆不包含离线安装材料，不能直接把本仓库当作 `offline_install.ps1` 的 `-BundleRoot`。完整安装包的使用方式见 [原部署说明](README_中文.md)。

编辑 `configs/train.yaml`，填写真实数据、卫星图 / DSM、服务器已有 DINO 权重与 SHA256。相对路径现在解析到 `training/` 下，例如 `vendor/dinov2` 对应 `training/vendor/dinov2`，`weights/` 对应 `training/weights/`，`artifacts/` 对应 `training/artifacts/`。已有资源可继续使用绝对路径；旧运行目录不会自动迁移，重用 manifest / 缓存前应检查其中记录的实际路径。

`model.source` 可以指向服务器现有的、版本匹配的 DINO 源码目录。没有源码时，按 [DINO 源码准备说明](docs/DINO源码准备.md) 取得固定版本。权重与源码都不能只按文件名判断一致。

先完成 TXT 字段核验、几何检查和数据准备，再启动训练。MLP 还需要提前提取特征缓存；LoRA 从准备好的 manifest 读取原图。详细顺序见 [原部署说明](README_中文.md)。

| 训练方式 | PowerShell 入口 |
|---|---|
| MLP 单卡 | `scripts/train.ps1` |
| MLP 双卡 DDP | `scripts/train_mlp_ddp.ps1` |
| LoRA 单卡 | `scripts/train_lora.ps1` |
| LoRA 双卡 DDP | `scripts/train_lora_ddp.ps1` |
| LoRA 增强单卡 | `scripts/train_lora_aug.ps1` |
| LoRA 增强双卡 DDP | `scripts/train_lora_aug_ddp.ps1` |

例如，完成数据准备及特征提取后，复用既有环境启动 MLP：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/train.ps1 -EnvDir D:\SCC_Retrieval\env
```

## 两端协作

1. 推理端将实际使用的代码同步到仓库根目录下与 `training/` 并列的 `inference/`，特别是 `backends.py`、`utils.py`、`DINOv2_Shared.py`、`feature_extract.py` 及去除机器私有信息的参数示例，并注明实际启动入口。
2. 对照 [训练推理对齐清单](docs/训练推理对齐.md)，核验同一帧的 TXT 字段、几何、图块、归一化张量、描述子及候选排序。记录实际结果，不能把待确认项直接改成已确认。
3. 任一端改动预处理、特征或候选协议时，同步更新对齐文档、配置与必要的回归验证。用 Git commit 标识代码，用权重 SHA256 标识模型；通过既有文件渠道交换大文件。

本仓库保留原文件字节和换行符，避免 Windows / Linux 自动换行转换改变源码指纹。运行产生的 manifest / 特征缓存不入库；源码指纹或预处理发生变化时需要重新核验缓存有效性。

## 验证

在 `training/` 目录、已有环境中运行：

```powershell
. ./scripts/common.ps1
Set-EnvironmentPath 'D:\SCC_Retrieval\env'
& 'D:\SCC_Retrieval\env\python.exe' -m pytest -q
```

已有验证包含 81 项回归测试以及实际 DINO 的合成流程检查；历史范围和限制见 [验证状态](docs/验证状态.md)。这些结果不代表真实定位精度，也不代表已完成目标双卡服务器或当前推理系统的一致性验收。

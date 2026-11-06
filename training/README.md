# vis-loc / training（倾斜专用）

红外—卫星视觉定位中 SCC-Loc 检索模块的微调工程，**只服务倾斜视角（DS/NS）**。垂直视角使用另一套权重、配置和 .so，不在本分支。

运行步骤、修改内容和检查点见 **[倾斜训练与部署指南](docs/倾斜训练与部署指南.md)**；两端一致性见 [训练推理对齐](docs/训练推理对齐.md) 末节。

## 流程

`inspect_dataset.py` → `prepare_dataset.py`（按推理原代码生成候选块并用真值足迹打标签）→ `extract_features.py`（冻结 DINOv2 ViT-L GeM-1024，双卡分片缓存）→ `train.py`（双域 MLP 头，512 维）→ `export.py`（`scc_retrieval_head.pth`）→ `evaluate.py`（`gem1024` 与 `heads` 的 R@k）。

一键脚本：`scripts/run_mlp.ps1 -EnvDir <环境> -Config configs/slant.yaml`。

## 倾斜专用改动

| 内容 | 文件 |
|---|---|
| 搜索窗口以视场中心（飞机前方 AGL/tan(俯角)）为中心 | `scc_inference.frame_retrieval` |
| 查询图透视矫正为俯视图，候选块按查询可见区域遮挡 | `scc_oblique.py`（推理 `oblique.py` 的逐字节拷贝）、`scc_inference.TileSource` |
| 块大小按短边 × `query_crop_scale` 的中心区域估算 | `preprocess.py`、`scc_inference.py` |
| 倾斜配置 | `configs/slant.yaml` |
| 检查工具 | `slant_report.py`（逐帧 AGL/俯角/块/窗口/足迹距离/正样本）、`eval_one.py`（单帧检索排名） |

已删除 LoRA、外观增强和 DDP 训练（MLP 单卡训练只需几分钟），以及垂直配置。

## 目录

| 路径 | 内容 |
|---|---|
| `configs/train.yaml`、`configs/slant.yaml` | 完整默认值；倾斜精简配置（`base: train.yaml`） |
| `records.py`、`geometry.py`、`preprocess.py`、`prepare_dataset.py` | 记录解析、几何投影、查询预处理、候选生成与标签 |
| `scc_inference.py`、`scc_geodata.py`、`scc_oblique.py` | 推理代码的逐字拷贝 / 逐字节拷贝（测试会比对） |
| `models.py`、`losses.py`、`train.py`、`evaluate.py`、`export.py` | 模型、损失、训练、评估、导出 |
| `scripts/` | Windows 启动、离线安装与检查脚本 |
| `tests/` | 回归测试（`python -m pytest -q tests`） |
| `docs/`、`reference/` | 说明文档、接口与参考附件 |

仓库不含环境、权重、真实影像 / DSM、特征缓存和训练输出。

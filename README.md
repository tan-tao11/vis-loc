# vis-loc

红外—卫星视觉定位：训练端与推理端的源码协作仓库。

仓库按训练与推理分为两个独立目录。训练端已包含 MLP、LoRA、单卡 / 双卡 DDP、LoRA 数据增强、几何诊断、可视化和接入适配器；实际服务器上的完整推理源码尚待同步。

两端 AI 请先阅读 [AGENTS.md](AGENTS.md) 和 [训练推理对齐清单](training/docs/训练推理对齐.md)。

## 仓库结构

| 目录 | 内容与入口 |
|---|---|
| [training/](training/README.md) | 完整训练工程：源码、配置、测试、脚本、文档、参考附件与推理适配器 |
| [inference/](inference/README.md) | 实际推理工程：预留当前部署源码的接收位置，现阶段只有接入说明 |

根目录只保留仓库说明、AI 协作约定和 Git 配置文件。`training/reference/scc_snapshot/` 是历史参考快照，不能代表当前生产推理实现。

环境、安装器、依赖 wheel、模型权重、真实影像 / DSM、特征缓存和训练输出不入 Git。依赖版本清单保存在训练目录。

## 训练入口

```powershell
git clone https://github.com/tan-tao11/vis-loc.git
cd vis-loc/training
```

后续训练命令在 `training/` 目录执行，配置中的相对路径也以该目录为基准。复用已有环境，通过启动脚本的 `-EnvDir` 指定环境绝对路径。

完整步骤、六个训练入口和测试命令见 [训练说明](training/README.md)。例如，在完成数据准备和 MLP 特征提取后：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/train.ps1 -EnvDir D:\SCC_Retrieval\env
```

旧部署的环境、权重和运行目录不会随 Git 目录调整自动迁移。使用绝对路径指向既有资源，或按训练说明准备新的相对路径；不要覆盖旧运行目录。

## 推理接入与对齐

推理端将实际源码及去除机器私有信息的配置示例同步到 `inference/`，保持其内部结构，并注明启动入口和工作目录。接收清单见 [推理端说明](inference/README.md)。

两端按 [对齐清单](training/docs/训练推理对齐.md) 比较同一输入的几何、图块、归一化张量、描述子、排序与坐标回映射。未经核验的配置标记保持未确认。

既有 81 项测试和合成检查的验证范围见 [验证状态](training/docs/验证状态.md)；它们不代表真实定位精度或当前生产系统的两端一致性验收。

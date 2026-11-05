# MLP 双卡 DDP 备选

MLP 和 LoRA 现在都有独立双卡入口。已有单卡 MLP、单卡 LoRA 和 LoRA 双卡文件全部保留。

## 安装和启动

**最新整合版完整包已内置本功能，新安装直接使用下方训练命令，无需安装增补包。** 以下步骤仅供旧部署升级：将 `mlp_ddp_addon.zip` 解压到单独目录，运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\mlp_ddp_addon\install_mlp_ddp_addon.ps1 -ProjectDir D:\SCC_Retrieval_A\project
```

进入工程目录后，一条命令启动双卡 MLP，并自动导出最佳权重：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_mlp_ddp.ps1 -EnvDir D:\SCC_Retrieval_A\env
```

默认复用 `artifacts/real/manifest.json` 和 `artifacts/real/features`，输出到 `artifacts/real/mlp_ddp`；如果准备数据时使用了其他目录，加 `-DataRun` 指定即可。可用 `-Run` 指定新的输出目录，或 `-Devices cuda:0,cuda:1` 指定两张可见 GPU。

不需要再填写一份 YAML。模型结构、训练轮数、学习率、困难负例和 batch size 都读取已核验 manifest 中保存的原配置。`train.batch_size` 是两张卡合计的全局批量，默认 8 即通常每卡 4 组；原配置是多少就保留多少，学习率不自动放大。要修改训练参数，应按原数据准备/缓存流程生成新的匹配运行，不能直接改 manifest 绕过指纹校验。

本包可独立安装，不要求先装 LoRA 双卡增补包；共享 DDP 文件内容一致时自动跳过。安装器先检查全部哈希和冲突，然后只添加缺少的文件，同名文件不同则拒绝写入。没有增加 Python 依赖，也不包含 DINO 权重；服务器继续使用现有权重。原完整部署 ZIP 无需重新下载。

## 与原训练的关系

| 方案 | 启动脚本 | 默认训练输出 |
|---|---|---|
| MLP 单卡 | `scripts/train.ps1` | `artifacts/real/training` |
| MLP 双卡 | `scripts/train_mlp_ddp.ps1` | `artifacts/real/mlp_ddp` |
| LoRA 单卡 | `scripts/train_lora.ps1` | `artifacts/real/lora` |
| LoRA 双卡 | `scripts/train_lora_ddp.ps1`（LoRA 双卡增补包） | `artifacts/real/lora_ddp` |

MLP 双卡训练复用冻结 DINO 的特征缓存，每张卡都有红外、卫星两个 MLP 头；查询组分给不同进程，使用标准 `DistributedDataParallel` / Gloo 同步梯度。训练阶段不加载 DINO 模型，但缓存校验仍核对本地 DINO 文件哈希。两张卡不是各自训练一个不同方案。

原分组平衡、有放回采样及训练集困难负例规则保留。困难负例挖掘按查询分到两卡，再合并；不足整批时用零损失占位，按真实组数求均值，不额外重复样本。验证由 rank 0 对完整验证集和全部候选执行，保留原 GeM1024/MLP 对照指标，只按验证 R@1 选最佳模型。测试集不参与选模。损失曲线记录每轮真实查询组的平均损失。

Windows 入口内部用 spawn 启动两个进程，不必额外套 `torchrun`。每次启动先检查所选设备的 Gloo CUDA all-reduce。默认必须是两张不同 GPU；只有一张卡会报错，不会隐式伪装成双卡。无需 nvcc 或额外编译 CUDA 扩展，沿用当前 PyTorch 2.5.1+cu118 环境。

MLP 的计算量比在线 DINO/LoRA 小，通信和进程开销可能抵消加速收益。建议依据同一数据、批量下的实际总耗时决定使用哪条入口。本实现每个 worker 各加载一份 CPU 特征缓存，缓存内存约为单进程的两份。

## 断点、输出与评估

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_mlp_ddp.ps1 -EnvDir D:\SCC_Retrieval_A\env -Resume artifacts\real\mlp_ddp\last.pth
```

新训练要求空输出目录。恢复时使用原目录、保留 best，并使用相同数据、配置、代码、设备和 PyTorch 版本；单卡 checkpoint 不能充当这个 DDP 入口的恢复断点。保存每个进程的随机数状态、优化器、调度器、困难负例、最佳分数及早停状态。

- `best.pth` / `last.pth`：最佳验证权重和最近完整 epoch 的断点。
- `scc_retrieval_head.pth`：PowerShell 入口自动调用原 `export.py` 生成，沿用原检索适配器接口。
- `history.json/csv/png`：损失、验证 R@1、两路梯度、每轮副本权重哈希。
- `best_validation_review.json` 与对应报告目录：GeM/MLP 排名、HTML、CSV、候选图，沿用原输出功能。
- `rank_0.log` / `rank_1.log`、`rank_*_status.json`：进程日志及当前阶段。
- `performance_epoch_*.json`：每个进程训练用时、显存峰值、查询数和总体吞吐；不含挖掘和验证，比较总耗时应另外计时。
- `job_status.json`：最近一次启动的权威状态；`rank_*_failure.json` 保存异常，恢复后可能仍保留历史失败。

只有 rank 0 写汇总权重和报告。任一进程失败，父进程结束其他 worker，整次作业标记失败；不会自动导出此次失败的训练。导出的头可以直接用于原 SCC 接入。手动评估可调用原 `evaluate.py --manifest ... --cache ... --checkpoint ... --split val --output ...`，指定新的 DDP checkpoint 路径即可。

## 已验证范围

测试覆盖两个 MLP 头的 DDP 梯度相对串行全局均值（含只有一个真实组的尾批）、有 dropout 的连续训练与断点恢复逐张量一致、两路梯度及副本一致性、现有导出/检索适配器兼容、单 rank 异常联动。

准备机仅一张 RTX4060。已使用实际 DINO 提取的合成数据缓存，在同一张 GPU 上启动两个进程，完成训练、全量验证和报告输出；这明确属于 `diagnostic_only=true` 的通信与管线测试，不是两张 A4000 的性能验收，也不代表真实定位精度。目标服务器使用默认两张不同 GPU 运行后，才能评估实际加速效果。

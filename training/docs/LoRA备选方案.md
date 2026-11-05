# DINOv2 LoRA 备选方案

原有冻结 DINO + 双 MLP 方案继续保留。新增方案在检索专用 DINOv2 中训练 LoRA，共用既有图像预处理、地理划分、manifest、候选标签和召回统计。两种方案是独立实验，默认不叠加 MLP 与 LoRA。

| 项目 | 原有 MLP | 新增 LoRA |
|---|---|---|
| 可训练部分 | IR/卫星各一套 MLP | DINO 最后 4 个 block 的 Q、V 低秩增量 |
| DINO 原始权重 | 冻结 | 冻结；输出由 LoRA 改变 |
| 两域参数 | 独立投影头 | 共享同一个适配后的检索 DINO |
| 训练输入 | 已缓存的固定 1024 维特征 | 按需读图、实时前向与反向 |
| 检索输出 | 512 维、L2 归一化 | 固定 GeM(p=4)，1024 维、L2 归一化 |
| 部署开销 | 共享原始 DINO，增加小型投影头 | 增加独立检索 DINO 实例及前向计算 |

## 默认配置与内存

`configs/lora.yaml`：24 层 ViT-L/14 的最后 4 层，即从 0 起编号的 20–23 层；仅 Q、V，rank=8、alpha=16、dropout=0.05。计算 `W(x) + (alpha/r) B A x`，A 随机初始化、B 初始化为零，因此初始检索输出与原始 DINO 一致。可训练参数 **131,072**；K、MLP、LayerNorm、位置编码及全部原始权重均冻结。IR 与卫星使用共享适配器。

单卡默认 AMP、非重入梯度检查点、图像 microbatch=1、每 4 个查询组累积一次梯度。前 20 层不保存反向图；最后 4 层仍通过冻结线性层向 LoRA 参数反传。每组最多采样 4 个正例、8 个困难负例、4 个随机负例。一个组内的图像图会保留到该组 loss 反传，所以 microbatch=1 不等于整组只占一张图的显存。显存不足时降低每组正负例数量、microbatch 或适配层数，保持梯度检查点；不能将两张 16GB 卡当成一张 32GB 卡。

原生 PyTorch 实现，无新增依赖，沿用 **Python3.10 / torch2.5.1+cu118** 离线环境；不需要 nvcc、PEFT、Triton、xFormers 或编译扩展。用户提供的目标驱动 527.27 高于 CUDA11.8 的保守配套 Windows 驱动基线 522.06；`nvidia-smi` 的 CUDA12.0 表示驱动能力，不要求将本环境换成 cu120。目标服务器实际训练、显存和吞吐仍需在该机运行自检确认。

## 训练和比较

首先按主 README 完成真实数据映射、几何标定、部署一致性检查和 `prepare_dataset.py`，生成原有 `artifacts/real/manifest.json`。已有可用 manifest 可直接使用；LoRA 不需要先运行 `extract_features.py`，也不读取 MLP 特征缓存。源数据文件仍须在 manifest 记录的位置。

```powershell
cd D:\SCC_Retrieval_A\project
# 可选：先验证目标机器；默认输出目录必须没有已有训练运行
powershell -ExecutionPolicy Bypass -File scripts\lora_smoke.ps1 `
  -EnvDir D:\SCC_Retrieval_A\env -Device cuda:0 -Output artifacts/lora_smoke_server

# 正式训练，并自动导出 adapter-only 权重
powershell -ExecutionPolicy Bypass -File scripts\train_lora.ps1 `
  -EnvDir D:\SCC_Retrieval_A\env -Device cuda:0

# 在同一个 manifest 的验证集比较 LoRA / 原始 GeM
powershell -ExecutionPolicy Bypass -File scripts\evaluate_lora.ps1 `
  -EnvDir D:\SCC_Retrieval_A\env -Split val

# 同时比较 MLP：必须指定同一个 manifest 训练得到的 best.pth
powershell -ExecutionPolicy Bypass -File scripts\evaluate_lora.ps1 `
  -EnvDir D:\SCC_Retrieval_A\env -Split val `
  -MlpCheckpoint artifacts/real/training/best.pth

# 确定配置后，才进行最终测试
powershell -ExecutionPolicy Bypass -File scripts\evaluate_lora.ps1 `
  -EnvDir D:\SCC_Retrieval_A\env -Split test
```

MLP 的 `best.pth` 路径以实际 `scripts/train.ps1 -Run` 输出为准，不能使用不带训练 manifest 身份的部署投影头冒充同集比较。默认 LoRA 输出为 `artifacts/real/lora`，包含 `run.json`、`history.json`、`best_validation.json`、`best.pth`、`last.pth`、困难负例记录和 `scc_retrieval_lora.pth`。后者是部署文件，不包含原始 DINO 权重、GT、优化器或数据路径。

中断后继续同一个运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_lora.ps1 `
  -EnvDir D:\SCC_Retrieval_A\env `
  -Resume artifacts/real/lora/last.pth
```

恢复要求 manifest、配置、源代码及基座身份完全一致，使用原输出目录并保留 best.pth；恢复优化器、调度器、AMP scaler、困难负例和随机数状态。更换参数应另起 `-Run`，不要修改旧运行后强行续训。默认单卡，第二张卡可以独立运行另一套配置和输出目录。

## 实验口径

训练使用组内多正例对比损失。只从 train 采样和挖掘困难负例，ignore 不参与训练；没有正例或没有合法负例的组跳过训练并记录。正例超过上限时随机抽取，未选正例不会转成负例。DS/DV/NS/NV 按有资格训练的组均衡采样，每轮保留全部原始训练帧，再过采样较小分组。困难负例定期用当前 LoRA 重算，包含地理近邻和外观相似的合法负例。

每轮验证全体固定候选，所有合法查询都进入总体召回，候选无正例计为未召回；选优只看 val，先 R@1，再 R@5 和首正例平均名次打破平局。test 不参与训练、挖掘或选优。最终比较使用同一 manifest、同一预处理、同一基座和全候选集。三种方法的特征维度不同，实验包含这一容量差异；MLP 和 LoRA 训练采样方式也不同，不能称为只改变一种变量的消融实验。

在线评估单列读图、预处理及 DINO 前向耗时，不含模型加载；排序耗时单列。它不是 SCC 整条匹配定位管线的端到端时延，桌面显卡测得的耗时也不代表 Orin 性能。若改用 LoRA 预计算部署图库，必须使用对应 LoRA 版本重新生成特征，禁止混用原始 DINO 或 MLP 图库。

## SCC 接入

`integration/lora_adapter.py` 每次启用时从本地源码/权重创建**独立的检索 DINO 实例**，只替换用于检索相似度的全局特征。原共享 DINO 的 dense、CLS 和 MINIMA-RoMa 匹配模型继续使用原始参数；不向它们注入或合并 LoRA。保留 SGVA 的原始 dense/CLS 意味着部署时需要两路 DINO 前向，必须实测显存和时延。

```python
from common import config
from integration.lora_adapter import LoRAInferenceAdapter

adapter = LoRAInferenceAdapter(
    'artifacts/real/lora/scc_retrieval_lora.pth',
    data_config=config('configs/train.yaml'),
    enabled=True, device='cuda:0')

# images 为 SCC 本次已归一化的 N×3×336×336 图像张量
original = original_scc_model(images, return_dense=True)
global_feat, dense, cls = adapter.replace_global(original, images)
```

入口校验原始 DINO 权重/源码哈希和预处理/姿态配置；默认拒绝合成权重或不匹配文件。禁用时直接返回原始全局特征，dense/CLS 保持同一对象。只有显式 `allow_fallback=True` 才允许加载失败后警告并回退到原始 GeM1024。

`integration/feature_extract.lora.snapshot.patch` 展示交接源码快照中查询、候选两处接入。实际 `utils.py` 调用方还须传入 `retrieval_lora=adapter`。这个 patch 与原 MLP patch 是备选接法，不应直接叠加使用；不能直接套用到未知修改版或编译后的 .so。须先核对比赛机实际源码、共享权重、输入归一化、尺寸和坐标协议，再在目标平台重建并验证。

## 验证与资料

本机 RTX4060 已使用真实 DINOv2 权重完成 12 查询/300 候选、3 轮 LoRA 合成训练及全候选评估；零初始化对照和导出回读的最大绝对误差均为 0，A/B 均有非零训练梯度。完整测试 **36 项通过**，包含中断恢复结果一致性。增量包已在 Windows PowerShell5.1 和原方式A离线恢复环境安装并再次通过36项测试，数据配置哈希保持不变。同一 manifest 的 GeM、MLP、LoRA 三方案比较也已实际运行。

详细结果见完整交付包的 `docs/验证状态.md` 和 `validation/lora/`。合成流程用于确认训练、梯度、导出回读和接口行为，**不证明真实红外检索精度有提升**。目前没有真实 LoRA 业务权重；目标服务器和 Orin 尚未实际验收。

方法依据：[LoRA 原论文](https://arxiv.org/abs/2106.09685)。本工程选择最后 4 层 Q/V 是资源受限下的初始配置，不是论文对本数据的最优结论。[固定 DINOv2 源码](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)、[PyTorch 非重入 checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)、[CUDA11.8 Windows 驱动说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html)。运行行为以本包 torch2.5.1 的实际测试为准。

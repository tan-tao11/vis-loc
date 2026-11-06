# 推理源码：原部署包＋对话确认的缓存修改

2026-11-05：按用户明确授权，以原部署源码和最近提供的手动修改重建本目录。**这是用于训练／推理对照的合并源码，不是从比赛机器实时回收并逐字节验证的缓存。** 开发板 USB SSH 不可达，本次未执行合并版本的 GPU／真实数据验收。

已合并参考图旋转、可选近下视对齐分支、简化正方形候选窗口、`reference_window_scale`、关闭 SCC 排序后 SGVA 位移／放大、搜索中心日志，以及可选 matching 调试输出。没有混入此前未采用的查询图旋转实验、矩形足迹／自动地图扩展方案，也没有新接入 MLP 或 LoRA。

## 倾斜专用版本（2026-11-07，slant 分支）

本源码用于**倾斜（DS/NS）专用 .so**；垂直数据使用另一个 .so。`infrared.camera.oblique: true` 时：

1. `BaseBackend.prepare` 先在飞机先验处读 200 m 的 DSM 得到 AGL，再把搜索地图中心前移到光轴地面点（AGL/tan(俯角)），返回给 SCC 的先验仍是飞机位置；
2. `SCCBackend.localize` 用 `single-so/python/oblique.py` 把查询图矫正成“一个块大小、航向朝上”的俯视图后再送检索（`utils.retrieval_all` 的 `query_prepared`）；
3. `feature_extract` 用同一个可见区域 `tile_mask` 遮挡每个候选块；
4. 块大小按短边 × `camera.query_crop_scale` 的中心区域估算。

训练端 `training/scc_oblique.py` 与 `oblique.py` 逐字节相同。配置见 `naviParams.example.txt`，步骤见 [倾斜训练与部署指南](../training/docs/倾斜训练与部署指南.md)。另含 `prior_tracker` 的 `offset_model`（固定 / 缓慢变化的先验偏移）、失败时打印堆栈和 `[RETRIEVAL_TOP]` 日志。

## 目录与入口

| 路径 | 用途 |
|---|---|
| `single-so/src/`、`include/` | 官方接口、ELF适配、运行时解包和内部worker通信 |
| `single-so/python/bootstrap.py`、`worker.py` | 内嵌Python启动及逐帧输入 |
| `single-so/python/backends.py` | 字段、K／AGL／姿态、参考旋转、中心日志、算法调用 |
| `single-so/python/geodata.py`、`tuning.py` | GeoTIFF／TFW／PRJ、地图读取、调参及可选matching诊断 |
| `projects/SCC-Loc/` | 红外共享DINOv2检索、MINIMA-RoMa及项目内源依赖 |
| `projects/AnyVisLoc-Benchmark/` | RGB CAMP／RoMa及项目内源依赖 |
| `naviParams.example.txt` | 去除机器私有路径的配置示例，不是比赛机实测配置 |
| `single-so/WEIGHTS.json` | 原部署权重的名称、大小和SHA256，不包含权重 |
| `runtime-requirements.snapshot.txt` | 原ARM64运行时包元数据，不是Windows安装锁文件 |
| `SOURCE_MANIFEST.json`、`tools/compare_cache.py` | 源码哈希、运行时映射、实际缓存只读比对 |
| `docs/` | 合并记录、边界及原手动修改片段 |

第三方源码保持原相对目录和已有许可证。只同步源代码、依赖声明与必要配置，不包含环境、权重、地图、真实图像、编译产物和输出。上游部分训练／演示入口引用的资产不提供，生产入口以本适配器为准。

## 当前IR流程

1. 官方接口传给worker，`*_gt` 不进入推理进程。
2. `prepare()` 先按飞机位置先验读取局部地图，再计算K、DSM高度差和有效光轴姿态。
3. `pose_priori: yp` 时先旋转参考图。查询保持原中心裁剪／缩放路径。
4. 打印三个中心／姿态日志，不更改中心。
5. `estimate_reference_side()` 根据实际查询ROI、K、向下角和局部平地近似，估算方形边长并乘 `reference_window_scale`。保持原 `finescale`；地图不足时报错。
6. 在已读取地图中按原网格生成候选；DINOv2 ViT-L/14 → GeM(p=4) → 1024维描述子 → 点积排序。当前没有MLP／LoRA加载。
7. 排序后 `shifts=None`，不应用SGVA的候选移动／放大；dense／CLS仍计算。
8. MINIMA-RoMa匹配、逆旋转、DSM和PnP输出位置。

近下视 `nadir_reference_alignment` 是按对话重建的可选近似，示例默认false。EO仅接受 `innsPitch≈-90°`，返回 `innsYaw+innsRoll`；非EO仅接受sensor角及INS pitch／roll接近零，返回 `innsYaw`。1°门限不是标定精度。斜视EO须关闭该分支。它不等于训练侧完整的安装矩阵／三轴射线投影。

## 大先验误差：定位跟踪与扫描捕获（2026-11-05）

比赛先验误差可达 2–3 海里、高度 ±200 m，误差可能逐帧随机，也可能连续漂移；单帧预算 10 秒。`infrared.prior_tracking.enabled: true` 时由 `python/prior_tracker.py` 决定每帧的搜索中心和窗口：
- **跟踪（主方案）**：定位结果经下一帧确认后，中心 = 上次定位 + 速度（最近两次定位之差，能跟随转弯），小窗口；不依赖先验误差的形式。
- **捕获**：在平滑先验周围逐帧扫描 `max_window_m` 窗口覆盖 `prior_error_m` 范围。`acquisition: smooth` 只看平滑中心（误差逐帧独立时更快，连续漂移时会失效），`hybrid` 二者交替。
- **输出**（误差 >150 m 记 0 分）：跟踪中定位失败或不可信时输出预测位置（置信度 0），见 `BaseBackend.localize_tracked` 和日志 `[PRIOR_OUT]`。
- `max_window_m` 按 10 秒/帧尽量取大，看 `[PRIOR_TRACK]` 与 `[NAV_TIMING]`。`enabled: false` 恢复原固定窗口行为。

详见 [训练推理对齐](../training/docs/训练推理对齐.md) 末节。改动后需按下文 `repack_runtime.py` 重新打包。

## MLP 检索头（2026-11-06）

`infrared.retrieval_head` 指向训练导出的 `scc_retrieval_head.pth` 时，`feature_extract.extract_features` 对查询/卫星的 GeM 描述子分别应用 `ir`/`satellite` 头（`single-so/python/retrieval_head.py`），日志 `[RETRIEVAL_HEAD] loaded ...`。不填则为原 GeM-1024 检索。

## 使用与构建

**从旧 .so 重新编译（推荐）**：`cd single-so && bash rebuild_from_old_so.sh /path/to/old/libvNavigatorInterface.so`（取回内嵌运行时 → `repack_runtime.py` → `build.sh` → `runtime_check`）。


本目录不是可在Windows直接运行的推理环境。实际入口仍为Orin／Linux官方框架加载 `.so`。运行时映射为：`single-so/python/` → `<runtime>/adapter/`，`projects/` → `<runtime>/projects/`；ARM64 Python、env、extra、native依赖另行准备。

先对照实际缓存，工具只读、不覆盖：

```bash
python3 inference/tools/compare_cache.py --cache /actual/runtime/cache
```

重建单文件库需在兼容的ARM64 Ubuntu中准备原运行时，进入 `inference/single-so`。如重新捕获环境，使用 `tools/capture_runtime.py --source-home /prepared/source/home`，依赖wheel另行提供。**构建前运行 `python3 tools/repack_runtime.py` 同步adapter和两个项目树**，然后 `bash build.sh`。只更新adapter会遗漏本次 `utils.py` 修改。未准备运行时或未做目标机测试时，不应声称已生成验证过的新 `.so`。

配置路径相对配置文件；bootstrap会覆盖project_root为内嵌目录。官方框架通常从其自身目录启动，后端切换到对应模型项目目录。不要从仓库根目录随意执行上游 `run_avl.py` 代替官方入口。

## 验证与边界

见 [合并记录](docs/SNAPSHOT.md)、[训练推理对齐清单](../training/docs/训练推理对齐.md) 和 `VALIDATION.json`，目标设备确认仍为false。

当前C++将**初始化时**的 `isEO` 传给逐帧metadata；`isSlant` 为逐帧值。此次没有顺便改变这一行为。`debug_matching: true` 可保存匹配输入／点坐标／图像，默认关闭，开启影响耗时和磁盘。

本地CPU几何／源级回归不代表真实定位精度或比赛GPU速度。下一步需固定同帧输入，比较候选框、张量、特征、排序及坐标回映射，再决定是否改变搜索中心或接入训练后模型。

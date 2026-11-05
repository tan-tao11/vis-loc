# DINOv2 源码与权重身份

本文命令在仓库的 `training/` 目录执行；下文的相对路径均以该目录为基准。`training/vendor/` 和 `training/weights/` 不入 Git。优先复用服务器已有资源，通过 `configs/train.yaml` 中 `model.source`、`model.checkpoint` 指向实际位置。

训练默认使用非 register 的 DINOv2 ViT-L/14，固定源码版本为：

```text
repository: https://github.com/facebookresearch/dinov2
commit: 7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

已有源码必须检查实际版本，不能仅新建标记文件就认定匹配。需要联网取得源码时，在**尚不存在 `vendor/dinov2`** 的新克隆仓库中执行以下 PowerShell 命令；只取得源码，不下载权重或环境：

```powershell
$DinoCommit = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
if (Test-Path -LiteralPath 'vendor/dinov2') { throw 'vendor/dinov2 already exists; inspect it before proceeding.' }
git clone --no-checkout https://github.com/facebookresearch/dinov2.git vendor/dinov2
if ($LASTEXITCODE -ne 0) { throw 'DINO clone failed' }
git -C vendor/dinov2 checkout --detach $DinoCommit
if ($LASTEXITCODE -ne 0) { throw 'DINO checkout failed' }
$ActualDinoCommit = (git -C vendor/dinov2 rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualDinoCommit -ne $DinoCommit) { throw 'DINO source commit mismatch' }
Set-Content -LiteralPath vendor/dinov2/SOURCE_COMMIT.txt -Value $DinoCommit -Encoding ASCII
```

保留上游源码中的许可证。两端同时核验以下字段：

- `source_commit`：上述源码 commit 标记。
- `source_tree_sha256`：`models.backbone_identity` 对全部 Python 源文件字节计算的指纹；相同 commit 但存在本地修改或不同换行符时也可能不同。
- `checkpoint_sha256`：实际 DINO 骨干权重的 SHA256。
- 模型结构：非 register ViT-L/14，24 blocks，1024 维。

原部署包中的官方参考权重哈希为 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`，仅作身份比对。服务器已有的 SCC 权重是否与它一致仍需实测；不要因此替换服务器模型，也不要把整套 `minima_roma.pth` 当作单独 DINO state_dict。

MLP 加载器当前核验调用方提供的 backbone 权重哈希和预处理 / 姿态指纹；它不会自行重算实际推理 DINO 的源码身份。LoRA 加载器会构建独立骨干并比较完整 `backbone` 身份字典。部署核验必须涵盖源码和实际计算路径，不能只以适配器成功加载作为一致性证明。

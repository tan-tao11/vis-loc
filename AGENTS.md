# Training / inference collaboration

This repository coordinates the SCC visual-localization training and inference code. Read `README.md`, `training/docs/训练推理对齐.md` and the relevant adapters before changing model behavior.

- The root has two project directories: `training/` for the complete training project, and `inference/` for actual deployment source. Run training Python commands and tests from `training/`; its config paths resolve against that directory. `training/reference/scc_snapshot/` is historical reference, not authoritative production code.
- Keep the MLP baseline, LoRA alternative, single-GPU, DDP and augmentation entry points available. Use separate output directories; do not overwrite an existing run to compare methods.
- Treat preprocessing, coordinate conventions, candidate ordering, backbone identity, feature normalization and exported metadata as a joint training/inference contract. Update the alignment document when these change, and validate the affected path at both ends when the deployment source is available.
- Keep unverified configuration flags false. Missing deployment code, calibration, real data or target-device evidence must stay explicitly unverified.
- Preserve the original DINO dense/CLS tensors used by SGVA and matching. MLP only replaces the retrieval global descriptor. LoRA uses a separate retrieval backbone; do not merge its adapters into MINIMA/RoMa's shared matching backbone.
- Appearance augmentation is training-only. Mining, validation, test and inference use clean images. Never insert validation/test ground truth into the candidate search.
- Source-byte fingerprints participate in manifest/cache validity. Preserve line endings as configured in `.gitattributes`; changes to `training/records.py`, `training/geometry.py`, `training/preprocess.py` or `training/prepare_dataset.py` require checking manifest/cache compatibility. A directory-only move preserves source hashes, but existing manifests can still contain absolute resource paths; verify those separately.
- Keep environments, weights, data, credentials, caches and output artifacts out of Git. Dependency/version lists and sanitized configuration examples are source documentation; actual local paths belong in ignored local configs when possible.
- Run the relevant existing tests for behavioral changes. Report which machine/data were tested and which alignment checks remain pending. Do not equate synthetic tests with real-data accuracy or target-device validation.
- Use ordinary commits/branches and fetch remote changes before publishing; preserve the other side's work. A user's explicit instructions take precedence over these collaboration conventions.

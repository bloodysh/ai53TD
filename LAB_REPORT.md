# AI53 MLOps Project Phase 1 - Lab Report

## Milestone 0 - Starter audit

| # | File / setting | Problem | Suggested fix |
| :- | :--- | :--- | :--- |
| 1 | `src/reproducibility.py` | Only Python `random` is seeded. NumPy, PyTorch (CPU/CUDA), cuDNN, the DataLoader shuffle generator and worker seeds are left uncontrolled, so weight initialisation and batch order change between runs. | Seed every generator in `set_reproducibility()`, return a seeded `torch.Generator` from `make_generator()` and seed NumPy/`random` in `seed_worker()`. Check with `tests/test_reproducibility.py`. |
| 2 | `src/validate_data.py` | `validate_dataset()` raises `NotImplementedError`: there is no quality gate, so a missing split, an extra class, a wrong count or a corrupt JPEG would only be discovered (or silently ignored) during training. | Check splits, classes, exact counts, full pixel decoding and empty classes; raise on failure and write a deterministic `reports/data_quality.json`. |
| 3 | `data/raw/` + `.gitignore` | The images are ignored by Git and not tracked by anything else, so the exact data version used by a run cannot be recovered. | Track `data/raw` with DVC, commit `data/raw.dvc`, and push the images to a DVC remote. |
| 4 | `dvc.yaml` (missing) | There is no pipeline: the order VALIDATE → PREPARE → TRAIN/EVALUATE, and the files/params each step uses, exist only in people's heads. `prepare.py` even says it "must depend on the successful data-quality report" but nothing enforces it. | Write a four-stage `dvc.yaml` with explicit `deps`, `params`, `outs` and `metrics`; commit `dvc.lock`. |
| 5 | `src/provenance.py` | `build_run_manifest()` is not implemented, so a run records no Git commit, dirty flag, package versions, platform or data identity. | Collect Git SHA + `git status --porcelain`, `training_data_identity()`, library versions via `importlib.metadata`, platform/CUDA and the run settings. |
| 6 | `src/train.py` | An MLflow Run is opened but nothing is logged (no params, metrics, dataset, artifacts or model). The only result is `models/model.pt`, which is overwritten by every run, so there is no run history to compare. | Log params, per-epoch metrics, the dataset input, report artifacts and the `bean-cnn` model with signature and metadata. |
| 7 | `params.yaml` → `selection.run_id: ""` | No model has been selected and nothing prevents someone from evaluating several candidates on the test set and picking the best one (test-set leakage). | Choose a Run using validation metrics only, freeze its ID in `selection.run_id`, and run EVALUATE once. |
| 8 | `params.yaml` → `compute.device: "auto"` | The same config can run on CPU on one machine and CUDA on another, which changes numerics and timings. | Record the resolved device in the manifest/MLflow params (and pin `cpu` for this lab if comparing runs across machines). |
| 9 | `mlflow.db`, `mlartifacts/` in `.gitignore` | The Run history lives only on the local machine; it is neither in Git nor in DVC. | Keep and back up `mlflow.db` + `mlartifacts/` for Phase 2; record the selected Run ID in Git. |

### Engineering checkpoint 1

**Why is a working `model.pt` insufficient evidence that an experiment is reproducible?**

`model.pt` only contains the result (weights and a few settings). It does not say which code commit produced it, which exact images and preprocessing were used, which package versions and hardware ran it, which seed and hyperparameters were set, or what validation scores it reached. It is also overwritten by every training run. Without that provenance nobody can rebuild the same model or check that the reported scores belong to that file.

## Milestone 1 - Repeatable training

**`pyproject.toml`** declares the project's direct dependencies and the allowed version ranges (for example `torch>=2.10,<2.12`), the Python range and the CPU PyTorch index. **`uv.lock`** pins the exact resolved version and hash of every package, including transitive ones, so `uv sync --locked` installs the same environment on every machine. **`.venv/`** is the local virtual environment that uv builds from the lock file; it is machine-specific, is not committed, and can always be recreated.

Changes in `src/reproducibility.py`:

- `set_reproducibility()` seeds Python `random`, NumPy, `torch.manual_seed` (CPU) and `torch.cuda.manual_seed_all` when CUDA is available, and sets `cudnn.deterministic=True`, `cudnn.benchmark=False`.
- `make_generator()` returns a `torch.Generator` seeded with `random_state`, used for DataLoader shuffling.
- `seed_worker()` seeds NumPy and `random` in each worker from `torch.initial_seed() % 2**32`.

```text
$ uv run --locked pytest -q tests/test_reproducibility.py
...                                                                      [100%]
3 passed
```

### Engineering checkpoint 2

**Does `random_state=42` guarantee bit-for-bit identical results on every CPU/GPU/software stack?**

No. The seed fixes the sequence of random numbers, but floating-point results also depend on the hardware, the kernels chosen (CPU vector instructions, cuDNN algorithms, non-deterministic GPU atomics), thread counts and the exact PyTorch/CUDA/driver versions. The seed gives repeatable results on the same machine and locked environment; across different stacks you should expect small numerical differences.

## Milestone 2 - Data versioning

I used Option A (the included archives). The checksums were verified and the images were extracted into `data/raw/`.

```text
$ uv run --locked python scripts/check_dataset.py
[train]       angular_leaf_spot 345 | bean_rust 348 | healthy 341 | TOTAL 1034 (expected 1034)
[validation]  angular_leaf_spot  44 | bean_rust  45 | healthy  44 | TOTAL  133 (expected 133)
[test]        angular_leaf_spot  43 | bean_rust  43 | healthy  42 | TOTAL  128 (expected 128)
```

`data/raw.dvc` records `nfiles: 1296`, one more than the 1,295 images. The extra file is `train/healthy/healthy_train.120tore`, a macOS `.DS_Store` file (6,148 bytes) that is part of the original `train.zip`. It explains why the iBean repository reports 1,296 files. It is kept so that `data/raw` matches the published archive exactly; the validator and `prepare.py` ignore it because they only read `.jpg/.jpeg/.png`.

DVC remote: `lab-data` → `../ai53-phase1-dvc-storage` (default).

### Engineering checkpoint 3

- **Git remote (GitHub)** stores the code, configuration, report and the small DVC pointer files (`data/raw.dvc`, later `dvc.yaml`/`dvc.lock`) together with the commit history.
- **DVC remote** (`../ai53-phase1-dvc-storage`) stores the actual image files and other large outputs, addressed by their content hash.
- **`git push` does not transfer the raw images**; it only pushes `data/raw.dvc`. The images reach the DVC remote only through `dvc push`.

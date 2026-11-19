# Lab Report: Project Phase 1 - Starter Audit

## Starter Codebase Audit

| File / Setting | Issue / Problem | Suggested Fix |
| :--- | :--- | :--- |
| `src/reproducibility.py` | Seeds are not set for Python `random`, `numpy`, or `torch`[cite: 5, 6]. | Implement `set_seed()` to set seeds for `random`, `numpy`, PyTorch CPU, and CUDA[cite: 6]. |
| `src/validate_data.py` | `validate_dataset()` is incomplete/missing data checks[cite: 7]. | Implement full checks for split existence, class names, image counts, and corrupted files[cite: 7]. |
| `src/provenance.py` | Run manifest generator is incomplete/missing system tracking[cite: 10]. | Write metadata collector using `platform`, `importlib`, and `subprocess`[cite: 10]. |
| `src/train.py` | MLflow logging functions and dataset input logging are incomplete[cite: 10, 11]. | Add MLflow run initialization, metric tracking per epoch, artifact logging, and model logging[cite: 10, 11]. |
| `dvc.yaml` | The pipeline definition file `dvc.yaml` does not exist at the project root yet[cite: 8]. | Define the 4 stages (`validate`, `prepare`, `train`, `evaluate`) in `dvc.yaml`[cite: 8, 9]. |
| `params.yaml` | Parameters need verification for downstream pipeline stages[cite: 5, 8]. | Ensure hyperparameters (learning rate, batch size, seed) are explicitly referenced by pipeline steps[cite: 8, 9]. |

---

## Engineering Checkpoint 1

**Question:** Why is a working `model.pt` insufficient evidence that an experiment is reproducible?

**Answer:** A `model.pt` file only contains the trained model weights, not the process used to produce them. It lacks the exact code version (Git commit), software package dependencies (`pyproject.toml`/`uv.lock`), random seeds, data split manifests, and hyperparameter configuration[cite: 4, 10]. Without recording these parameters and environmental factors, another developer cannot recreate the exact same model weights from scratch[cite: 1, 4].


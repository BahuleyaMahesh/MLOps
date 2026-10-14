# Data Directory (`data/`)

## Strict Data Versioning Policy

**DO NOT COMMIT RAW OR PROCESSED DATASETS DIRECTLY TO GIT.**

All datasets in this project must be managed strictly using **Data Version Control (DVC)**.
Git tracks only `.dvc` pointer files (e.g., `data/raw.dvc`, `data/processed.dvc`) and DVC-generated `.gitignore` files.

### Why This Policy Exists

1. **Academic Licensing Compliance:**
   - The EngSAF dataset (`IsmaelMousa/engsaf`) is explicitly licensed strictly for academic and research purposes with **no redistribution without permission**. Committing it to a Git remote violates these licensing terms.
2. **Repository Size & Performance:**
   - Large text datasets (e.g., StackPulse raw files reaching several hundred megabytes) degrade Git performance and violate GitHub repository limits.
3. **MLOps Reproducibility:**
   - Versioning datasets with DVC guarantees that every model experiment in MLflow can be traced back to an exact, immutable dataset hash.

---

## Directory Organization

```text
data/
├── raw/         # Raw dataset files ingested from Hugging Face (tracked by DVC)
└── processed/   # Cleaned, tokenized, and split artifacts (tracked by DVC)
```

---

## Standard Workflow

### 1. Ingesting / Updating Data
When raw data is downloaded or processed:
```bash
# Add data directory to DVC tracking
dvc add data/raw
dvc add data/processed

# Track the generated .dvc files in Git
git add data/raw.dvc data/processed.dvc data/.gitignore
git commit -m "Track updated dataset version with DVC"
```

### 2. Pushing Data to Shared Remote
Once remote storage (e.g., DagsHub) is configured:
```bash
dvc push
```

### 3. Pulling Data on a Clean Clone
When collaborating or running in Google Colab:
```bash
git pull
dvc pull
```

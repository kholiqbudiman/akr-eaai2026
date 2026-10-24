# AKR research artifact

Companion artifact for **Skip the Noise, Keep the Signal: Difficulty-Aware Knowledge Routing for Ultra-Compact Model Distillation**.

## Reproduce reported analyses (CPU; no training)

Python 3.11 or compatible Python with NumPy/SciPy:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
python analysis/reproduce.py
```

The script verifies SHA-256 checksums and generates compact comparison, BH-adjusted tests, factorial/no-TA, sensitivity, and CIFAR summaries under `generated/`. Reference outputs are in `expected/`. It checks 280 compact comparisons, 167 run wins, 21/28 positive configuration means, 600 sensitivity records, and 80 no-TA records. This command does not load datasets, train models, or require a GPU.

## Evidence selection

- Seven `kablation_*.json` waste snapshots provide shared benchmark results; `kablation_cifar10.json` is the designated CIFAR snapshot. `kablation_cifar10_latest.json` is retained but excluded from aggregation.
- AAKD uses `aakd_benchmark.json`. Compact students have fewer than 100,000 parameters. Pairing key: task K, student, seed.
- Factorial Full-AKR/Vanilla arms reuse benchmark records; NoSkip/SkipOnly use `factorial_results.json`.
- No-TA uses the pre-existing `nota_complete/akr_nota_merged.json`: 80 unique keys, with exact record matches to worker JSONs. Other no-TA snapshots are retained for provenance and are not pooled. Inclusion follows the merged archive, not a best-score rule.
- Sensitivity uses the 12 explicitly named K=6/28, Micro/Small files. Additional `_g1` files are retained but not added as independent runs.
- Main task-level tests form one retrospective BH family of 84 contrasts; paired AKR comparisons form a separate family of 11. Task variants share image sources; seeds are not independent datasets.

## Relationship to CDR

The CDR companion preprint is https://doi.org/10.2139/ssrn.7635173. Dataset preparation, teachers, training infrastructure, and benchmark result archives are shared. Archive counts are not counts of experiments newly performed for AKR. AKR contributes sample routing and component/sensitivity analyses; shared results should not be counted as independent corroboration.

## Training source and scope

`scripts/` contains retained historical training and analysis source, including the main K-ablation runner and no-TA runner. These files preserve original paths and behavior; they are not run by the reproduction command. Fresh training additionally requires PyTorch/torchvision, scikit-learn and script-specific dependencies, datasets, teacher checkpoints and caches at the configured paths. The original training environment was not captured in a complete lockfile. Teacher weights and datasets are not bundled here.

Some local WaRP dataset versions contain train/val/test folders. The retained benchmark/no-TA/sensitivity runners use a held-out loader for both checkpoint selection and final scoring. The artifact reproduces the historical results; it does not retrospectively create an independent test protocol.

The default reproduction covers the numerical analyses listed above, not every qualitative figure, Fruits-domain statistic or R meta-analysis in the manuscript. Historical scripts are included for inspection. Checksum manifests establish the released snapshot, not an unrecorded source commit for every historical run.

No new training was run to prepare this release. JSON source records are preserved unchanged.

# AKR: Adaptive Knowledge Routing

Reproducibility bundle for the *Engineering Applications of Artificial Intelligence* submission:

> **Skip the Noise, Keep the Signal: Difficulty-Aware Knowledge Routing for Ultra-Compact Model Distillation**
> Kholiq Budiman, Aufaclav Zatu Kusuma Frisky, Jazi Eko Istiyanto
> Universitas Gadjah Mada, 2026

## Status

Manuscript is under review at *Engineering Applications of Artificial Intelligence* (Elsevier).

Companion diagnostic paper is available as an SSRN preprint and under review at *Neural Networks*:

> Budiman, K., Istiyanto, J.E., Frisky, A.Z.K. (2026). **When Dark Knowledge Becomes Noise: Two Failure Modes of Distillation in Compact Neural Networks.** SSRN preprint.

## Planned release upon acceptance

- Training scripts (PyTorch) for AKR and 11 comparison KD methods
- Per-seed result CSVs across 7 waste datasets and CIFAR-10 (10 and 5 seeds respectively)
- Difficulty-partition logs (per-sample teacher confidence margins and route assignments)
- INT8-quantised student checkpoints for the compact-scale students
- Frozen commit hash of the training pipeline used for each experimental campaign

## Scope condition

AKR is designed for compact-scale, imbalanced classification and is recommended when the imbalance ratio exceeds 2 and the Capacity-Dimensionality Ratio is below 10,000. On balanced datasets (e.g., CIFAR-10 at IR = 1.0), the routing signal collapses and the method should not be used. This scope condition is reported openly and diagnosed in Section 5 of the manuscript.

## Contact

Corresponding author: Aufaclav Zatu Kusuma Frisky (`aufaclav@ugm.ac.id`), Department of Computer Science and Electronics, Universitas Gadjah Mada.

## Licence

Code will be released under the MIT licence. Data is subject to the original dataset licences (WaRP-C: CC BY 4.0; TrashNet: open academic benchmark; Waste Classification K=2: CC via Kaggle; CIFAR-10: research commons).

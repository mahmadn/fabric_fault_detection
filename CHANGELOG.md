# Changelog

Every meaningful change to the pipeline, dataset, or model is documented here.
Each entry records what changed, what the evidence said, and what was decided.

---

## [Current] — v1.2 · Cleaned Dataset

**Model:** SVM · RBF kernel · C=1.0 · class_weight='balanced'
**Dataset:** 1382 tiles (207 normal · 1175 faulty: miss-pick, double-pick, broken-pick)
**Features:** 12 (3 intensity + 4 spatial + 1 texture + 4 row projection)

| Metric | Value |
|---|---|
| F1 (QC threshold) | **0.930** |
| ROC-AUC | **0.957** |
| Recall at QC threshold | **0.901** |
| Precision at QC threshold | 0.937 |
| Train/test gap | **0.000** |
| QC threshold | 0.641 |
| Faults missed (FN) | 117 / 1175 (10.0%) |
| False alarms (FP) | 43 / 207 (20.8%) |

**What changed from v1.1:**
Model analysed its own false negatives. 24 tiles with P(fault) < 0.40 confirmed as mislabeled (no fault visible in original or residual). Removed from training set. FN rate: 14.1% → 10.0%.

**Decision record:** `suspicious_threshold = 0.40` — tiles where model assigns < 40% fault probability and is labelled faulty. Visual inspection confirmed all 24. The 145 borderline FNs (P = 0.40–0.641) are genuinely subtle faults or annotation ambiguities — not removable without domain expert review.

---

## [v1.1] — Full Dataset · All Fault Types

**Dataset:** 1406 tiles (207 normal · 1199 faulty)

| Metric | Value |
|---|---|
| F1 (CV, thresh=0.50) | 0.905 |
| ROC-AUC | 0.948 |
| Miss-pick recall | 0.829 |
| Double-pick recall | 0.835 |
| Broken-pick recall | 0.882 |

**What changed from v1.0:**
- Added double-pick (85 tiles) and broken-pick (17 tiles) to FAULTY class
- Edge margin analysis: tested margins 0, 32, 64, 96px across all 12 features
  - margin=32 improved global stats (kurt +0.433, std +0.327) but degraded row_peak_width (−0.160)
  - Decision: margin=0 retained (find_peaks provides implicit edge protection)
- Tukey windowing tested and abandoned (edge ratio 1.25x → 1.55x — worse)
- Normalisation validated: OLD wins 12/12 features, AUC delta +0.182 vs shared
- QC threshold tuned: 0.50 → 0.641 (recall ≥ 0.90 constraint)

**Experiments:** `03_edge_margin_tukey.ipynb`, `01_normalisation_comparison.ipynb`

---

## [v1.0] — Row Projection Features

**Dataset:** ~400 tiles (miss-pick only) · **Features:** 10

| Metric | Value |
|---|---|
| F1 | 0.878 |
| ROC-AUC | 0.948 |
| Train/test gap | 0.006 ✓ |

**What changed from v0.3:**
Domain insight: miss-pick faults are horizontal stripes → project residual onto row axis → detect spike. Four features added: `row_max_prominence` (KS=0.720), `row_sum_prominence` (0.607), `row_proj_std` (0.440), `row_peak_width` (0.607).

`row_max_prominence` alone has KS=0.720 — 1.6× stronger than the best previous feature (grid_mean_std, 0.493).

**Alternative tested:** PCA of raw 896-dim row projection. 21 components for 95% variance → F1=0.857. Hand-crafted 4 features → F1=0.878. Domain knowledge won.

**Experiments:** `04_pca_vs_handcrafted.ipynb`

---

## [v0.3] — Feature Pruning

**Dataset:** ~400 tiles · **Features:** 6 (pruned from 14)

| Metric | Value |
|---|---|
| F1 | 0.739 |
| Train/test gap | 0.019 ✓ |

**What changed from v0.2:**
Correlation analysis identified redundant clusters. Removed features with KS < 0.15 and r > 0.80 with a higher-KS feature. Sample/feature ratio: 28:1 → 67:1.

Learning curve showed test F1 flat (slope ≈ 0) — feature ceiling reached, not data ceiling. Decision: add domain features rather than collect more data.

---

## [v0.2] — Structural Features

**Dataset:** ~400 tiles · **Features:** 51

| Metric | Value |
|---|---|
| F1 | 0.792 |
| Train/test gap | 0.183 ⚠ |

**What changed from v0.1:**
Added 6 feature groups: intensity percentiles, histogram bins, spatial grid, connected components, blob shape, texture. 51 features total.

Sample/feature ratio: 400/51 = 8:1 — below the 20:1 minimum for reliable generalisation. Result: severe overfitting (gap = 0.183).

---

## [v0.1] — Baseline

**Dataset:** 200 tiles (100 normal · 100 miss-pick) · **Model:** Random Forest

| Metric | Value |
|---|---|
| F1 | 0.601 |
| ROC-AUC | 0.724 |

**Notes:**
3 features: `old_mean`, `old_p99`, `old_max`. All correlated at r > 0.80 — effectively 1 feature. New normalisation tested: KS = 0.110 vs 0.265 for OLD on residual_mean. OLD retained throughout.

---

## Decisions log

| Decision | Evidence | Outcome |
|---|---|---|
| OLD normalisation over NEW | 12/12 KS wins · AUC Δ +0.182 | Retained throughout |
| SVM over Random Forest | Lower gap (0.004 vs 0.082) on same data | SVM in production |
| margin=0 over margin=32 | row_peak_width KS 0.580→0.420 outweighed gains | margin=0 retained |
| Tukey windowing abandoned | Edge ratio 1.25x → 1.55x (worse) | No windowing |
| Hand-crafted projection over PCA | F1 0.878 vs 0.857 · 4 vs 21 features | Hand-crafted retained |
| 12 features not 51 | Gap 0.183→0.004 · KS + correlation pruning | 12 final features |
| Drop 24 mislabeled tiles | P<0.40 + visual confirmation | FN 14.1%→10.0% |

---

## Roadmap

- [ ] FFT grid search: N_PEAKS ∈ {7,9,11,13} × PEAK_RADIUS ∈ {5..10} (needs 150+ tiles/type)
- [ ] Column projection features for warp-direction faults (broken end, double end)
- [ ] Dataset expansion: 23 fault types identified · review pipeline built
- [ ] Stage 2 fault-type classifier
- [ ] Inference latency benchmarking on target hardware

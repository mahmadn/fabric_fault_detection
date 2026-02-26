"""
features.py
-----------
Feature extraction from FFT residual maps.

All 12 features are computed on the residual map output by pipeline.compute_residual().
Features were selected by:
  1. Computing KS statistic (class separation) for every candidate feature
  2. Removing features with KS < 0.15 (insufficient signal)
  3. Removing correlated features (r > 0.80) — keeping highest KS of each cluster
  4. Final validation: train/test gap < 0.05, sample/feature ratio > 20:1

Feature groups:
  Intensity  (3): overall residual magnitude
  Spatial    (4): where is the energy concentrated?
  Texture    (1): are fault boundaries sharp?
  Projection (4): is there a horizontal stripe? (miss-pick detector)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def extract_features(res):
    """
    Extract 12 features from a residual map.

    Args:
        res: float32 residual map, shape (H, W), output of pipeline.compute_residual()

    Returns:
        dict of 12 float features
    """
    h, w  = res.shape
    flat  = res.ravel()
    f     = {}

    # ── Intensity ─────────────────────────────────────────────────
    # Global magnitude of the residual.

    # p95: 95th percentile. More robust than max (ignores single bright pixels
    # from noise) and more discriminative than mean (averaged over the large
    # near-zero background). KS=0.373.
    f['p95'] = float(np.percentile(flat, 95))

    # std: standard deviation. Faults create a bimodal residual — near-zero
    # background + bright fault region — producing high std. KS=0.400.
    f['std'] = float(flat.std())

    # kurt: kurtosis (tail heaviness). Normal tiles have roughly Gaussian
    # residuals. Faults create a few extremely bright pixels → heavy right
    # tail → high kurtosis. KS=0.227.
    f['kurt'] = float(pd.Series(flat).kurt())

    # ── Spatial non-uniformity (3×3 grid) ─────────────────────────
    # Divide residual into 9 cells. A fault is localised — some cells
    # will be bright, others dark. Uniform tiles have similar cell means.
    grid_means = [
        float(res[i * h // 3:(i + 1) * h // 3,
                  j * w // 3:(j + 1) * w // 3].mean())
        for i in range(3) for j in range(3)
    ]

    # grid_mean_std: std of 9 cell means. KS=0.493 (strongest non-projection feature).
    f['grid_mean_std']  = float(np.std(grid_means))

    # grid_max_ratio: hottest cell / average cell. How much one region stands out.
    # KS=0.487.
    f['grid_max_ratio'] = float(max(grid_means) / (np.mean(grid_means) + 1e-12))

    # grid_11_max: peak value in centre cell. Tiles are often centred during
    # imaging, so faults appear near centre. KS=0.393.
    f['grid_11_max']    = float(res[h // 3:2 * h // 3, w // 3:2 * w // 3].max())

    # ── Spatial concentration ─────────────────────────────────────
    # gini: Gini coefficient of residual values.
    # Borrowed from economics (income inequality).
    # gini=0: perfectly uniform residual (normal tile).
    # gini≈1: all energy concentrated in one small region (fault).
    # KS=0.327.
    s = np.sort(flat)
    n = len(s)
    f['gini'] = float(
        (2 * np.sum(np.arange(1, n + 1) * s)) / (n * s.sum() + 1e-12) - (n + 1) / n
    )

    # ── Texture sharpness ─────────────────────────────────────────
    # grad_p95: 95th percentile of gradient magnitude.
    # Real faults have sharp spatial boundaries (steep residual gradients).
    # Noise from normal tiles is smooth (gentle gradients).
    # KS=0.407.
    gy, gx = np.gradient(res)
    f['grad_p95'] = float(np.percentile(np.sqrt(gx**2 + gy**2), 95))

    # ── Row projection — miss-pick / horizontal stripe detector ───
    # Domain insight: miss-pick faults remove a weft (horizontal) thread.
    # Projecting the residual onto the row axis (mean per row) converts
    # the 2D stripe detection into a 1D spike detection problem.
    # find_peaks requires a valley on both sides → automatically ignores
    # edge artifacts at row 0 and row H-1 (no left/right neighbour).
    row_proj = res.mean(axis=1)          # shape: (H,)
    row_s    = gaussian_filter1d(row_proj, sigma=3.0)
    detrend  = row_s - np.median(row_s)  # remove baseline drift

    peaks, props = find_peaks(detrend, prominence=0.01, distance=10)
    prom = props['prominences'] if len(peaks) > 0 else np.array([0.0])

    # row_max_prominence: height of tallest spike above surrounding baseline.
    # Single strongest feature. KS=0.720.
    f['row_max_prominence'] = float(prom.max())

    # row_sum_prominence: total spike energy across all detected spikes.
    # Useful when multiple threads are missing. KS=0.607.
    f['row_sum_prominence'] = float(prom.sum())

    # row_proj_std: variability of the row projection.
    # Flat profile → normal. Spiked profile → faulty. KS=0.440.
    f['row_proj_std'] = float(row_proj.std())

    # row_peak_width: how many rows exceed half the tallest spike height.
    # Encodes physical fault thickness. KS=0.607.
    if len(peaks) > 0:
        best_peak = peaks[np.argmax(prom)]
        half_h    = detrend[best_peak] / 2
        f['row_peak_width'] = int((detrend > half_h).sum())
    else:
        f['row_peak_width'] = 0

    return f


def tile_to_features(path, pipeline_fn=None):
    """
    Full pipeline: path → feature dict.

    Args:
        path: path to tile image
        pipeline_fn: callable(path) → residual. Defaults to pipeline.tile_to_residual.

    Returns:
        dict of features, or None if tile cannot be read.
    """
    if pipeline_fn is None:
        from src.pipeline import tile_to_residual
        pipeline_fn = tile_to_residual

    res = pipeline_fn(path)
    if res is None:
        return None
    try:
        return extract_features(res)
    except Exception as e:
        import os
        print(f'  WARN {os.path.basename(str(path))}: {e}')
        return None


# Feature metadata: name → (group, ks_stat, description)
FEATURE_METADATA = {
    'p95'               : ('intensity',  0.373, '95th percentile of residual'),
    'std'               : ('intensity',  0.400, 'Standard deviation of residual'),
    'kurt'              : ('intensity',  0.227, 'Kurtosis — tail heaviness'),
    'grid_mean_std'     : ('spatial',    0.493, 'Std of 3×3 grid cell means'),
    'grid_max_ratio'    : ('spatial',    0.487, 'Hottest grid cell / average'),
    'grid_11_max'       : ('spatial',    0.393, 'Peak residual in centre cell'),
    'gini'              : ('spatial',    0.327, 'Gini coefficient of residual'),
    'grad_p95'          : ('texture',    0.407, '95th pct of gradient magnitude'),
    'row_max_prominence': ('projection', 0.720, 'Tallest row projection spike'),
    'row_sum_prominence': ('projection', 0.607, 'Total row spike energy'),
    'row_proj_std'      : ('projection', 0.440, 'Row projection variability'),
    'row_peak_width'    : ('projection', 0.607, 'Width of tallest spike (rows)'),
}

FEATURE_COLS = list(FEATURE_METADATA.keys())

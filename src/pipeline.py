"""
pipeline.py
-----------
Core FFT residual pipeline for fabric fault detection.

The idea: fabric has a periodic weave structure. FFT isolates that
periodic background. Subtracting it reveals what doesn't belong —
the residual. Faults live in the residual; normal tiles have near-zero residuals.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure

# ── FFT parameters ────────────────────────────────────────────────
# N_PEAKS: how many frequency peaks to keep for reconstruction.
#   More peaks = tighter reconstruction = smaller residuals overall.
#   Too few: misses weave harmonics, normal tiles get high residual.
#   Too many: starts capturing fault frequencies, misses faults.
#   Value of 11 validated visually on miss-pick data. Systematic
#   grid search planned once dataset reaches 150+ tiles per fault type.
N_PEAKS = 11

# PEAK_RADIUS: size of mask around each frequency peak (pixels in FFT space).
#   Larger = captures more energy per peak.
#   Value of 8 validated visually on miss-pick data.
PEAK_RADIUS = 8

# DC_SUPPRESS_RADIUS: suppress DC component (average brightness).
#   DC has nothing to do with weave structure — just overall tile brightness.
DC_SUPPRESS_RADIUS = 3

# RESIDUAL_SMOOTH: Gaussian smoothing sigma applied to residual.
#   Reduces single-pixel noise while preserving fault regions (which are larger).
RESIDUAL_SMOOTH = 5


def apply_clahe(gray_float):
    """
    Contrast Limited Adaptive Histogram Equalisation.

    Enhances local contrast so thread crossings are crisp regardless
    of overall tile brightness variation. Input: grayscale 0-255 float32.
    Output: 0-1 float32.
    """
    return exposure.equalize_adapthist(
        gray_float / 255.0, clip_limit=0.02
    ).astype(np.float32)


def compute_residual(img,
                     n_peaks=N_PEAKS,
                     peak_radius=PEAK_RADIUS,
                     dc_suppress_radius=DC_SUPPRESS_RADIUS,
                     smooth_sigma=RESIDUAL_SMOOTH):
    """
    Compute the FFT residual map for a fabric tile.

    Pipeline:
      1. FFT of mean-centred image
      2. Pick n_peaks strongest frequencies (the weave pattern)
      3. Reconstruct background from those peaks only
      4. Residual = |original - reconstruction|, independently normalised

    Why independent normalisation (OLD style):
      When a thread is missing, the reconstruction loses some energy —
      its dynamic range shrinks slightly. Independent min-max scaling
      amplifies this mismatch proportionally. Shared normalisation
      suppresses it. Validated: OLD wins 12/12 features in KS comparison,
      AUC delta = +0.182 over shared normalisation.

    Args:
        img: CLAHE-processed grayscale tile, float32, shape (H, W)

    Returns:
        residual: float32 array shape (H, W), values roughly 0..0.5
    """
    h, w   = img.shape
    cy, cx = h // 2, w // 2

    # FFT and shift DC to centre
    F      = np.fft.fft2(img - img.mean())
    Fshift = np.fft.fftshift(F)
    mag    = np.log1p(np.abs(Fshift))

    # Suppress DC component before peak picking
    mag_copy    = mag.copy()
    y_, x_      = np.ogrid[:h, :w]
    dc_mask     = (x_ - cx)**2 + (y_ - cy)**2 <= dc_suppress_radius**2
    mag_copy[dc_mask] = 0.0

    # Pick n_peaks strongest frequencies, enforcing minimum separation
    flat_idx    = np.argsort(mag_copy.ravel())[::-1]
    peak_coords = []
    for idx in flat_idx:
        if len(peak_coords) >= n_peaks:
            break
        yy, xx = idx // w, idx % w
        if mag_copy[yy, xx] <= 0:
            continue
        too_close = any(
            (yy - py)**2 + (xx - px)**2 <= (peak_radius * 1.5)**2
            for py, px in peak_coords
        )
        if too_close:
            continue
        peak_coords.append((yy, xx))

    # Build frequency mask (peaks + conjugate symmetry + DC)
    mask = np.zeros((h, w), dtype=np.uint8)
    for (py, px) in peak_coords:
        yy, xx = np.ogrid[:h, :w]
        mask[(yy - py)**2 + (xx - px)**2 <= peak_radius**2] = 1
        # Conjugate symmetry: each peak has a mirror
        symy, symx = (h - py) % h, (w - px) % w
        mask[(yy - symy)**2 + (xx - symx)**2 <= peak_radius**2] = 1
    mask[cy - 2:cy + 3, cx - 2:cx + 3] = 1  # keep DC

    # Reconstruct periodic background from weave peaks only
    recon_raw  = np.real(
        np.fft.ifft2(np.fft.ifftshift(Fshift * mask))
    ).astype(np.float32)

    # Independent min-max normalisation (see docstring for why)
    orig_norm  = (img       - img.min())       / (img.max()       - img.min()       + 1e-12)
    recon_norm = (recon_raw - recon_raw.min()) / (recon_raw.max() - recon_raw.min() + 1e-12)

    residual = np.abs(orig_norm - recon_norm)
    return ndi.gaussian_filter(residual, sigma=smooth_sigma)


def load_tile(path):
    """
    Load a JPEG/PNG tile and return grayscale float32.
    Returns None if file cannot be read.
    """
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def tile_to_residual(path):
    """
    Full pipeline: path → smoothed residual map.
    Returns None if tile cannot be read.
    """
    gray = load_tile(path)
    if gray is None:
        return None
    return compute_residual(apply_clahe(gray))

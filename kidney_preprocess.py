"""Phase 1 (KIDNEY member): CT windowing, spatial + frequency enhancement, quality metrics.

Algorithms in this file (chosen so they do not overlap with the tumor member's):
  - HU window [-100, 300] -> 8-bit           (kidney cortex enhances to ~150-250 HU after contrast)
  - Bilateral filter (edge-preserving)        compared against Gaussian smoothing
  - CLAHE vs global Histogram Equalization vs Gamma correction (contrast)
  - 2D DFT log-magnitude spectrum + Gaussian low-pass filter (frequency domain)
  - PSNR and CNR to quantify each step
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

WINDOW = (-100, 300)          # HU range kept; everything outside is clipped
BODY_HU = -300                # air is about -1000 HU, fat about -100 HU: anything above -300 is the patient
BONE_HU = 350                 # cortical bone is several hundred HU, enhanced kidney stays below this
FINAL_CONTRAST = None         # default contrast step: None, "clahe", "he" or "gamma". run_kidney_pipeline.py
                              # picks the best one per strategy (results/chosen_params.json).
LOWPASS_CUTOFF = 50           # default Gaussian low-pass cutoff D0; the pipeline picks it by training CNR.


# ---------------------------------------------------------------- intensity normalization
def window_normalize(hu, lo=WINDOW[0], hi=WINDOW[1]):
    """Clip HU to [lo, hi] and stretch linearly to the 8-bit range 0..255."""
    clipped = np.clip(hu, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def body_mask(hu):
    """The patient's body: everything denser than air, largest connected blob, holes filled
    (removes the scanner table and air outside the patient)."""
    parts = label(hu > BODY_HU)
    if parts.max() == 0:
        return np.zeros(hu.shape, bool)
    largest = np.argmax(np.bincount(parts.ravel())[1:]) + 1
    return ndi.binary_fill_holes(parts == largest)


def bone_mask(hu, threshold=BONE_HU, max_hole=5000):
    """Bone (spine, ribs) from the RAW HU values. After windowing, bone and enhanced kidney both
    saturate near 255, so bone has to be found before the window throws that information away.
    Only the hard outer shell of a vertebra is above the threshold; the spongy bone inside is
    ~150-300 HU, the same as enhanced kidney. So small regions enclosed by bone are filled in."""
    bone = ndi.binary_closing(hu > threshold, structure=np.ones((5, 5), bool))
    holes = label(ndi.binary_fill_holes(bone) & ~bone)
    small = np.bincount(holes.ravel()) < max_hole
    small[0] = False
    return ndi.binary_dilation(bone | small[holes], iterations=2)


# ---------------------------------------------------------------- spatial domain filtering
def denoise(img, method="bilateral", sigma=1.5, d=7, sigma_color=40, sigma_space=5):
    """Spatial smoothing.
    bilateral: weights neighbours by distance AND by grey-level similarity, so pixels across an
               organ edge (very different grey level) barely contribute -> edges stay sharp.
    gaussian : weights neighbours by distance only -> smooths noise and edges alike (baseline)."""
    if method == "bilateral":
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), sigma)
    raise ValueError(method)


def enhance_contrast(img, method="clahe", clip=2.0, tiles=8, gamma=0.8):
    """Contrast enhancement.
    clahe: histogram equalization in small tiles with a clip limit (limits noise amplification)
    he   : one global histogram equalization for the whole image
    gamma: power-law mapping; gamma < 1 brightens mid-greys, gamma > 1 darkens them"""
    if method == "clahe":
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(img)
    if method == "he":
        return cv2.equalizeHist(img)
    if method == "gamma":
        return (255 * (img / 255.0) ** gamma).astype(np.uint8)
    raise ValueError(method)


# ---------------------------------------------------------------- frequency domain filtering
def log_spectrum(img):
    """Centred log-magnitude spectrum of the 2D DFT, scaled to 0..255 for display."""
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))))
    return (mag / mag.max() * 255).astype(np.uint8)


def gaussian_lowpass(img, cutoff=50):
    """Gaussian low-pass filter in the frequency domain: H(u,v) = exp(-D(u,v)^2 / (2 * D0^2)).
    Unlike an ideal low-pass it has no sharp cut-off, so it causes no ringing artefacts."""
    rows, cols = img.shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    dist2 = u[:, None] ** 2 + v[None, :] ** 2
    h = np.exp(-dist2 / (2.0 * cutoff ** 2))
    spectrum = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * h)))
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- final chain
def enhance(hu, contrast=FINAL_CONTRAST, cutoff=LOWPASS_CUTOFF):
    """Final Phase 1 chain: window -> bilateral -> (optional contrast step) -> Gaussian low-pass."""
    img = denoise(window_normalize(hu), "bilateral")
    if contrast:
        img = enhance_contrast(img, contrast)
    return gaussian_lowpass(img, cutoff)


# ---------------------------------------------------------------- quality metrics
def psnr(reference, processed):
    """Peak signal-to-noise ratio in dB: how close the processed image stays to the reference."""
    mse = np.mean((reference.astype(np.float64) - processed.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def cnr(img, roi_mask, background_mask):
    """Contrast-to-noise ratio: |mean(kidney) - mean(surroundings)| / std(surroundings)."""
    roi, bg = img[roi_mask].astype(np.float64), img[background_mask].astype(np.float64)
    return float(abs(roi.mean() - bg.mean()) / (bg.std() + 1e-8))


def surrounding_ring(mask, body, width=10):
    """Band of tissue just outside the kidney (inside the body): the background for CNR."""
    return ndi.binary_dilation(mask, iterations=width) & ~mask & body


if __name__ == "__main__":
    # Self-check on synthetic data: a bright disk on noisy background.
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:128, :128]
    disk = (yy - 64) ** 2 + (xx - 64) ** 2 < 25 ** 2
    hu = (np.where(disk, 200, 40) + rng.normal(0, 25, (128, 128))).astype(np.float32)
    base = window_normalize(hu)
    ring = surrounding_ring(disk, np.ones_like(disk))
    out = enhance(hu)
    assert out.shape == hu.shape and out.dtype == np.uint8
    assert cnr(out, disk, ring) > cnr(base, disk, ring), "enhancement must raise CNR"
    for m in ("clahe", "he", "gamma"):
        assert enhance_contrast(base, m).dtype == np.uint8
    print("kidney_preprocess self-check OK")

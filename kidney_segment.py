"""Phase 2 (KIDNEY member): morphology + two classical kidney segmentation strategies + overlap metrics.

Algorithms in this file (no overlap with the tumor member's Multi-Otsu / seeded region growing):
  - Strategy A: adaptive local thresholding (threshold = Gaussian-weighted local mean - offset)
  - Strategy B: marker-controlled watershed on the Sobel gradient
  - Custom structuring elements: hand-built cross (marker erosion), diamond (opening), disk (closing)
  - Anatomical rule filter: kidneys lie in the back half of the body, one on each side of the spine
  - Dice and IoU against the radiologist mask

Input image = Phase 1 output (uint8). The raw HU slice is only used for the body and bone masks.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import diamond, disk
from skimage.segmentation import watershed

import kidney_preprocess as pre

MIN_KIDNEY_PIXELS = 150        # smaller blobs are noise / vessels, not a kidney cross-section
MAX_KIDNEY_PIXELS = 12000      # larger blobs are liver / bowel merged into one region
# Where a kidney may lie relative to the spine (fractions of body height) and how big it may be.
# run_kidney_pipeline.py replaces this with a window LEARNED from the training ground truth.
DEFAULT_WINDOW = {"dx_min": 0.08, "dx_max": 0.45, "dy_min": -0.25, "dy_max": 0.12,
                  "area_min": MIN_KIDNEY_PIXELS, "area_max": MAX_KIDNEY_PIXELS}

# Default parameters. run_kidney_pipeline.py sweeps them on TRAINING cases only and writes the
# chosen values to results/chosen_params.json, which the pipeline and the app then use.
DEFAULTS = {
# Grey-level thresholds are PERCENTILES of the patient's own body pixels, not fixed grey levels:
# how much the kidney brightens depends on the scan timing after contrast injection, so a fixed
# level tuned on some patients does not carry over to others.
    "adaptive": {"bone_hu": 350, "block_size": 101, "offset": -5, "min_pct": 60, "open_r": 2, "close_r": 3},
    "watershed": {"bone_hu": 350, "high_pct": 90, "low_pct": 55, "open_r": 2, "close_r": 3},
}


# ---------------------------------------------------------------- custom structuring elements
def cross_se(r=1):
    """Hand-built plus-shaped SE: erodes along rows and columns only, so thin diagonal
    bridges survive less than the solid kidney core. Used to shrink watershed markers."""
    se = np.zeros((2 * r + 1, 2 * r + 1), bool)
    se[r, :] = se[:, r] = True
    return se


def structuring_elements(open_r=2, close_r=3):
    """The SEs used in this file, for the report figure."""
    return {"cross (markers)": cross_se(2), f"diamond r={open_r} (opening)": diamond(open_r),
            f"disk r={close_r} (closing)": disk(close_r)}


# ---------------------------------------------------------------- morphology
def remove_small_objects(mask, min_pixels):
    """Drop connected components smaller than min_pixels (written out because the scikit-image
    version of this function changed its parameters between releases)."""
    parts = label(mask)
    sizes = np.bincount(parts.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[parts]


def refine_mask(mask, open_r=2, close_r=3, min_pixels=MIN_KIDNEY_PIXELS):
    """Opening (diamond) cuts thin bridges, e.g. kidney touching bowel or a vessel.
    Closing (disk) bridges small gaps, e.g. the notch of the renal sinus.
    Hole filling closes the darker renal pelvis inside the kidney; tiny blobs are removed."""
    if open_r:
        mask = ndi.binary_opening(mask, structure=diamond(open_r))
    if close_r:
        mask = ndi.binary_closing(mask, structure=disk(close_r))
    mask = ndi.binary_fill_holes(mask)
    return remove_small_objects(mask, min_pixels)


def search_region(hu, bone_hu=pre.BONE_HU):
    """Where a kidney can be: inside the body, but not bone. A lower bone_hu also removes bowel
    filled with oral contrast, which is as bright as bone."""
    return pre.body_mask(hu) & ~pre.bone_mask(hu, bone_hu)


# ---------------------------------------------------------------- Strategy A: adaptive thresholding
def segment_adaptive(img, region, block_size=101, offset=-5, min_pct=60, open_r=2, close_r=3):
    """A pixel is a candidate if it is brighter than its own neighbourhood (local Gaussian mean
    over block_size x block_size, minus offset) AND brighter than the min_pct percentile of the
    body (rules out fat, which can be locally bright next to air). A single global threshold fails
    because brightness drifts across the abdomen; the local threshold adapts to that drift."""
    local = threshold_local(img, block_size, method="gaussian", offset=offset)
    min_level = np.percentile(img[region], min_pct)
    candidate = (img > local) & (img > min_level) & region
    return refine_mask(candidate, open_r, close_r)


# ---------------------------------------------------------------- Strategy B: marker-controlled watershed
def watershed_markers(img, region, high, low):
    """Markers: label 1 = sure background (dark tissue, fat, outside the region);
    labels 2.. = sure bright organ cores (bright pixels, eroded with the cross SE so markers sit
    well inside an organ); 0 = unknown, decided by the watershed."""
    markers = np.zeros(img.shape, np.int32)
    markers[(img <= low) | ~region] = 1
    cores = remove_small_objects(ndi.binary_erosion((img >= high) & region, structure=cross_se(2)), 20)
    core_labels = label(cores)
    markers[cores] = core_labels[cores] + 1
    return markers


def segment_watershed(img, region, high_pct=90, low_pct=55, open_r=2, close_r=3):
    """Flood the Sobel gradient magnitude from the markers: each region grows until it meets
    another at a gradient ridge (an organ edge). Pixels reached from a bright core = candidate.
    Marker levels = percentiles of the body's grey levels (adapts to each patient's contrast)."""
    high, low = np.percentile(img[region], [high_pct, low_pct])
    markers = watershed_markers(img, region, high, low)
    regions = watershed(sobel(img), markers)
    return refine_mask(regions > 1, open_r, close_r)


# ---------------------------------------------------------------- candidates + anatomical rule
def candidates(mask, min_pixels=MIN_KIDNEY_PIXELS):
    """Connected components big enough to be a kidney cross-section (list of bool masks)."""
    parts = label(mask)
    return [parts == r.label for r in regionprops(parts) if r.area >= min_pixels]


def anatomy(hu):
    """Landmarks used by the rule filter and the position features:
    spine centre (row, col) and body height. The spine = the biggest bone blob near the body's
    midline in its back half; if none is found, a typical position is assumed."""
    body = pre.body_mask(hu)
    rows, cols = np.where(body)
    top, height, mid = rows.min(), rows.max() - rows.min() + 1, (cols.min() + cols.max()) / 2
    width = cols.max() - cols.min() + 1
    best, spine = 0, (top + 0.7 * height, mid)
    for r in regionprops(label(pre.bone_mask(hu) & body)):
        cy, cx = r.centroid
        if abs(cx - mid) < 0.15 * width and cy > top + 0.5 * height and r.area > best:
            best, spine = r.area, (cy, cx)
    return {"spine_row": spine[0], "spine_col": spine[1], "height": height}


def blob_position(blob, a):
    """Sideways distance from the spine and depth relative to it (negative = in front), both as
    fractions of body height, plus the blob's area."""
    r = regionprops(blob.astype(np.uint8))[0]
    cy, cx = r.centroid
    return abs(cx - a["spine_col"]) / a["height"], (cy - a["spine_row"]) / a["height"], r.area, r.bbox


def learn_window(slices, margin=0.03):
    """Anatomical window learned from the TRAINING ground truth: where kidney blobs sit relative to
    the spine (2nd-98th percentile, widened by a margin) and how big they are."""
    dx, dy, area = [], [], []
    for s in slices:
        a = anatomy(s["hu"])
        for blob in candidates(s["kidney"]):
            x, y, ar, _ = blob_position(blob, a)
            dx.append(x), dy.append(y), area.append(ar)
    lo, hi = np.percentile([dx, dy, area], 2, axis=1), np.percentile([dx, dy, area], 98, axis=1)
    return {"dx_min": float(max(lo[0] - margin, 0)), "dx_max": float(hi[0] + margin),
            "dy_min": float(lo[1] - margin), "dy_max": float(hi[1] + margin),
            "area_min": float(max(lo[2] * 0.7, MIN_KIDNEY_PIXELS)), "area_max": float(hi[2] * 1.3)}


def rule_filter(mask, hu, window=None):
    """Anatomical prior instead of a classifier (the classifier in Phase 3 replaces this):
    a kidney does not cross the spine's column, lies inside the learned position window relative to
    the spine (this excludes the aorta / vena cava right in front of the spine, and liver / spleen
    far forward or far out) and has kidney-like size. On each side the largest such blob is kept."""
    w, a = window or DEFAULT_WINDOW, anatomy(hu)
    best = {}                                          # side -> (area, blob)
    for blob in candidates(mask):
        dx, dy, area, bbox = blob_position(blob, a)
        if bbox[1] < a["spine_col"] < bbox[3]:
            continue
        if not (w["dx_min"] <= dx <= w["dx_max"] and w["dy_min"] <= dy <= w["dy_max"]
                and w["area_min"] <= area <= w["area_max"]):
            continue
        side = "left" if bbox[3] <= a["spine_col"] else "right"
        if area > best.get(side, (0, None))[0]:
            best[side] = (area, blob)
    out = np.zeros(mask.shape, bool)
    for _, blob in best.values():
        out |= blob
    return out


def segment_kidney(enhanced, hu, method="watershed", rule=True, window=None, **params):
    """Phase 2 entry point: raw candidate mask of one strategy, optionally rule-filtered."""
    p = {**DEFAULTS[method], **params}
    region = search_region(hu, p.pop("bone_hu"))
    raw = segment_adaptive(enhanced, region, **p) if method == "adaptive" else segment_watershed(enhanced, region, **p)
    return rule_filter(raw, hu, window) if rule else raw


# ---------------------------------------------------------------- overlap metrics
def dice(pred, gt):
    """Dice = 2|A n B| / (|A| + |B|). 1 = perfect overlap. Both empty counts as perfect."""
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else float(2 * (pred & gt).sum() / total)


def iou(pred, gt):
    """Jaccard / IoU = |A n B| / |A u B|."""
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float((pred & gt).sum() / union)


if __name__ == "__main__":
    # Self-check on a synthetic abdomen: body ellipse, spine at the back, two bright kidneys
    # beside it and a large liver-like organ at the front that the rule filter must reject.
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:256, :256]
    body = ((yy - 128) / 100) ** 2 + ((xx - 128) / 120) ** 2 < 1
    spine = (yy - 175) ** 2 + (xx - 128) ** 2 < 14 ** 2
    kid_l = ((yy - 165) / 22) ** 2 + ((xx - 80) / 14) ** 2 < 1
    kid_r = ((yy - 165) / 22) ** 2 + ((xx - 176) / 14) ** 2 < 1
    liver = ((yy - 95) / 30) ** 2 + ((xx - 100) / 45) ** 2 < 1
    hu = np.full((256, 256), -1000.0)
    hu[body] = 40
    hu[liver] = 120
    hu[kid_l | kid_r] = 200
    hu[spine] = 700
    hu += rng.normal(0, 15, hu.shape)
    gt = kid_l | kid_r
    img = pre.enhance(hu)
    for m in ("adaptive", "watershed"):
        score = dice(segment_kidney(img, hu, m), gt)
        print(f"{m}: synthetic Dice {score:.3f}")
        assert score > 0.8, m
    assert dice(gt, gt) == 1.0 and iou(gt, np.zeros_like(gt)) == 0.0
    print("kidney_segment self-check OK")

"""Phase 3 (KIDNEY member): features of candidate regions + SVM classifier "kidney vs not kidney".

Phase 2 produces candidate blobs. Some are kidneys; others are false positives with similar density
(liver edge, spleen, bowel loops, vessels). Each blob becomes a feature vector and an SVM decides
whether it is a kidney. Ground-truth label of a blob: kidney if >= 50% of it lies in the radiologist mask.

Features (17): 6 shape + 5 GLCM texture (averaged over 0, 45, 90, 135 degrees) + 4 first-order HU
               + 2 position (sideways distance from the spine, depth relative to the spine; normalised).
Classifier: Support Vector Machine with an RBF kernel (the tumor member uses a Random Forest).
Evaluation: 70/30 split BY PATIENT, 5-fold grouped cross-validation on the 70% to tune C and gamma.
"""
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import kidney_segment as seg

ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
LEVELS = 32                      # grey levels for the GLCM (level 0 is reserved for "outside the blob")
OVERLAP_FOR_KIDNEY = 0.5
FEATURE_NAMES = ["area", "perimeter_area_ratio", "circularity", "compactness", "eccentricity", "solidity",
                 "glcm_energy", "glcm_contrast", "glcm_homogeneity", "glcm_correlation", "glcm_entropy",
                 "hu_mean", "hu_variance", "hu_skewness", "hu_kurtosis",
                 "dist_from_spine_x", "depth_vs_spine"]
PARAM_GRID = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}


# ---------------------------------------------------------------- features
def shape_features(mask):
    """Area, perimeter/area, circularity (4*pi*A/P^2), compactness (P^2/(4*pi*A)), eccentricity, solidity."""
    r = regionprops(mask.astype(np.uint8))[0]
    p = r.perimeter + 1e-8
    return [r.area, p / r.area, 4 * np.pi * r.area / p ** 2, p ** 2 / (4 * np.pi * r.area),
            r.eccentricity, r.solidity]


def glcm_features(enhanced, mask):
    """GLCM energy, contrast, homogeneity, correlation and entropy, averaged over 4 angles.
    Only pixel pairs that are BOTH inside the blob are counted: pixels outside are set to level 0
    and row/column 0 of the matrix is dropped, so the surrounding tissue does not leak into the texture."""
    ys, xs = np.where(mask)
    crop = enhanced[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.int32)
    inside = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    q = np.where(inside, 1 + crop * (LEVELS - 1) // 256, 0).astype(np.uint8)
    glcm = graycomatrix(q, [1], ANGLES, levels=LEVELS, symmetric=True, normed=False)[1:, 1:].astype(np.float64)
    glcm /= glcm.sum(axis=(0, 1), keepdims=True) + 1e-12
    props = [graycoprops(glcm, p).mean() for p in ("energy", "contrast", "homogeneity", "correlation")]
    entropy = -np.mean(np.sum(glcm * np.log2(glcm + 1e-12), axis=(0, 1)))
    return props + [entropy]


def intensity_features(hu, mask):
    """First-order attenuation statistics inside the blob: mean HU, variance, skewness, kurtosis."""
    v = hu[mask].astype(np.float64)
    return [v.mean(), v.var(), skew(v), kurtosis(v)]


def position_features(mask, anatomy):
    """Kidneys sit right beside the spine; bowel, liver and spleen lie further out or further
    forward. Distances are divided by the body height so patients of different size compare."""
    cy, cx = regionprops(mask.astype(np.uint8))[0].centroid
    return [abs(cx - anatomy["spine_col"]) / anatomy["height"],
            (cy - anatomy["spine_row"]) / anatomy["height"]]


def extract_features(enhanced, hu, mask, anatomy=None):
    """17-value feature vector of one candidate blob."""
    anatomy = anatomy or seg.anatomy(hu)
    return np.array(shape_features(mask) + glcm_features(enhanced, mask)
                    + intensity_features(hu, mask) + position_features(mask, anatomy), dtype=np.float64)


def is_kidney(blob, gt_kidney):
    """Ground-truth label of a candidate: 1 if at least half of it lies inside the radiologist's kidney."""
    return int((blob & gt_kidney).sum() >= OVERLAP_FOR_KIDNEY * blob.sum())


# ---------------------------------------------------------------- dataset
def candidates_of_slice(enhanced, hu, method, params):
    """All raw candidate blobs of one slice (no rule filter: the classifier does the filtering)."""
    raw = seg.segment_kidney(enhanced, hu, method, rule=False, **params)
    return seg.candidates(raw)


def build_dataset(slices, method, params):
    """slices: list of dicts with hu, kidney, case, enhanced.
    Returns X (n, 17), y (n,), groups = case id per row, owner = slice index per row, blobs (list)."""
    X, y, groups, owner, blobs = [], [], [], [], []
    for i, s in enumerate(slices):
        anatomy = seg.anatomy(s["hu"])
        for blob in candidates_of_slice(s["enhanced"], s["hu"], method, params):
            X.append(extract_features(s["enhanced"], s["hu"], blob, anatomy))
            y.append(is_kidney(blob, s["kidney"]))
            groups.append(s["case"])
            owner.append(i)
            blobs.append(blob)
    return np.array(X), np.array(y), np.array(groups), np.array(owner), blobs


def split_cases(cases, test_fraction=0.3, seed=0):
    """70/30 split BY PATIENT: returns (train_cases, test_cases). No slice of a test patient is
    ever seen in training, otherwise near-identical neighbouring slices would inflate the scores."""
    cases = np.array(sorted(set(cases)))
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(test_fraction * len(cases))))
    test = sorted(int(c) for c in rng.choice(cases, n_test, replace=False))
    return [int(c) for c in cases if c not in test], test


# ---------------------------------------------------------------- classifier
def make_model():
    """Standardise features (SVMs are distance-based), then RBF-kernel SVM.
    class_weight='balanced' because non-kidney blobs usually outnumber kidney blobs."""
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))


def train_and_evaluate(X, y, groups, test_cases):
    """Tune C and gamma with 5-fold grouped CV on the training patients, refit, score on test patients."""
    test = np.isin(groups, test_cases)
    train = ~test
    folds = GroupKFold(n_splits=min(5, len(set(groups[train]))))
    # zero_division=0: a CV fold whose patients have no kidney blob scores 0 instead of warning
    search = GridSearchCV(make_model(), PARAM_GRID, cv=folds, scoring=make_scorer(f1_score, zero_division=0))
    search.fit(X[train], y[train], groups=groups[train])
    cv = cross_val_score(search.best_estimator_, X[train], y[train], groups=groups[train], cv=folds)
    model = search.best_estimator_
    pred = model.predict(X[test])
    tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[0, 1]).ravel()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "accuracy": float((tp + tn) / max(len(pred), 1)),
        "recall": float(recall),                       # sensitivity
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(precision),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-8)),
        "cv_accuracy_mean": float(cv.mean()), "cv_accuracy_std": float(cv.std()),
        "best_params": {k.replace("svc__", ""): v for k, v in search.best_params_.items()},
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "n_kidney_train": int(y[train].sum()), "n_kidney_test": int(y[test].sum()),
        "test_cases": [int(c) for c in test_cases],
        "model": model, "test_pred": pred,
    }


if __name__ == "__main__":
    # Self-check 1: separable synthetic features must be classified well.
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 60)
    X = rng.normal(0, 1, (120, len(FEATURE_NAMES))) + y[:, None] * 2
    groups = np.tile(np.arange(12), 10)
    train_cases, test_cases = split_cases(groups)
    assert len(test_cases) == 4 and not set(train_cases) & set(test_cases)
    assert train_and_evaluate(X, y, groups, test_cases)["accuracy"] > 0.9
    # Self-check 2: feature vector of a toy disk has the right length and sane circularity.
    yy, xx = np.mgrid[:64, :64]
    blob = (yy - 32) ** 2 + (xx - 32) ** 2 < 15 ** 2
    img = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    f = extract_features(img, img.astype(float), blob, anatomy={"spine_row": 48, "spine_col": 32, "height": 64})
    assert f.shape == (len(FEATURE_NAMES),) and 0.8 < f[2] <= 1.1
    print("kidney_classify self-check OK")

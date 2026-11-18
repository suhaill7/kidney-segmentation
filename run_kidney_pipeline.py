"""End-to-end KIDNEY pipeline on all prepared slices. Writes every figure / table for the report to results/.

Run (after prepare_kidney_data.py):  python run_kidney_pipeline.py

Order:
  0. split patients 70/30 (by case) - every choice below is made on TRAINING patients only
  1. anatomical window learned from training ground truth; Phase 1 Butterworth D0 and order n chosen by
     training Dice (Gaussian low-pass swept alongside for the comparison table only)
  2. Phase 2: for every contrast step x strategy, sweep the parameters (training patients);
     each strategy keeps its best contrast step + parameters; then Dice / IoU on all slices, overlays
     (Phase 1 PSNR / CNR tables and figures are made once the final chain is known)
  3. Phase 3: candidate features -> SVM kidney vs not-kidney, confusion matrix, final masks
"""
import json
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import diamond, disk

import kidney_classify as clf
import kidney_preprocess as pre
import kidney_segment as seg

ROOT = Path(__file__).resolve().parent
SLICES = ROOT / "data" / "slices"
RES = ROOT / "results"
TEST_IMAGES = ROOT / "test_images"
PARAMS_FILE = RES / "chosen_params.json"
MODEL_FILE = RES / "kidney_svm.joblib"
METHODS = ("adaptive", "watershed")
SWEEP_GRID = {                     # bone_hu first: it decides what the later parameters work on
    "adaptive": {"bone_hu": [250, 300, 350, 400], "block_size": [51, 101, 151, 201], "offset": [0, -5, -10, -20],
                 "min_pct": [40, 50, 60, 70, 80], "open_r": [0, 1, 2, 3], "close_r": [0, 2, 3, 5]},
    "watershed": {"bone_hu": [250, 300, 350, 400], "high_pct": [80, 85, 90, 94, 97], "low_pct": [40, 55, 70],
                  "open_r": [0, 1, 2, 3], "close_r": [0, 2, 3, 5]},
}
CONTRAST_OPTIONS = [None, "clahe", "he", "gamma"]
CUTOFF_OPTIONS = [15, 20, 30, 40, 60, 80, 120]   # D0 values tried for both low-pass filters
ORDER_OPTIONS = [1, 2, 3]                       # Butterworth order n


# ---------------------------------------------------------------- data
def load_slices():
    """All prepared slices as dicts (hu, kidney, tumor, case, z, name)."""
    files = sorted(SLICES.glob("case_*.npz"))
    if not files:
        raise SystemExit("No slices found: run  python prepare_kidney_data.py  first.")
    out = []
    for f in files:
        d = np.load(f)
        out.append({"hu": d["hu"], "kidney": d["kidney"].astype(bool), "tumor": d["tumor"].astype(bool),
                    "case": int(d["case"]), "z": int(d["z"]), "name": f.stem})
    return out


def cache_enhanced(slices, cutoff, order):
    """Phase 1 output of every slice for every contrast option (computed once, reused by the sweep)."""
    for s in slices:
        s["enh"] = {c: pre.enhance(s["hu"], c, cutoff, order) for c in CONTRAST_OPTIONS}


def use_contrast(slices, contrast):
    """Make the Phase 1 output with this contrast step the one that Phase 2 / 3 work on."""
    for s in slices:
        s["enhanced"] = s["enh"][contrast]


WINDOW = dict(seg.DEFAULT_WINDOW)       # replaced by the window learned from training ground truth


def slice_dice(s, method, params, window):
    return seg.dice(seg.segment_kidney(s["enhanced"], s["hu"], method, window=window, **params), s["kidney"])


def mean_dice(slices, method, params):
    """Mean Dice of one strategy (with the anatomical rule filter) over some slices, on all CPU cores."""
    return float(np.mean(Parallel(n_jobs=-1)(delayed(slice_dice)(
        {"enhanced": s["enhanced"], "hu": s["hu"], "kidney": s["kidney"]}, method, params, WINDOW)
        for s in slices)))


# ---------------------------------------------------------------- Phase 1
def phase1_variants(hu, contrast=None, cutoff=pre.LOWPASS_CUTOFF, order=pre.LOWPASS_ORDER):
    """Every enhancement step applied to the windowed slice, for side-by-side comparison."""
    base = pre.window_normalize(hu)
    return {
        "Window [-100,300]": base,
        "Gaussian": pre.denoise(base, "gaussian"),
        "Bilateral": pre.denoise(base, "bilateral"),
        "CLAHE": pre.enhance_contrast(base, "clahe"),
        "Histogram eq.": pre.enhance_contrast(base, "he"),
        "Gamma 0.8": pre.enhance_contrast(base, "gamma"),
        f"Butterworth D0={cutoff} n={order}": pre.butterworth_lowpass(base, cutoff, order),
        f"Gaussian LP D0={cutoff} (compared)": pre.gaussian_lowpass(base, cutoff),
        "Final chain": pre.enhance(hu, contrast, cutoff, order),
    }


def phase1_metrics(slices, contrast, cutoff, order):
    """Mean PSNR (vs the windowed slice) and CNR (kidney vs a 10-px ring around it) per step."""
    rows = []
    for s in slices:
        body = pre.body_mask(s["hu"]) & ~pre.bone_mask(s["hu"])
        ring = pre.surrounding_ring(s["kidney"], body)
        variants = phase1_variants(s["hu"], contrast, cutoff, order)
        base = variants["Window [-100,300]"]
        for name, img in variants.items():
            rows.append({"step": name, "psnr": np.nan if img is base else pre.psnr(base, img),
                         "cnr": pre.cnr(img, s["kidney"], ring)})
    table = pd.DataFrame(rows).groupby("step", sort=False).mean().round(2)
    table.to_csv(RES / "phase1_psnr_cnr.csv")
    return table


def frequency_figure(s, cutoff, order):
    """Brief Phase 1.3: filter shapes H(u,v) (Butterworth n = 1, 2, 3 vs Gaussian, same D0), the
    log-magnitude spectrum before / after each filter, and the filtered slices."""
    base = pre.window_normalize(s["hu"])
    bw, ga = pre.butterworth_lowpass(base, cutoff, order), pre.gaussian_lowpass(base, cutoff)
    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_subplot(2, 4, 1)
    d = np.arange(0, 4 * cutoff)
    for n in ORDER_OPTIONS:
        ax.plot(d, 1 / (1 + (d / cutoff) ** (2 * n)), label=f"Butterworth n={n}" + (" (chosen)" if n == order else ""))
    ax.plot(d, np.exp(-d ** 2 / (2 * cutoff ** 2)), "k--", label="Gaussian (compared)")
    ax.axvline(cutoff, color="0.6", lw=0.8)
    ax.set_xlabel("D(u,v)  (distance from spectrum centre)")
    ax.set_ylabel("H")
    ax.set_title(f"Filter profiles, D0 = {cutoff}")
    ax.legend(fontsize=8)
    c = base.shape[0] // 2
    half = min(4 * cutoff, c)
    zoom = slice(c - half, c + half)
    panels = [(f"H Butterworth D0={cutoff} n={order}", pre.butterworth_transfer(base.shape, cutoff, order)[zoom, zoom], None),
              (f"H Gaussian D0={cutoff}", pre.gaussian_transfer(base.shape, cutoff)[zoom, zoom], None),
              ("Windowed slice", base, (0, 255)),
              ("log|DFT| before", pre.log_spectrum(base), (0, 255)),
              ("log|DFT| after Butterworth", pre.log_spectrum(bw), (0, 255)),
              ("log|DFT| after Gaussian", pre.log_spectrum(ga), (0, 255)),
              ("After Butterworth", bw, (0, 255))]
    for k, (title, img, lim) in enumerate(panels, start=2):
        ax = fig.add_subplot(2, 4, k)
        ax.imshow(img, cmap="gray", **({"vmin": lim[0], "vmax": lim[1]} if lim else {}))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RES / "phase1_frequency.png", dpi=100)
    plt.close(fig)


def phase1_figures(s, contrast, cutoff, order):
    """Comparison grid, DFT spectra, histograms and an edge-profile figure for one example slice."""
    variants = phase1_variants(s["hu"], contrast, cutoff, order)
    base = variants["Window [-100,300]"]
    frequency_figure(s, cutoff, order)
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    panels = list(variants.items()) + [("|DFT| before", pre.log_spectrum(base)),
                                       ("|DFT| after Butterworth", pre.log_spectrum(pre.butterworth_lowpass(base, cutoff, order)))]
    for ax, (name, img) in zip(axes.flat, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Phase 1 enhancement steps ({s['name']})")
    fig.tight_layout()
    fig.savefig(RES / "phase1_comparison.png", dpi=110)
    plt.close(fig)

    body = pre.body_mask(s["hu"])
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for col, name in enumerate(["Window [-100,300]", "CLAHE", "Histogram eq.", "Gamma 0.8"]):
        values = variants[name][body]
        axes[0, col].hist(values, bins=64, color="0.3")
        axes[0, col].set_title(f"{name}: histogram (body)")
        axes[1, col].plot(np.cumsum(np.bincount(values, minlength=256)) / values.size)
        axes[1, col].set_title("cumulative")
    fig.tight_layout()
    fig.savefig(RES / "phase1_histograms.png", dpi=110)
    plt.close(fig)

    # Edge figure: zoom on the kidney and a horizontal intensity profile through its centre.
    rows, cols = np.where(s["kidney"])
    r0, c0 = int(rows.mean()), int(cols.mean())
    y0, y1, x0, x1 = max(r0 - 50, 0), r0 + 50, max(c0 - 70, 0), c0 + 70
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, name in zip(axes[:3], ["Window [-100,300]", "Gaussian", "Bilateral"]):
        ax.imshow(variants[name][y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        ax.axhline(r0 - y0, color="yellow", lw=0.8)
        ax.contour(s["kidney"][y0:y1, x0:x1], [0.5], colors="lime", linewidths=0.6)
        ax.set_title(name)
        ax.axis("off")
    for name in ["Window [-100,300]", "Gaussian", "Bilateral"]:
        axes[3].plot(variants[name][r0, x0:x1], label=name)
    axes[3].set_title("Profile along the yellow line")
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "phase1_edges.png", dpi=110)
    plt.close(fig)


def choose_lowpass(train):
    """Frequency-filter sweep on TRAINING slices (no contrast step, default Phase 2 settings):
    Butterworth for every order n and cut-off D0 -> the pair with the best mean training Dice is used.
    The Gaussian low-pass is swept over the same D0 values for the comparison table only.
    CNR is reported but is not the criterion: CNR against a ring right next to the kidney keeps
    rising as smoothing gets weaker, even though the noisier image segments worse."""
    rings = [pre.surrounding_ring(s["kidney"], pre.body_mask(s["hu"]) & ~pre.bone_mask(s["hu"])) for s in train]
    configs = [("butterworth", n) for n in ORDER_OPTIONS] + [("gaussian", None)]
    rows = []
    for name, n in configs:
        for d0 in CUTOFF_OPTIONS:
            for s in train:
                s["enhanced"] = pre.enhance(s["hu"], None, d0, n or 1, lowpass=name)
            dice = {m: mean_dice(train, m, seg.DEFAULTS[m]) for m in METHODS}
            rows.append({"filter": name, "order_n": n if n else "-", "D0": d0,
                         "train_cnr": round(float(np.mean([pre.cnr(s["enhanced"], s["kidney"], r)
                                                           for s, r in zip(train, rings)])), 3),
                         **{f"train_dice_{m}": round(d, 4) for m, d in dice.items()},
                         "train_dice_mean": round(float(np.mean(list(dice.values()))), 4)})
            print(f"  {name:11s} n={n or '-'}  D0={d0:3d}: Dice {rows[-1]['train_dice_mean']:.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(RES / "phase1_frequency_comparison.csv", index=False)
    bw = table[table["filter"] == "butterworth"]
    best = bw.loc[bw["train_dice_mean"].idxmax()]
    gauss = table[table["filter"] == "gaussian"]
    g_best = gauss.loc[gauss["train_dice_mean"].idxmax()]
    print(f"  best Gaussian (comparison only): D0={int(g_best['D0'])}, Dice {g_best['train_dice_mean']:.3f}")
    return int(best["D0"]), int(best["order_n"])


# ---------------------------------------------------------------- Phase 2
def sweep_method(train, m, contrast):
    """One parameter at a time, in order, each time keeping the value with the best training Dice."""
    params, rows = dict(seg.DEFAULTS[m]), []
    score = mean_dice(train, m, params)
    for name, grid in SWEEP_GRID[m].items():
        values = sorted(set(grid) | {params[name]})    # current value always included: never gets worse
        scores = [mean_dice(train, m, {**params, name: v}) for v in values]
        rows += [{"method": m, "contrast": contrast or "none", "param": name, "value": v, "train_dice": round(d, 4)}
                 for v, d in zip(values, scores)]
        params[name], score = values[int(np.argmax(scores))], max(scores)
    return params, rows, score


def phase2_sweep(train):
    """For each strategy: try every contrast step, tune the parameters for each, keep the best pair.
    (Choosing the contrast step with untuned parameters would favour whichever step happens to suit
    the default thresholds.)"""
    chosen, contrast_of, rows, check = {}, {}, [], []
    for m in METHODS:
        best = (-1.0, None, None)
        for c in CONTRAST_OPTIONS:
            use_contrast(train, c)
            default = mean_dice(train, m, seg.DEFAULTS[m])
            params, r, score = sweep_method(train, m, c)
            rows += r
            check.append({"method": m, "contrast": c or "none", "dice_default": round(default, 4),
                          "dice_tuned": round(score, 4)})
            print(f"  {m:9s} contrast {c or 'none':6s}: default Dice {default:.3f} -> tuned {score:.3f}")
            if score > best[0]:
                best = (score, c, params)
        chosen[m], contrast_of[m] = best[2], best[1]
        print(f"  -> {m}: contrast {best[1] or 'none'}, {best[2]}")
    pd.DataFrame(check).to_csv(RES / "phase1_chain_check.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(RES / "phase2_sweep.csv", index=False)
    sweep_figure(table, chosen, contrast_of)
    return chosen, contrast_of


def sweep_figure(table, chosen, contrast_of):
    """One small plot per swept parameter (for each strategy's chosen contrast step);
    the chosen value is marked with a red star."""
    keep = np.zeros(len(table), bool)
    for m in METHODS:
        keep |= (table["method"] == m) & (table["contrast"] == (contrast_of[m] or "none"))
    groups = list(table[keep].groupby(["method", "param"], sort=False))
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, ((m, p), g) in zip(axes.flat, groups):
        ax.plot(g["value"], g["train_dice"], "o-")
        star = g[g["value"] == chosen[m][p]]
        ax.plot(star["value"], star["train_dice"], "r*", ms=14)
        ax.set_title(f"{m} ({contrast_of[m] or 'no contrast'}): {p}", fontsize=9)
        ax.set_ylabel("train Dice")
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RES / "phase2_sweep.png", dpi=100)
    plt.close(fig)


def phase2_evaluate(slices, chosen, contrast_of, test_cases, window):
    """Dice / IoU of both strategies on every slice; summary for training and test patients."""
    for m in METHODS:                                  # each strategy on its own best Phase 1 output
        for s in slices:
            s[f"enh_{m}"] = s["enh"][contrast_of[m]]
    rows = []
    for s in slices:
        row = {"slice": s["name"], "case": s["case"], "split": "test" if s["case"] in test_cases else "train"}
        for m in METHODS:
            pred = seg.segment_kidney(s[f"enh_{m}"], s["hu"], m, window=window, **chosen[m])
            s[f"pred_{m}"] = pred
            row[f"dice_{m}"], row[f"iou_{m}"] = seg.dice(pred, s["kidney"]), seg.iou(pred, s["kidney"])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RES / "phase2_per_slice.csv", index=False)
    summary = table.groupby("split")[[c for c in table.columns if c.startswith(("dice", "iou"))]].mean().round(3)
    summary.to_csv(RES / "phase2_summary.csv")
    return table, summary


def overlay(ax, s, pred, title):
    """CT slice with PREDICTED kidney in red and GROUND TRUTH in green (brief 7.2)."""
    ax.imshow(pre.window_normalize(s["hu"]), cmap="gray")
    ax.contour(s["kidney"], [0.5], colors="lime", linewidths=0.8)
    if pred.any():
        ax.contour(pred, [0.5], colors="red", linewidths=0.8)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def spread(slices, n=8):
    """Up to n slices spread over as many patients as possible (round-robin over cases)."""
    by_case = {}
    for s in slices:
        by_case.setdefault(s["case"], []).append(s)
    picked, i = [], 0
    while len(picked) < min(n, len(slices)):
        for group in by_case.values():
            if i < len(group) and len(picked) < n:
                picked.append(group[i])
        i += 1
    return picked


def overlay_grid(slices, key, path, title, n=8):
    """Grid of overlays (red = predicted, green = ground truth)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, s in zip(axes.flat, slices[:n]):
        overlay(ax, s, s[key], f"{s['name']}  Dice {seg.dice(s[key], s['kidney']):.2f}")
    for ax in axes.flat[len(slices[:n]):]:
        ax.axis("off")
    fig.suptitle(title + "   (red = predicted, green = ground truth)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def morphology_figure(s, method, params, open_r=2, close_r=4):
    """Demonstration for the report: raw strategy output -> opening -> closing -> holes filled ->
    small objects removed + rule filter, zoomed on the body, plus the structuring elements.
    Fixed demo radii are used so every step visibly changes something."""
    region = seg.search_region(s["hu"], params["bone_hu"])
    p = {k: v for k, v in params.items() if k != "bone_hu"} | {"open_r": 0, "close_r": 0}
    raw = seg.segment_adaptive(s["enhanced"], region, **p) if method == "adaptive" else \
        seg.segment_watershed(s["enhanced"], region, **p)
    opened = ndi.binary_opening(raw, structure=diamond(open_r))
    closed = ndi.binary_closing(opened, structure=disk(close_r))
    filled = ndi.binary_fill_holes(closed)
    final = seg.rule_filter(seg.remove_small_objects(filled, seg.MIN_KIDNEY_PIXELS), s["hu"], WINDOW)
    steps = [(f"raw {method}", raw), (f"opening, diamond r={open_r}", opened),
             (f"closing, disk r={close_r}", closed), ("holes filled", filled), ("small removed + rule filter", final)]
    rows, cols = np.where(pre.body_mask(s["hu"]))
    crop = (slice(rows.min(), rows.max()), slice(cols.min(), cols.max()))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, (name, m) in zip(axes[0], steps):
        ax.imshow(m[crop], cmap="gray")
        ax.contour(s["kidney"][crop], [0.5], colors="lime", linewidths=0.7)
        ax.set_title(f"{name}\nDice {seg.dice(m, s['kidney']):.2f}", fontsize=9)
        ax.axis("off")
    for ax, (name, se) in zip(axes[1], seg.structuring_elements(open_r, close_r).items()):
        ax.imshow(se, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[1, 3:]:
        ax.axis("off")
    fig.suptitle("Morphological post-processing (green = ground truth)")
    fig.tight_layout()
    fig.savefig(RES / "morphology_demo.png", dpi=100)
    plt.close(fig)


def candidates_figure(slices, method, params, path):
    """For failure analysis: every raw candidate blob (yellow), the one(s) the rule kept (red),
    the ground truth (green) and the detected spine centre (cyan cross)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, s in zip(axes.flat, slices):
        raw = seg.segment_kidney(s[f"enh_{method}"], s["hu"], method, rule=False, **params)
        kept = seg.rule_filter(raw, s["hu"], WINDOW)
        a = seg.anatomy(s["hu"])
        ax.imshow(pre.window_normalize(s["hu"]), cmap="gray")
        for m, colour in ((raw, "yellow"), (s["kidney"], "lime"), (kept, "red")):
            if m.any():
                ax.contour(m, [0.5], colors=colour, linewidths=0.8)
        ax.plot(a["spine_col"], a["spine_row"], "c+", ms=14, mew=2)
        ax.set_title(f"{s['name']}  {len(seg.candidates(raw))} candidates", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(slices):]:
        ax.axis("off")
    fig.suptitle(f"{method}: yellow = all candidates, red = kept by rule, green = ground truth, cyan + = spine")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- Phase 3
def phase3(slices, method, params, test_cases, contrast, lowpass):
    """Kidney vs not-kidney SVM on candidate blobs; final kidney mask = blobs the SVM accepts."""
    X, y, groups, owner, blobs = clf.build_dataset(slices, method, params)
    print(f"  {len(y)} candidate blobs, {int(y.sum())} kidney / {int((1 - y).sum())} not kidney")
    result = clf.train_and_evaluate(X, y, groups, test_cases)
    joblib.dump({"model": result["model"], "method": method, "params": params,
                 "contrast": contrast, "lowpass_D0_n": lowpass}, MODEL_FILE)

    table = pd.DataFrame(X, columns=clf.FEATURE_NAMES)
    table.insert(0, "label", np.where(y == 1, "kidney", "not_kidney"))
    table.insert(0, "case", groups)
    table.insert(1, "slice", [slices[i]["name"] for i in owner])
    table.round(4).to_csv(RES / "feature_table.csv", index=False)
    table.drop(columns=["case", "slice"]).groupby("label").agg(["mean", "std"]).round(4).T.to_csv(
        RES / "feature_means_by_class.csv")
    glcm_cols = ["label"] + [c for c in clf.FEATURE_NAMES if c.startswith("glcm")]
    table[glcm_cols].groupby("label").agg(["mean", "std"]).round(4).T.to_csv(RES / "glcm_feature_table.csv")

    # final masks of the test slices: union of the blobs the classifier calls "kidney"
    test_rows = np.where(np.isin(groups, test_cases))[0]
    for s in slices:
        s["pred_svm"] = np.zeros_like(s["kidney"])
    for row, keep in zip(test_rows, result["test_pred"]):
        if keep:
            slices[owner[row]]["pred_svm"] |= blobs[row]
    confusion_figure(result["confusion_matrix"])
    return result


def confusion_figure(cm):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["not kidney", "kidney"])
    ax.set_yticks([0, 1], ["not kidney", "kidney"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("SVM on test patients")
    fig.tight_layout()
    fig.savefig(RES / "confusion_matrix.png", dpi=110)
    plt.close(fig)


def save_test_images(test):
    """Brief 7.1: all test files must be provided. Windowed CT + ground-truth mask as PNG."""
    TEST_IMAGES.mkdir(exist_ok=True)
    for s in test:
        plt.imsave(TEST_IMAGES / f"{s['name']}_ct.png", pre.window_normalize(s["hu"]), cmap="gray")
        plt.imsave(TEST_IMAGES / f"{s['name']}_kidney_gt.png", s["kidney"], cmap="gray")


# ---------------------------------------------------------------- main
def main():
    RES.mkdir(exist_ok=True)
    slices = load_slices()
    train_cases, test_cases = clf.split_cases([s["case"] for s in slices])
    train = [s for s in slices if s["case"] in train_cases]
    test = [s for s in slices if s["case"] in test_cases]
    print(f"{len(slices)} slices | train cases {train_cases} | test cases {test_cases}")

    global WINDOW
    WINDOW = seg.learn_window(train)
    print("Anatomical window learned from training ground truth:",
          {k: round(v, 3) for k, v in WINDOW.items()})
    print("Phase 1: Butterworth D0 x order sweep, Gaussian alongside for comparison (training Dice)")
    cutoff, order = choose_lowpass(train)
    print(f"  -> Butterworth D0 = {cutoff}, order n = {order}")
    cache_enhanced(slices, cutoff, order)

    print("Phase 2: contrast step x parameter sweep (training cases only, takes a while)")
    chosen, contrast_of = phase2_sweep(train)
    PARAMS_FILE.write_text(json.dumps({"cutoff": cutoff, "order": order, "contrast": contrast_of, "segment": chosen,
                                       "window": WINDOW, "test_cases": test_cases}, indent=2))
    per_slice, p2 = phase2_evaluate(slices, chosen, contrast_of, test_cases, WINDOW)
    print(p2.to_string())
    best = max(METHODS, key=lambda m: p2.loc["train", f"dice_{m}"])
    print(f"  -> best strategy on training cases: {best}")
    for m in METHODS:
        overlay_grid(spread(test), f"pred_{m}", RES / f"overlays_{m}.png", f"{m} + rule filter, test patients")
    worst = per_slice.nsmallest(8, f"dice_{best}")["slice"].tolist()
    worst_slices = [s for s in slices if s["name"] in worst]
    overlay_grid(worst_slices, f"pred_{best}", RES / "failures.png", f"8 worst slices ({best})")
    candidates_figure(worst_slices, best, chosen[best], RES / "failures_candidates.png")

    print(f"Phase 1: tables + figures (final chain = window -> bilateral -> "
          f"{contrast_of[best] or 'no contrast step'} -> Butterworth low-pass D0={cutoff}, n={order})")
    p1 = phase1_metrics(slices, contrast_of[best], cutoff, order)
    print(p1.to_string())
    phase1_figures(test[0], contrast_of[best], cutoff, order)
    use_contrast(slices, contrast_of[best])
    morphology_figure(test[0], best, chosen[best])

    print("Phase 3: SVM kidney vs not-kidney")
    p3 = phase3(slices, best, chosen[best], test_cases, contrast_of[best], (cutoff, order))
    rule_dice = float(np.mean([seg.dice(s[f"pred_{best}"], s["kidney"]) for s in test]))
    svm_dice = float(np.mean([seg.dice(s["pred_svm"], s["kidney"]) for s in test]))
    overlay_grid(spread(test), "pred_svm", RES / "overlays_svm.png", f"{best} + SVM filter, test patients")
    print(f"  accuracy {p3['accuracy']:.3f}  recall {p3['recall']:.3f}  specificity {p3['specificity']:.3f}  "
          f"F1 {p3['f1']:.3f}  CV {p3['cv_accuracy_mean']:.3f} +/- {p3['cv_accuracy_std']:.3f}")
    print(f"  test Dice: rule filter {rule_dice:.3f}  vs  SVM filter {svm_dice:.3f}")

    save_test_images(test)
    metrics = {
        "train_cases": train_cases, "test_cases": test_cases,
        "lowpass": {"filter": "butterworth", "D0": cutoff, "order_n": order},
        "contrast_step": contrast_of, "rule_window": WINDOW,
        "phase1": p1.reset_index().to_dict("records"),
        "phase2_params": chosen, "phase2_summary": p2.reset_index().to_dict("records"), "best_strategy": best,
        "phase3": {k: v for k, v in p3.items() if k not in ("model", "test_pred")},
        "test_dice_rule_filter": rule_dice, "test_dice_svm_filter": svm_dice,
    }
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"Done. Everything is in {RES}")


if __name__ == "__main__":
    main()

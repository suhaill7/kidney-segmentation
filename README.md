# Kidney Segmentation (KiTS23) — kidney member, CT036-3-3 IPPR

Classical computer-vision pipeline that finds the **kidneys** in abdominal CT slices. No deep learning, no pre-trained models.
This is the kidney half of the group project. The tumour half (Maneet) uses the same data format, so the kidney mask from this repo can feed his tumour stage.

```
Raw CT slice (HU)
  -> Phase 1  window [-100, 300] HU -> bilateral filter -> (contrast step chosen per strategy by tuned training Dice) -> Butterworth low-pass (D0 and order n by training Dice)
  -> Phase 2  adaptive local thresholding  |  marker-controlled watershed   + custom-SE morphology + anatomical rule filter
  -> Phase 3  17 features per candidate blob -> SVM (RBF) -> kidney / not kidney -> final kidney mask
```

## Algorithms, and how they differ from the tumour member's

| Step | Tumour member (Maneet) | This repo (kidney) |
|---|---|---|
| Window | [-160, 240] HU | [-100, 300] HU + bone mask from raw HU (threshold tuned, vertebra inside filled) |
| Edge-preserving filter | Anisotropic diffusion | **Bilateral filter** |
| Baseline smoothing | Median | **Gaussian** |
| Contrast | CLAHE vs HE | CLAHE vs HE vs **Gamma correction** |
| Frequency domain | Homomorphic filter (teammate) | **Butterworth low-pass** (Gaussian low-pass compared only) |
| Segmentation A | Multi-Otsu | **Adaptive local thresholding** |
| Segmentation B | Seeded region growing | **Marker-controlled watershed** |
| Morphology SEs | disk opening, ellipse closing | **hand-built cross** (markers), **diamond** opening, disk closing |
| Classifier | Random Forest, healthy vs tumour | **SVM (RBF)**, kidney vs not-kidney |
| Cases | 100–112 (no 106) | 0, 1, 2, … (first 12 usable, never 100–112) |

## Setup (macOS)

```bash
cd kidney-segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                 # 9 quick tests on synthetic data, should all pass
```

The app needs Tkinter. The python.org installer includes it. With Homebrew Python, run `brew install python-tk`.
Check with `python -c "import tkinter"`.

## Run

```bash
python prepare_kidney_data.py       # downloads 12 cases (labels + imaging), saves 72 slices to data/slices/
python run_kidney_pipeline.py       # all three phases; writes figures, tables and metrics to results/
python kidney_app.py                # prototype GUI
```

- `prepare_kidney_data.py` downloads one CT volume at a time (a few hundred MB each). It keeps only the 6 chosen slices and deletes the volume. Expect it to take a while on the first run.
- `run_kidney_pipeline.py` takes several minutes: it tunes every strategy once per contrast option (4 x 2 sweeps), using all CPU cores.
- In the app, click **Load folder…** and pick `data/slices`. The app reads `.npz` files because it needs the real HU values. The PNGs in `test_images/` are only copies of the test slices for submission (brief 7.1).
- **Link tumour repo…** in the app: pick Maneet's repo folder. His tumour segmentation then runs inside *your* predicted kidney mask, and the **All organs** tab shows both.

## Data contract (same as the tumour repo)

`data/slices/case_XXXXX_zNNN.npz`: `hu` float32 (H, W) raw Hounsfield units; `kidney` bool (labels 1+2+3, kidney including tumour and cyst); `tumor` bool (labels 2+3); `case` int; `z` int.

## Files

| File | Job |
|---|---|
| `prepare_kidney_data.py` | Download cases, pick 6 slices per case spread across the kidney, save `.npz` |
| `kidney_preprocess.py` | Phase 1: window, body / bone masks, bilateral vs Gaussian, CLAHE / HE / gamma, DFT spectrum, Butterworth low-pass (+ Gaussian low-pass for comparison), PSNR, CNR |
| `kidney_segment.py` | Phase 2: custom SEs, morphology, adaptive thresholding, marker-controlled watershed, rule filter, Dice, IoU |
| `kidney_classify.py` | Phase 3: shape / GLCM / HU / position features, patient-wise 70/30 split, SVM with grouped 5-fold CV |
| `run_kidney_pipeline.py` | Runs everything, makes all report figures and tables |
| `kidney_app.py` | Tkinter prototype (brief 7.2) |
| `test_kidney_pipeline.py` | Quick synthetic tests |

## What ends up in `results/`

| File | For the report |
|---|---|
| `data_montage.png` | One slice per case with the ground-truth kidney |
| `phase1_psnr_cnr.csv`, `phase1_comparison.png` | PSNR / CNR table; all Phase 1 steps side by side, DFT spectrum before / after low-pass |
| `phase1_histograms.png` | Histograms + cumulative histograms: window vs CLAHE vs HE vs gamma |
| `phase1_edges.png` | Zoomed kidney edge + intensity profile: Gaussian blurs the edge, bilateral keeps it |
| `phase1_frequency_comparison.csv` | Butterworth (n = 1, 2, 3) and Gaussian low-pass for every D0: training CNR and Dice. Best Butterworth goes in the chain; Gaussian is comparison only |
| `phase1_frequency.png` | Filter profiles H(D) for Butterworth n = 1, 2, 3 vs Gaussian, H(u,v) images, log-magnitude spectrum before / after each filter, filtered slice |
| `phase1_chain_check.csv` | Training Dice for each strategy x contrast step, before and after tuning |
| `phase2_sweep.csv/png` | Parameter sweep on training patients (red star = chosen) |
| `phase2_per_slice.csv`, `phase2_summary.csv` | Dice / IoU per slice and mean, train vs test, both strategies |
| `overlays_adaptive.png`, `overlays_watershed.png`, `overlays_svm.png` | Red = predicted, green = ground truth, test patients |
| `failures.png` | The 8 worst slices: material for the critical-analysis section |
| `failures_candidates.png` | Same slices: every candidate blob (yellow), what the rule kept (red), detected spine (cyan +) |
| `morphology_demo.png` | Each morphology step + the structuring elements |
| `feature_table.csv`, `feature_means_by_class.csv`, `glcm_feature_table.csv` | Feature values; mean ± std per class |
| `confusion_matrix.png`, `metrics.json` | Classifier results; every number in one file |
| `chosen_params.json`, `kidney_svm.joblib` | Tuned settings + trained SVM (the app loads these) |

## Method notes (for the report)

- **Patient-wise split.** 70/30 by case (8 training, 4 test patients). Every choice (rule window, Butterworth D0 and n, contrast step, all Phase 2 parameters, SVM `C` and `gamma`) is made on training patients only, and the test patients are touched once at the end.
- **Why bone is removed from raw HU.** After a soft-tissue window, bone and contrast-enhanced kidney both saturate near 255. The bone mask has to come from the raw HU before the window discards that difference.
- **Why the classifier is kidney vs not-kidney.** Both strategies return several bright blobs per slice: kidneys, but also liver edge, spleen, bowel and vessels with similar density. The rule filter (largest blob on each side of the spine inside a position/size window learned from the training ground truth) is the hand-made baseline. The SVM learns the same decision from features, and `metrics.json` compares the two (`test_dice_rule_filter` vs `test_dice_svm_filter`).
- **Why Butterworth.** The brief (section 6, Phase 1.3) asks for a Butterworth low-pass or a homomorphic filter; the homomorphic filter is a teammate's, so this part uses the Butterworth, H(u,v) = 1 / (1 + (D/D0)^(2n)). The order n is the extra freedom a Gaussian lacks: it sets how sharply the filter cuts at D0. The Gaussian low-pass is kept only as a comparison.
- **Relative thresholds.** Threshold levels are percentiles of each patient's own body grey levels, because how strongly the kidney brightens depends on scan timing after the contrast injection.
- **Masked GLCM.** Only pixel pairs that are both inside the blob are counted, so surrounding fat does not leak into the texture features.

## Known limitations (be honest about these in the report)

- **Ground truth includes tumour and cyst.** Dark tumours and cysts are not bright, so both strategies miss them. Hole filling only recovers them when they are fully enclosed by kidney tissue. Expect lower Dice on slices with big lesions.
- **Bone threshold is a single value per strategy (tuned 250–400 HU).** In a late (excretory) phase, contrast in the renal pelvis can exceed it and is then removed as "bone". A lower threshold also removes bowel full of oral contrast, but risks cutting bright kidney cortex. Check `failures.png`.
- **Anatomical variants break the spine-based rule.** A kidney lying in front of the spine and crossing the midline (case 5 looks like this, possibly a horseshoe or ectopic kidney) is rejected by the rule filter by design.
- **Pixel spacing is ignored.** Areas are in pixels, and spacing differs between patients, so size features are not in mm².
- **One-at-a-time parameter sweep.** Parameters are tuned in sequence, not jointly. A value chosen at the edge of its grid means a better value may lie outside it.
- **Small data.** 12 patients, 4 test patients. Treat test scores as estimates, not precise numbers.
- **Kidney touching liver or spleen at similar brightness** can merge into one blob, which the size rule or the SVM then rejects as a whole.

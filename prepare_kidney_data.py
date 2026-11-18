"""Prepare KiTS23 data for the KIDNEY part: download 12 cases and save 6 axial slices per case.

Output: data/slices/case_XXXXX_zNNN.npz with keys hu, kidney, tumor, case, z
        (same data contract as the tumor member's repo, so both parts can be merged later)
        results/data_montage.png (one example slice per case)
Run:    python prepare_kidney_data.py

Only the files we need are downloaded:
  - segmentation.nii.gz for each case: sparse git checkout of the official KiTS23 repo
    (a full clone of that repo is many GB, because it also stores every annotator's raw masks)
  - imaging.nii.gz for each case: from the KiTS23 imaging mirror on Hugging Face
"""
import subprocess
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "data" / "_kits23_tmp"          # sparse labels checkout + downloaded imaging (temporary)
OUT = ROOT / "data" / "slices"
FIRST_CASE, N_CASES = 0, 12                  # cases 0, 1, 2, ... until 12 usable cases are found
TAKEN_BY_OTHERS = set(range(100, 113))       # tumor member (Maneet) uses cases 100-112; never reuse them
SLICES_PER_CASE = 6
MIN_AREA_FRACTION = 0.25                     # only slices where the kidney is >= 25% of its largest cross-section
IMG_URL = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_{:05d}.nii.gz"
LABEL_REPO = "https://github.com/neheller/kits23"


def init_label_repo():
    """Blob-less, sparse clone of the KiTS23 repo: nothing is downloaded until a path is checked out."""
    if not TMP.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
                        LABEL_REPO, str(TMP)], check=True)
        subprocess.run(["git", "-C", str(TMP), "sparse-checkout", "init", "--no-cone"], check=True)


def get_case(case):
    """Download segmentation + imaging of one case; return (ct_path, seg_path)."""
    folder = TMP / "dataset" / f"case_{case:05d}"
    seg = folder / "segmentation.nii.gz"
    if not seg.exists():
        # add this case's segmentation file to the sparse checkout list; git then fetches just that file
        subprocess.run(["git", "-C", str(TMP), "sparse-checkout", "add",
                        f"/dataset/case_{case:05d}/segmentation.nii.gz"], check=True)
        subprocess.run(["git", "-C", str(TMP), "checkout", "-q", "HEAD"], check=True)
    if not seg.exists():
        raise FileNotFoundError(f"case {case} has no segmentation (not a KiTS23 training case)")
    ct = folder / "imaging.nii.gz"
    if not ct.exists():
        print(f"  downloading imaging for case {case} ...")
        urllib.request.urlretrieve(IMG_URL.format(case), ct)
    return ct, seg


def load_upright(path):
    """Load a NIfTI as RAS array (x, y, z), then make each axial slice upright: anterior at the top.
    Identical to the tumor member's loader, so slices from both parts line up."""
    arr = nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
    return np.flip(arr.transpose(1, 0, 2), axis=0)     # (H, W, Z)


def pick_slices(kidney_area):
    """6 slices spread evenly over the part of the volume where the kidney is reasonably large.
    Spreading them (upper pole, hilum, lower pole) gives varied shapes and realistic failure cases,
    instead of 6 near-identical slices through the widest part."""
    eligible = np.where(kidney_area >= MIN_AREA_FRACTION * kidney_area.max())[0]
    if len(eligible) < SLICES_PER_CASE:
        return []
    picks = np.linspace(0, len(eligible) - 1, SLICES_PER_CASE).round().astype(int)
    return [int(eligible[i]) for i in picks]


def process_case(case):
    """Save the chosen slices of one case; return their z list (empty if unusable)."""
    ct_path, seg_path = get_case(case)
    hu, seg = load_upright(ct_path), load_upright(seg_path).astype(np.uint8)  # nibabel applies slope/intercept -> HU
    assert hu.shape == seg.shape and hu.shape[:2] == (512, 512), f"unexpected shape {hu.shape}"
    zs = pick_slices((seg > 0).sum(axis=(0, 1)))       # kidney = labels 1, 2, 3 (kidney incl. tumor + cyst)
    for z in zs:
        np.savez_compressed(OUT / f"case_{case:05d}_z{z:03d}.npz", hu=hu[:, :, z],
                            kidney=seg[:, :, z] > 0, tumor=seg[:, :, z] >= 2, case=case, z=z)
    (ct_path).unlink(missing_ok=True)                  # imaging volumes are large; the slices are all we keep
    return zs


def montage(files):
    """One slice per case with the ground-truth kidney in green, saved to results/data_montage.png."""
    first = {}
    for f in files:
        first.setdefault(int(f.name[5:10]), f)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, (case, f) in zip(axes.flat, sorted(first.items())):
        d = np.load(f)
        ax.imshow(np.clip(d["hu"], -100, 300), cmap="gray")
        ax.contour(d["kidney"], [0.5], colors="lime", linewidths=0.7)
        ax.set_title(f"case {case}  z={int(d['z'])}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    (ROOT / "results").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "data_montage.png", dpi=110)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    init_label_repo()
    done, case = [], FIRST_CASE
    while len(done) < N_CASES:
        if case not in TAKEN_BY_OTHERS:
            print(f"case {case}")
            try:
                if process_case(case):
                    done.append(case)
            except Exception as e:                     # failed download, missing case etc: take the next number
                print(f"  case {case} skipped: {e}")
        case += 1
    files = sorted(OUT.glob("case_*.npz"))
    px = np.array([np.load(f)["kidney"].sum() for f in files])
    print(f"cases {done}: {len(files)} slices, kidney px min/median/max = {px.min()}/{int(np.median(px))}/{px.max()}")
    montage(files)


if __name__ == "__main__":
    main()

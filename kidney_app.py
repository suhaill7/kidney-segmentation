"""Tkinter prototype for the KIDNEY part (brief 7.2).

  1. Load one slice or a group of slices (.npz files from prepare_kidney_data.py, or a whole folder)
  2. Show the segmented mask of each strategy
  3. Overlay PREDICTED (red contour) and GROUND TRUTH (green contour): kidney view, and an
     "all organs" view that adds the tumor member's segmentation when his repo is linked
  4. Show Dice / IoU for segmentation and accuracy / F1 / recall / specificity for classification

Run:  python kidney_app.py      (run run_kidney_pipeline.py once first: it saves the tuned settings + SVM)
"""
import importlib.util
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import kidney_classify as clf
import kidney_preprocess as pre
import kidney_segment as seg

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
TABS = ["Phase 1: Enhancement", "Phase 2: Segmentation", "Phase 3: Classification", "All organs"]


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def load_slice(path):
    """Read one .npz slice (data contract shared with the tumor member's repo)."""
    d = np.load(path)
    return {"hu": d["hu"], "kidney": d["kidney"].astype(bool), "tumor": d["tumor"].astype(bool),
            "name": Path(path).stem}


def load_module(name, path):
    """Import a .py file under a new name (the tumor repo also has preprocess.py / segment.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class KidneyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kidney CT pipeline - classical CV (CT036-3-3 IPPR)")
        self.geometry("1400x900")
        self.settings = load_json(RES / "chosen_params.json") or {
            "cutoff": pre.LOWPASS_CUTOFF, "order": pre.LOWPASS_ORDER, "contrast": {"adaptive": None, "watershed": None}, "segment": seg.DEFAULTS}
        self.metrics = load_json(RES / "metrics.json")
        self.model = joblib.load(RES / "kidney_svm.joblib") if (RES / "kidney_svm.joblib").exists() else None
        self.tumor = None                               # (tumor_preprocess, tumor_segment) once linked
        self.slices, self.cache = [], {}
        self._build_ui()

    # ------------------------------------------------------------ layout
    def _build_ui(self):
        left = ttk.Frame(self, padding=6)
        left.pack(side="left", fill="y")
        ttk.Button(left, text="Load slice(s)...", command=self.load_files).pack(fill="x")
        ttk.Button(left, text="Load folder...", command=self.load_folder).pack(fill="x", pady=2)
        ttk.Button(left, text="Evaluate all loaded", command=self.evaluate_all).pack(fill="x")
        ttk.Button(left, text="Link tumour repo...", command=self.link_tumor).pack(fill="x", pady=2)
        self.listbox = tk.Listbox(left, width=28, height=30)
        self.listbox.pack(fill="y", expand=True, pady=6)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.show())
        self.info = tk.Text(left, width=34, height=18, font=("Courier", 10))
        self.info.pack(fill="x")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="right", fill="both", expand=True)
        self.figures = {}
        for name in TABS:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            fig = Figure(figsize=(11, 8))
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figures[name] = (fig, canvas)
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.show())
        self.write_info(self.classifier_text())

    def write_info(self, text):
        self.info.delete("1.0", "end")
        self.info.insert("end", text)

    # ------------------------------------------------------------ loading
    def load_files(self):
        self.add_slices(filedialog.askopenfilenames(filetypes=[("CT slices", "*.npz")]))

    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.add_slices(sorted(str(p) for p in Path(folder).glob("*.npz")))

    def add_slices(self, paths):
        for p in paths:
            self.slices.append(load_slice(p))
            self.listbox.insert("end", Path(p).stem)
        if self.slices and not self.listbox.curselection():
            self.listbox.selection_set(0)
            self.show()

    def link_tumor(self):
        """Point at the tumor member's repo folder: his preprocess.py + segment.py are used for the tumor."""
        folder = filedialog.askdirectory(title="Folder of the tumour repo (contains segment.py)")
        if not folder:
            return
        try:
            self.tumor = (load_module("tumor_preprocess", Path(folder) / "preprocess.py"),
                          load_module("tumor_segment", Path(folder) / "segment.py"))
            self.cache.clear()
            self.show()
        except Exception as e:
            messagebox.showerror("Could not load tumour code", str(e))

    # ------------------------------------------------------------ processing
    def process(self, i):
        """Run the whole kidney pipeline on slice i (cached)."""
        if i in self.cache:
            return self.cache[i]
        s, cfg = self.slices[i], self.settings
        out = {}
        for m in ("adaptive", "watershed"):           # each strategy on its own best Phase 1 output
            out[f"enh_{m}"] = pre.enhance(s["hu"], cfg["contrast"][m], cfg["cutoff"], cfg.get("order", pre.LOWPASS_ORDER))
            out[m] = seg.segment_kidney(out[f"enh_{m}"], s["hu"], m, window=cfg.get("window"), **cfg["segment"][m])
        best = self.model["method"] if self.model else "adaptive"
        out["enhanced"] = out[f"enh_{best}"]
        out["accepted"], out["rejected"] = [], []
        if self.model:
            anatomy = seg.anatomy(s["hu"])
            for blob in clf.candidates_of_slice(out["enhanced"], s["hu"], best, self.model["params"]):
                x = clf.extract_features(out["enhanced"], s["hu"], blob, anatomy)
                keep = self.model["model"].predict(x[None])[0] == 1
                out["accepted" if keep else "rejected"].append(blob)
        if self.model:
            out["final"] = np.any(out["accepted"], axis=0) if out["accepted"] else np.zeros_like(s["kidney"])
        else:                                           # no SVM yet: fall back to the rule-filtered result
            out["final"] = out[best]
        if self.tumor:
            t_pre, t_seg = self.tumor
            out["tumor"] = t_seg.segment_tumor(t_pre.enhance(s["hu"]), out["final"], "region_growing")
        self.cache[i] = out
        return out

    def current(self):
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    # ------------------------------------------------------------ drawing
    def show(self):
        i = self.current()
        if i is None:
            return
        s, out = self.slices[i], self.process(i)
        tab = self.notebook.tab(self.notebook.select(), "text")
        fig, canvas = self.figures[tab]
        fig.clear()
        {TABS[0]: self.draw_phase1, TABS[1]: self.draw_phase2,
         TABS[2]: self.draw_phase3, TABS[3]: self.draw_all}[tab](fig, s, out)
        fig.tight_layout()
        canvas.draw()

    @staticmethod
    def overlay(ax, s, pred, gt, title):
        ax.imshow(pre.window_normalize(s["hu"]), cmap="gray")
        if gt.any():
            ax.contour(gt, [0.5], colors="lime", linewidths=0.9)
        if pred.any():
            ax.contour(pred, [0.5], colors="red", linewidths=0.9)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def draw_phase1(self, fig, s, out):
        base = pre.window_normalize(s["hu"])
        bilateral = pre.denoise(base, "bilateral")
        best = self.model["method"] if self.model else "adaptive"
        contrast, cutoff = self.settings["contrast"][best], self.settings["cutoff"]
        order = self.settings.get("order", pre.LOWPASS_ORDER)
        panels = [("Window [-100,300] HU", base), ("Bilateral filter", bilateral),
                  (f"Contrast: {contrast or 'none'}", pre.enhance_contrast(bilateral, contrast) if contrast else bilateral),
                  ("|DFT| (log)", pre.log_spectrum(base)), (f"Final (Butterworth D0={cutoff}, n={order})", out["enhanced"]),
                  ("Gaussian smoothing (compared)", pre.denoise(base, "gaussian"))]
        for k, (title, img) in enumerate(panels):
            ax = fig.add_subplot(2, 3, k + 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        ring = pre.surrounding_ring(s["kidney"], pre.body_mask(s["hu"]))
        self.write_info(f"{s['name']}\n\nPSNR final vs window: {pre.psnr(base, out['enhanced']):.2f} dB\n"
                        f"CNR window: {pre.cnr(base, s['kidney'], ring):.2f}\n"
                        f"CNR final : {pre.cnr(out['enhanced'], s['kidney'], ring):.2f}\n\n" + self.classifier_text())

    def draw_phase2(self, fig, s, out):
        text = [s["name"], ""]
        for k, m in enumerate(("adaptive", "watershed")):
            ax = fig.add_subplot(2, 2, k + 1)
            ax.imshow(out[m], cmap="gray")
            ax.set_title(f"{m} mask", fontsize=9)
            ax.axis("off")
            d, j = seg.dice(out[m], s["kidney"]), seg.iou(out[m], s["kidney"])
            self.overlay(fig.add_subplot(2, 2, k + 3), s, out[m], s["kidney"], f"{m}: Dice {d:.3f}  IoU {j:.3f}")
            text.append(f"{m:9s} Dice {d:.3f} IoU {j:.3f}")
        self.write_info("\n".join(text) + "\n\nred = predicted\ngreen = ground truth\n\n" + self.classifier_text())

    def draw_phase3(self, fig, s, out):
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(pre.window_normalize(s["hu"]), cmap="gray")
        for blob in out["rejected"]:
            ax.contour(blob, [0.5], colors="yellow", linewidths=0.8, linestyles="dashed")
        for blob in out["accepted"]:
            ax.contour(blob, [0.5], colors="red", linewidths=0.9)
        ax.set_title("Candidates: red = SVM says kidney, yellow = rejected", fontsize=9)
        ax.axis("off")
        d = seg.dice(out["final"], s["kidney"])
        self.overlay(fig.add_subplot(1, 2, 2), s, out["final"], s["kidney"], f"Final kidney mask: Dice {d:.3f}")
        note = "" if self.model else "\n(no SVM yet: run run_kidney_pipeline.py)"
        self.write_info(f"{s['name']}\nfinal Dice {d:.3f}\n{len(out['accepted'])} accepted, "
                        f"{len(out['rejected'])} rejected{note}\n\n" + self.classifier_text())

    def draw_all(self, fig, s, out):
        self.overlay(fig.add_subplot(1, 3, 1), s, out["final"], s["kidney"], "Kidney")
        ax_t, ax_all = fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)
        if "tumor" in out:
            self.overlay(ax_t, s, out["tumor"], s["tumor"], f"Tumour  Dice {seg.dice(out['tumor'], s['tumor']):.3f}")
        else:
            ax_t.text(0.5, 0.5, "Link the tumour repo\nto show the tumour", ha="center")
            ax_t.axis("off")
        self.overlay(ax_all, s, out["final"], s["kidney"], "All organs (tumour dashed)")
        if "tumor" in out:
            if s["tumor"].any():
                ax_all.contour(s["tumor"], [0.5], colors="lime", linewidths=0.9, linestyles="dashed")
            if out["tumor"].any():
                ax_all.contour(out["tumor"], [0.5], colors="red", linewidths=0.9, linestyles="dashed")

    # ------------------------------------------------------------ metrics
    def classifier_text(self):
        if not self.metrics:
            return "No metrics yet:\nrun run_kidney_pipeline.py"
        p3 = self.metrics["phase3"]
        return ("SVM (test patients)\n"
                f"accuracy    {p3['accuracy']:.3f}\nF1          {p3['f1']:.3f}\n"
                f"recall      {p3['recall']:.3f}\nspecificity {p3['specificity']:.3f}\n"
                f"test Dice   {self.metrics['test_dice_svm_filter']:.3f}")

    def evaluate_all(self):
        """Mean Dice / IoU over every loaded slice (the 'group of images' view)."""
        if not self.slices:
            return
        rows = []
        for i, s in enumerate(self.slices):
            out = self.process(i)
            rows.append([seg.dice(out[k], s["kidney"]) for k in ("adaptive", "watershed", "final")] +
                        [seg.iou(out["final"], s["kidney"])])
        mean = np.mean(rows, axis=0)
        self.write_info(f"{len(rows)} slices (mean)\n\nDice adaptive  {mean[0]:.3f}\nDice watershed {mean[1]:.3f}\n"
                        f"Dice final     {mean[2]:.3f}\nIoU final      {mean[3]:.3f}\n\n" + self.classifier_text())


if __name__ == "__main__":
    KidneyApp().mainloop()

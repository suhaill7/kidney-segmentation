"""Quick tests on synthetic data (no CT download needed).  Run:  python -m pytest -q"""
import numpy as np

import kidney_classify as clf
import kidney_preprocess as pre
import kidney_segment as seg


def synthetic_abdomen(seed=0):
    """Body ellipse, spine at the back, two bright kidneys beside it, a liver-like organ at the front."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:256, :256]
    body = ((yy - 128) / 100) ** 2 + ((xx - 128) / 120) ** 2 < 1
    spine = (yy - 175) ** 2 + (xx - 128) ** 2 < 14 ** 2
    kidneys = (((yy - 165) / 22) ** 2 + ((xx - 80) / 14) ** 2 < 1) | (((yy - 165) / 22) ** 2 + ((xx - 176) / 14) ** 2 < 1)
    liver = ((yy - 95) / 30) ** 2 + ((xx - 100) / 45) ** 2 < 1
    hu = np.full((256, 256), -1000.0)
    hu[body], hu[liver], hu[kidneys], hu[spine] = 40, 120, 200, 700
    return hu + rng.normal(0, 15, hu.shape), kidneys


def test_window_range():
    out = pre.window_normalize(np.array([[-1000.0, -100.0, 100.0, 300.0, 2000.0]]))
    assert out.min() == 0 and out.max() == 255 and out.dtype == np.uint8


def test_enhancement_raises_cnr():
    hu, kidneys = synthetic_abdomen()
    ring = pre.surrounding_ring(kidneys, pre.body_mask(hu))
    assert pre.cnr(pre.enhance(hu), kidneys, ring) > pre.cnr(pre.window_normalize(hu), kidneys, ring)


def test_bone_is_found_before_windowing():
    hu, _ = synthetic_abdomen()
    assert pre.bone_mask(hu)[175, 128] and not pre.bone_mask(hu)[165, 80]


def test_both_strategies_find_the_kidneys():
    hu, kidneys = synthetic_abdomen()
    img = pre.enhance(hu)
    for method in ("adaptive", "watershed"):
        assert seg.dice(seg.segment_kidney(img, hu, method), kidneys) > 0.8


def test_rule_filter_rejects_front_organ():
    hu, kidneys = synthetic_abdomen()
    liver = np.zeros_like(kidneys)
    liver[80:110, 70:130] = True
    assert seg.dice(seg.rule_filter(kidneys | liver, hu), kidneys) > 0.99


def test_metrics():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    b = np.zeros((10, 10), bool)
    b[:, :5] = True
    assert abs(seg.dice(a, b) - 0.5) < 1e-9 and abs(seg.iou(a, b) - 1 / 3) < 1e-9


def test_feature_vector_and_split():
    hu, kidneys = synthetic_abdomen()
    blob = seg.candidates(kidneys)[0]
    f = clf.extract_features(pre.enhance(hu), hu, blob)
    assert f.shape == (len(clf.FEATURE_NAMES),) and np.all(np.isfinite(f))
    train, test = clf.split_cases(range(12))
    assert len(test) == 4 and not set(train) & set(test)


def test_spongy_vertebra_inside_is_bone():
    """The inside of a vertebra (~200 HU, like kidney) must count as bone because the shell encloses it."""
    yy, xx = np.mgrid[:128, :128]
    r2 = (yy - 64) ** 2 + (xx - 64) ** 2
    hu = np.full((128, 128), 40.0)
    hu[r2 < 20 ** 2] = 200            # spongy bone
    hu[(r2 >= 17 ** 2) & (r2 < 20 ** 2)] = 600   # cortical shell
    assert pre.bone_mask(hu)[64, 64]

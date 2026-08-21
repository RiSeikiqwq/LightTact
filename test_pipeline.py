"""test_pipeline.py -- headless end-to-end verification of the LightTact
DContact pipeline (no camera, no display needed).

Runs with the MockCamera simulator:
  1. segment_contact unit tests (paper III-D 4-condition test)
  2. mock render sanity (dark background / bright contact / board dots)
  3. camera calibration end-to-end (artifacts + rectified-grid regularity)
  4. npy mapping reuse (rectification identical across instances)
  5. DContact runtime (IoU vs ground truth, contact ratio, in_contact)
  6. demo smoke test (subprocess, headless)

Run:  python test_pipeline.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import yaml  # noqa: E402

from camera_2 import segment_contact  # noqa: E402
from mock_camera import MockCamera  # noqa: E402
from calibration_2 import run_calibration, _sort_grid_points  # noqa: E402
from dcontact import DContact  # noqa: E402

# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    status = "PASS" if cond else "FAIL"
    print("[{:<4}] {}".format(status, name) + (": {}".format(detail) if detail else ""))
    return bool(cond)


def make_test_config():
    """Config with calibration output redirected to test_output/."""
    with open("shape_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["calibration_root_dir"] = "test_output"
    os.makedirs("test_output", exist_ok=True)
    cfg_path = os.path.join("test_output", "shape_config_test.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return cfg_path


# ---------------------------------------------------------------------------
# 1. segment_contact unit tests
# ---------------------------------------------------------------------------


def test_segment_contact():
    print("\n=== 1. segment_contact (paper III-D) ===")
    ref = np.zeros((10, 10, 3), np.uint8)

    # A: single channel +30 -> contact (condition 2: >=1 channel > t1=20)
    s = np.zeros((10, 10, 3), np.uint8); s[:, :, 0] = 30
    check("A: single channel +30 -> contact", segment_contact(s, ref)[0, 0] == 255)

    # B: two channels +25 -> contact (condition 2; mean 16.7 <= t0)
    s = np.zeros((10, 10, 3), np.uint8); s[:, :, 0] = 25; s[:, :, 1] = 25
    check("B: two channels +25 -> contact", segment_contact(s, ref)[0, 0] == 255)

    # C: all three +35 -> contact (condition 1: mean 35 > t0=25)
    s = np.full((10, 10, 3), 35, np.uint8)
    check("C: all three +35 -> contact", segment_contact(s, ref)[0, 0] == 255)

    # D: single channel +10 -> no contact (nothing exceeds)
    s = np.zeros((10, 10, 3), np.uint8); s[:, :, 0] = 10
    check("D: single channel +10 -> no contact", segment_contact(s, ref)[0, 0] == 0)

    # E: negative change (darker than ref) -> no contact
    r = np.full((10, 10, 3), 50, np.uint8)
    s = np.zeros((10, 10, 3), np.uint8)
    check("E: darker than ref -> no contact", segment_contact(s, r)[0, 0] == 0)

    # F: pure noise -> essentially no contact pixels
    rng = np.random.default_rng(1)
    ref_n = np.clip(rng.normal(0.5, 1.0, (100, 100, 3)), 0, 255).astype(np.uint8)
    s_n = np.clip(rng.normal(0.5, 1.0, (100, 100, 3)), 0, 255).astype(np.uint8)
    frac = (segment_contact(s_n, ref_n) > 0).mean()
    check("F: noise-only -> <0.1% contact", frac < 0.001, "frac={:.5f}".format(frac))


# ---------------------------------------------------------------------------
# 2. mock render sanity
# ---------------------------------------------------------------------------


def test_mock_render(cfg_path):
    print("\n=== 2. MockCamera rendering ===")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cam = MockCamera(cfg, calibrated=False, seed=7)

    bg = cam.get_raw_image()
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    check("2a: background near-black", float(gray.mean()) < 5.0,
          "mean={:.3f}".format(gray.mean()))
    check("2b: background noise std ~1", 0.2 < float(gray.std()) < 4.0,
          "std={:.3f}".format(gray.std()))

    gt = np.zeros((480, 640), np.uint8)
    gt[280:440, 180:460] = 255
    cam.set_contact_mask(gt)
    img = cam.get_raw_image()
    check("2c: contact region bright", float(img[gt > 0].mean()) > 60.0,
          "mean={:.1f}".format(img[gt > 0].mean()))
    check("2d: non-contact stays dark", float(img[gt == 0].mean()) < 5.0,
          "mean={:.3f}".format(img[gt == 0].mean()))

    cam.set_calibration_board(True)
    board = cam.get_raw_image()
    bmask = (cv2.cvtColor(board, cv2.COLOR_BGR2GRAY) > 50).astype(np.uint8)
    cnts, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    check("2e: board renders 25 dots", len(cnts) == 25, "found={}".format(len(cnts)))
    return cam


# ---------------------------------------------------------------------------
# 3. calibration end-to-end
# ---------------------------------------------------------------------------


def test_calibration(cfg_path):
    print("\n=== 3. Camera calibration end-to-end (mock) ===")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cam = MockCamera(cfg, calibrated=False, seed=11)
    cam.enable_auto_board()  # board pressed after the reference capture
    run_calibration(cfg_path, save_debug=True, headless=True, camera=cam)

    cal_dir = cam.camera_calibration_dir
    ok = True

    for fname in ("position_scale.npy", "row_index.npy", "col_index.npy",
                  "valid_mask_cropped.npy", "ref_full.png", "sample_full.png",
                  "contact_mask_full.png", "contact_mask_cropped.png",
                  "raw_trapezoid_debug.png", "rectified_trapezoid_debug.png",
                  "rectified_trapezoid_masked_debug.png"):
        ok &= check("3a: artifact {}".format(fname),
                    os.path.exists(os.path.join(cal_dir, fname)))

    pos = np.load(os.path.join(cal_dir, "position_scale.npy"))
    ok &= check("3b: position_scale shape (4,)", pos.shape == (4,), str(pos.shape))
    px_per_mm = 0.5 * (1.0 / pos[2] + 1.0 / pos[3])
    ok &= check("3c: px_per_mm in [10, 30]", 10.0 < px_per_mm < 30.0,
                "{:.2f}".format(px_per_mm))
    # grid center in cropped coords: ideal (320, 107.5)
    ok &= check("3d: grid center near ideal", abs(pos[0] - 107.5) < 15 and abs(pos[1] - 320) < 15,
                "center=({:.1f},{:.1f})".format(pos[0], pos[1]))

    ri = np.load(os.path.join(cal_dir, "row_index.npy"))
    ci = np.load(os.path.join(cal_dir, "col_index.npy"))
    ok &= check("3e: index maps int32", ri.dtype == np.int32 and ci.dtype == np.int32)
    ok &= check("3f: index maps same shape", ri.shape == ci.shape, str(ri.shape))
    H_out, W_out = ri.shape
    ok &= check("3g: rectified size sane", 150 <= H_out <= 350 and 130 <= W_out <= 320,
                "{}x{}".format(W_out, H_out))
    ok &= check("3h: indices in bounds",
                ri.min() >= 0 and ri.max() <= 214 and ci.min() >= 0 and ci.max() <= 639,
                "row[{},{}] col[{},{}]".format(ri.min(), ri.max(), ci.min(), ci.max()))

    vm = np.load(os.path.join(cal_dir, "valid_mask_cropped.npy"))
    ok &= check("3i: valid mask shape (215, 640)", vm.shape == (215, 640), str(vm.shape))

    # --- grid regularity in the rectified sample image ---
    sample_full = cv2.imread(os.path.join(cal_dir, "sample_full.png"))
    cam2 = MockCamera(cfg, calibrated=True, seed=11)
    rect = cam2.rectify_image(sample_full)
    bmask = (cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) > 50).astype(np.uint8)
    cnts, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] > 0:
            centers.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    ok &= check("3j: 25 dots in rectified image", len(centers) == 25, "found={}".format(len(centers)))
    if len(centers) == 25:
        grid = _sort_grid_points(np.array(centers, np.float32), 5, 5)  # (y, x)
        # row spacing (5 rows), col spacing (5 cols)
        row_sp = np.mean([np.linalg.norm(grid[i + 5] - grid[i]) for i in range(20)])
        col_sp = np.mean([np.linalg.norm(grid[r * 5 + c + 1] - grid[r * 5 + c])
                          for r in range(5) for c in range(4)])
        expected = 3.0 * px_per_mm
        ok &= check("3k: row spacing ~= 3mm*px_per_mm",
                    abs(row_sp - expected) < 2.5,
                    "measured={:.2f} expected={:.2f}".format(row_sp, expected))
        ok &= check("3l: col spacing ~= 3mm*px_per_mm",
                    abs(col_sp - expected) < 2.5,
                    "measured={:.2f} expected={:.2f}".format(col_sp, expected))
        row_straightness = np.mean([grid[r * 5:(r + 1) * 5, 0].std() for r in range(5)])
        col_straightness = np.mean([grid[c::5, 1].std() for c in range(5)])
        ok &= check("3m: grid straight (std < 3 px)",
                    row_straightness < 3.0 and col_straightness < 3.0,
                    "row_std={:.2f} col_std={:.2f}".format(row_straightness, col_straightness))
    return ok, rect, H_out, W_out


# ---------------------------------------------------------------------------
# 4. npy reuse
# ---------------------------------------------------------------------------


def test_npy_reuse(cfg_path, rect_expected, H_out, W_out):
    print("\n=== 4. npy mapping reuse ===")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cam = MockCamera(cfg, calibrated=True, seed=11)
    cal_dir = cam.camera_calibration_dir
    sample_full = cv2.imread(os.path.join(cal_dir, "sample_full.png"))
    rect2 = cam.rectify_image(sample_full)
    ok = check("4a: rectified identical to fresh calibration",
               np.array_equal(rect_expected, rect2))
    ok &= check("4b: rectified shape matches calibration",
                cam.rectified_shape == (H_out, W_out),
                "{} vs {}".format(cam.rectified_shape, (H_out, W_out)))
    return ok


# ---------------------------------------------------------------------------
# 5. DContact runtime
# ---------------------------------------------------------------------------


def test_dcontact_runtime(cfg_path, H_out, W_out):
    print("\n=== 5. DContact runtime (IoU vs ground truth) ===")
    dc = DContact(cfg_path=cfg_path, calibrated=True, backend="mock")
    ok = check("5a: calibrated flag", dc.calibrated)

    dc.capture_reference(interactive=False)
    dc.save_reference()

    gt = np.zeros((480, 640), np.uint8)
    gt[280:440, 180:460] = 255
    dc.cam.set_contact_mask(gt)
    raw = dc.cam.get_raw_image()
    mask = dc.detect(raw)

    inter = int(((mask > 0) & (gt > 0)).sum())
    union = int(((mask > 0) | (gt > 0)).sum())
    iou = inter / union if union else 0.0
    ok &= check("5b: segmentation IoU > 0.9", iou > 0.9, "IoU={:.4f}".format(iou))

    mask_rect, img_rect = dc.detect_rectified(raw)
    ok &= check("5c: rectified mask shape", mask_rect.shape == (H_out, W_out),
                str(mask_rect.shape))
    ok &= check("5d: rectified image shape", img_rect.shape == (H_out, W_out, 3),
                str(img_rect.shape))

    ratio = dc.contact_ratio(mask_rect)
    ok &= check("5e: contact_ratio in (0.05, 0.95)", 0.05 < ratio < 0.95,
                "{:.4f}".format(ratio))
    ok &= check("5f: in_contact() True", dc.in_contact(raw))

    dc.cam.set_contact_mask(np.zeros((480, 640), np.uint8))
    raw_empty = dc.cam.get_raw_image()
    ok &= check("5g: in_contact() False without contact", not dc.in_contact(raw_empty))
    return ok


# ---------------------------------------------------------------------------
# 6. demo smoke
# ---------------------------------------------------------------------------


def test_demo_smoke(cfg_path):
    print("\n=== 6. Demo smoke (headless subprocess) ===")
    res = subprocess.run(
        [sys.executable, "_2_DContact_Demo.py", "--camera", "mock",
         "--headless", "--frames", "20", "--config", cfg_path],
        capture_output=True, text=True, timeout=180, cwd=HERE)
    ok = check("6a: demo exit code 0", res.returncode == 0,
               "rc={}".format(res.returncode))
    ok &= check("6b: demo printed contact_ratio", "contact_ratio" in res.stdout)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    cfg_path = make_test_config()
    test_segment_contact()
    cam = test_mock_render(cfg_path)
    ok3, rect_expected, H_out, W_out = test_calibration(cfg_path)
    ok4 = test_npy_reuse(cfg_path, rect_expected, H_out, W_out)
    ok5 = test_dcontact_runtime(cfg_path, H_out, W_out)
    ok6 = test_demo_smoke(cfg_path)

    print("\n" + "=" * 60)
    failed = [n for n, ok in RESULTS if not ok]
    print("{} checks, {} failed".format(len(RESULTS), len(failed)))
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

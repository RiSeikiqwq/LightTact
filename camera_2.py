"""camera_2.py -- LightTact camera driver + contact segmentation.

Reconstruction of the author's missing `camera_2` module used by
`calibration_2.py` (``from camera_2 import Camera``), following:

* LightTact paper Sec. III-D: fixed exposure (20 ms), auto-exposure disabled,
  5x5 calibration grid, frame-differencing segmentation with the 4-condition
  brightness test (t0, t1, t2, t3) = (25, 20, 30, 40).
* 9DTact `shape_reconstruction/camera.py` for the calibration-path layout
  (calibration_root_dir/sensor_<id>/camera_calibration/...).

The calibration scripts save dense ``row_index.npy`` / ``col_index.npy`` maps
(rectified space -> ROI-cropped raw space); with ``calibrated=True`` they are
loaded here so rectification can be reused on every subsequent run without
re-calibration.
"""

import os
import platform
import numpy as np
import cv2
import yaml

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def imwrite_unicode(path, img):
    """Unicode-safe cv2.imwrite (cv2.imwrite fails silently on non-ASCII
    paths on Windows, and this project lives under a non-ASCII path)."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("cv2.imencode failed for {}".format(path))
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def segment_contact(sample, ref, thresholds=(25, 20, 30, 40)):
    """Paper III-D segmentation: contact if ANY of

      1) mean RGB increase > t0
      2) at least one channel exceeds t1
      3) at least two channels exceed t2
      4) all three channels exceed t3

    Returns a full-frame uint8 0/255 mask. Fully vectorized.
    """
    t0, t1, t2, t3 = thresholds
    diff = sample.astype(np.int16) - ref.astype(np.int16)  # signed, no uint8 wrap
    c1 = diff.mean(axis=2) > t0
    c2 = (diff > t1).any(axis=2)
    c3 = (diff > t2).sum(axis=2) >= 2
    c4 = (diff > t3).all(axis=2)
    return (c1 | c2 | c3 | c4).astype(np.uint8) * 255


def handle_preview_key(key, cam):
    """Exposure keys shared by all interactive preview loops.

    Returns True if the key was consumed (exposure handling), False otherwise
    ('y'/'q' and anything else are left to the caller).
    """
    if cam.headless or key < 0:
        return False
    if key in (ord("+"), ord("=")):
        cam.adjust_exposure(+5)
        return True
    if key in (ord("-"), ord("_")):
        cam.adjust_exposure(-5)
        return True
    if key in (ord("e"), ord("E")):
        cam.set_exposure(cam.default_exposure_ms)
        return True
    return False


def _is_ascii_path(path):
    try:
        path.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class Camera:
    """UVC camera driver for LightTact.

    Compatible with the interface expected by `calibration_2.py`:
    ``Camera(cfg, calibrated=False)``, ``get_raw_image()``,
    ``get_raw_avg_image()``, ``get_contact_area(sample, ref)`` and the
    attributes ``camera_calibration_dir``, ``position_scale_path``,
    ``row_index_path``, ``col_index_path``.
    """

    def __init__(self, cfg, calibrated=False, exposure_ms=None):
        self.cfg = cfg
        self.headless = bool(cfg.get("headless", False))

        camera_setting = cfg["camera_setting"]
        camera_calibration = cfg["camera_calibration"]

        self.camera_channel = int(camera_setting.get("camera_channel", 0))
        res = camera_setting.get("resolution", [640, 480])
        self.width = int(res[0])
        self.height = int(res[1])
        self.fps = float(camera_setting.get("fps", 30))

        # -- capture --------------------------------------------------------
        self.cap = cv2.VideoCapture(self.camera_channel)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open camera channel {}. In WSL2 / headless "
                "environments without a camera, re-run with "
                "'--camera mock'.".format(self.camera_channel))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        print("------Camera is open------")

        # -- exposure -------------------------------------------------------
        self.default_exposure_ms = float(camera_setting.get("exposure_ms", 20.0))
        self.auto_exposure = bool(camera_setting.get("auto_exposure", False))
        self.exposure_units = str(camera_setting.get("exposure_units", "auto"))
        if exposure_ms is not None:
            self.default_exposure_ms = float(exposure_ms)
        self.exposure_ms = self.default_exposure_ms
        self._backend = self._detect_backend()
        self.exposure_ms_applied = None
        self.exposure_control_ok = False
        self._apply_exposure()

        # -- calibration paths (9DTact layout) ------------------------------
        sensor_id = cfg.get("sensor_id", 1)
        self.calibration_root_dir = cfg.get("calibration_root_dir", "calibration")
        self.calibration_sensor_dir = os.path.join(
            self.calibration_root_dir, "sensor_{}".format(sensor_id))
        self.camera_calibration_dir = os.path.join(
            self.calibration_sensor_dir,
            camera_calibration.get("camera_calibration_dir", "camera_calibration"))
        self.row_index_path = self.camera_calibration_dir + \
            camera_calibration.get("row_index_path", "/row_index.npy")
        self.col_index_path = self.camera_calibration_dir + \
            camera_calibration.get("col_index_path", "/col_index.npy")
        self.position_scale_path = self.camera_calibration_dir + \
            camera_calibration.get("position_scale_path", "/position_scale.npy")
        self.valid_mask_path = self.camera_calibration_dir + \
            camera_calibration.get("valid_mask_path", "/valid_mask_cropped.npy")

        # -- ROI + segmentation params ---------------------------------------
        self.ROI_Y0 = int(camera_calibration.get("roi_y0", 255))
        self.ROI_Y1 = int(camera_calibration.get("roi_y1", 470))
        self.thresholds = tuple(camera_calibration.get(
            "segmentation_thresholds", [25, 20, 30, 40]))
        self.reference_frames = int(camera_calibration.get("reference_frames", 10))

        # -- calibrated mode -------------------------------------------------
        self.row_index = None
        self.col_index = None
        self.position_scale = None
        self.valid_mask = None
        self.rectified_shape = None
        if calibrated:
            self.row_index = np.load(self.row_index_path)
            self.col_index = np.load(self.col_index_path)
            self.position_scale = np.load(self.position_scale_path)
            self.rectified_shape = self.row_index.shape
            if os.path.exists(self.valid_mask_path):
                self.valid_mask = np.load(self.valid_mask_path)
            else:
                print("[camera] WARNING: {} not found; valid-region masking "
                      "disabled.".format(self.valid_mask_path))
            print("[camera] Calibrated: rectified size = {} x {}, "
                  "position_scale = {}".format(
                      self.rectified_shape[1], self.rectified_shape[0],
                      self.position_scale))

    # ---------------- exposure control ----------------

    def _detect_backend(self):
        try:
            name = self.cap.getBackendName()
            if name:
                return name
        except Exception:
            pass
        try:
            info = cv2.getBuildInformation().lower()
            for key in ("v4l", "msmf", "dshow", "avfoundation", "gstreamer"):
                if key in info:
                    return key.upper()
        except Exception:
            pass
        return "UNKNOWN"

    def _apply_exposure(self):
        """Disable auto-exposure and set a fixed exposure, backend-aware.

        Never raises: if the camera does not expose the control we log a
        warning and continue. Verifies by reading the value back.
        """
        if self.auto_exposure:
            print("[exposure] auto-exposure left enabled (cfg "
                  "camera_setting.auto_exposure = true)")
            return
        be = self._backend.lower()
        # Candidate values that disable auto-exposure, backend-preferred order.
        if "v4l" in be:
            auto_off_values = (1.0, 0.25, 0.0)      # V4L2: 1 = V4L2_EXPOSURE_MANUAL
        elif "msmf" in be or "dshow" in be or "avfoundation" in be:
            auto_off_values = (0.25, 1.0, 0.0)
        else:
            auto_off_values = (0.25, 1.0, 0.0)
        # Candidate exposure settings as (scale_ms_per_unit, raw_value):
        #   V4L2 exposes 100 us units (20 ms -> 200), MSMF/DSHOW expose ms.
        units = self.exposure_units
        if units == "ms":
            cands = [(1.0, float(self.exposure_ms))]
        elif units == "100us":
            cands = [(0.1, float(self.exposure_ms) * 10.0)]
        elif "v4l" in be:
            cands = [(0.1, float(self.exposure_ms) * 10.0)]
        elif "msmf" in be or "dshow" in be or "avfoundation" in be:
            cands = [(1.0, float(self.exposure_ms))]
        else:
            cands = [(1.0, float(self.exposure_ms)),
                     (0.1, float(self.exposure_ms) * 10.0)]

        best = None  # (auto_val, raw_val, applied_ms)
        try:
            for auto_val in auto_off_values:
                try:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_val)
                except Exception:
                    pass
                for scale, raw_val in cands:
                    try:
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, raw_val)
                        read = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))
                    except Exception:
                        read = 0.0
                    applied_ms = read * scale if read > 0 else 0.0
                    if applied_ms > 0 and (
                            best is None or
                            abs(applied_ms - self.exposure_ms) <
                            abs(best[2] - self.exposure_ms)):
                        best = (auto_val, raw_val, applied_ms)
        except Exception as e:
            print("[exposure] WARNING: failed to query exposure: {}".format(e))

        if best is not None:
            self.exposure_ms_applied = best[2]
            self.exposure_control_ok = True
            print("[exposure] backend={} auto-exposure=off requested={:.0f} ms "
                  "(raw={:g}) -> read back {:.2f} ms".format(
                      self._backend, self.exposure_ms, best[1], best[2]))
        else:
            print("[exposure] WARNING: camera does not support exposure "
                  "control (or read back 0); continuing with current setting.")

    def get_exposure(self):
        return self.exposure_ms_applied

    def set_exposure(self, exposure_ms):
        self.exposure_ms = float(np.clip(exposure_ms, 1.0, 100.0))
        self._apply_exposure()
        print("[exposure] set to {:.0f} ms".format(self.exposure_ms))

    def adjust_exposure(self, delta_ms):
        base = self.exposure_ms_applied if self.exposure_ms_applied else self.exposure_ms
        self.set_exposure(base + delta_ms)

    def toggle_auto_exposure(self):
        self.auto_exposure = not self.auto_exposure
        if not self.auto_exposure:
            self._apply_exposure()
        print("[exposure] auto-exposure {}".format(
            "enabled" if self.auto_exposure else "disabled"))

    # ---------------- capture ----------------

    def get_raw_image(self):
        ret, img = self.cap.read()
        if not ret or img is None:
            return None
        return img

    def get_raw_avg_image(self, n_frames=None, interactive=True):
        """Interactive: preview until 'y' (exposure keys active), then average
        n_frames (default = reference_frames) into the reference image.
        Non-interactive: average n_frames immediately."""
        n_frames = n_frames if n_frames is not None else self.reference_frames
        if interactive and not self.headless:
            while True:
                img = self.get_raw_image()
                if img is None:
                    continue
                cv2.imshow("camera_preview", img)
                key = cv2.waitKey(1) & 0xFF
                if handle_preview_key(key, self):
                    continue
                if key == ord("y"):
                    cv2.destroyWindow("camera_preview")
                    break
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt
        img_add = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for _ in range(n_frames):
            raw = self.get_raw_image()
            if raw is None:
                continue
            img_add += raw
        return (img_add / n_frames).astype(np.uint8)

    capture_reference = get_raw_avg_image

    # ---------------- segmentation ----------------

    def get_contact_area(self, sample, ref):
        """Full-frame uint8 0/255 contact mask (paper III-D 4-condition test)."""
        return segment_contact(sample, ref, self.thresholds)

    # ---------------- rectification (calibrated mode) ----------------

    def rectify_image(self, img):
        """Rectify a raw full-frame image using the saved npy maps.

        The maps relate rectified space -> ROI-cropped raw space
        (crop = full frame [ROI_Y0:ROI_Y1, :]); works for both 2D (gray)
        and 3D (BGR) inputs."""
        cropped = img[self.ROI_Y0:self.ROI_Y1, :]
        return cropped[self.row_index, self.col_index]

    def rectify_mask(self, mask_full):
        """Rectify a full-frame 0/255 mask; zeroes pixels outside the
        convex-hull valid region recorded during calibration."""
        cropped = mask_full[self.ROI_Y0:self.ROI_Y1, :]
        rect = cropped[self.row_index, self.col_index]
        if self.valid_mask is not None:
            valid_rect = self.valid_mask[self.row_index, self.col_index]
            rect = rect.copy()
            rect[valid_rect == 0] = 0
        return rect

    def rectify_contact(self, raw, ref):
        """Segment + rectify in one call.
        Returns (mask_rectified, image_rectified)."""
        mask = self.get_contact_area(raw, ref)
        return self.rectify_mask(mask), self.rectify_image(raw)

    # ---------------- misc ----------------

    def release(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def make_camera(cfg, calibrated=False, backend="real", exposure_ms=None,
                headless=False):
    """Instantiate a Camera (real UVC hardware) or a MockCamera
    (synthetic frames, for development / testing without hardware)."""
    if backend == "mock":
        from mock_camera import MockCamera
        cam = MockCamera(cfg, calibrated=calibrated, exposure_ms=exposure_ms)
    else:
        cam = Camera(cfg, calibrated=calibrated, exposure_ms=exposure_ms)
    cam.headless = bool(headless)
    return cam


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LightTact camera preview")
    parser.add_argument("--config", default="shape_config.yaml")
    parser.add_argument("--camera", choices=["real", "mock"], default="real")
    parser.add_argument("--exposure", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cam = make_camera(cfg, calibrated=True, backend=args.camera,
                      exposure_ms=args.exposure)
    print("[camera] preview: +/- adjust exposure, 'e' reset, 'q' quit")
    while True:
        raw = cam.get_raw_image()
        if raw is None:
            continue
        cv2.imshow("raw", raw)
        if cam.rectified_shape is not None:
            rect = cam.rectify_image(raw)
            cv2.imshow("rectified", rect)
        key = cv2.waitKey(1) & 0xFF
        if handle_preview_key(key, cam):
            continue
        if key == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()

"""dcontact.py -- LightTact DContact runtime API (robot-facing entry point).

Reconstruction of the contact-segmentation software described in LightTact
paper Sec. III-D:

  * reference image I_ref  = average of N=10 no-contact frames
  * difference image       = I_raw - I_ref
  * contact classification = 4-condition brightness test (t0..t3 = 25,20,30,40)

After calibration (see _1_Camera_Calibration.py), the raw<->rectified mapping
saved as npy files is loaded automatically, so every frame can additionally be
rectified into the top-down, physically-calibrated view (isotropic px/mm --
positions can be read off in mm directly).

Typical robot-control use::

    dc = DContact(backend="real")        # loads calibration + saved reference
    if dc.in_contact():
        mask, rect = dc.detect_rectified()
        ratio = dc.contact_ratio(mask)
"""

import os
import sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_2 import make_camera  # noqa: E402
from calibration_2 import run_calibration  # noqa: E402


class DContact:
    """Fixed-exposure camera + reference subtraction + contact segmentation
    + (optionally calibrated) rectification, as one object."""

    def __init__(self, cfg_path="shape_config.yaml", calibrated=True,
                 backend="real", exposure_ms=None):
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.cfg_path = cfg_path

        # Calibration files exist? (fall back gracefully for a first run.)
        cam_cal = self.cfg["camera_calibration"]
        root = os.path.join(
            self.cfg.get("calibration_root_dir", "calibration"),
            "sensor_{}".format(self.cfg.get("sensor_id", 1)),
            cam_cal.get("camera_calibration_dir", "camera_calibration"))
        calibrated_ok = calibrated and os.path.exists(
            os.path.join(root, cam_cal.get("row_index_path", "/row_index.npy").lstrip("/")))

        self.cam = make_camera(self.cfg, calibrated=calibrated_ok,
                               backend=backend, exposure_ms=exposure_ms)
        self.calibrated = calibrated_ok
        self.min_contact_ratio = float(
            self.cfg.get("dcontact", {}).get("min_contact_ratio", 0.001))
        self._ref = None
        self.load_reference()

    # ---------------- reference image ----------------

    @property
    def reference(self):
        return self._ref

    def capture_reference(self, n_frames=None, interactive=True):
        """Average N no-contact frames into the reference image."""
        self._ref = self.cam.get_raw_avg_image(n_frames=n_frames,
                                               interactive=interactive)
        return self._ref

    def set_reference(self, ref):
        self._ref = np.asarray(ref)

    def reference_path(self):
        return os.path.join(self.cam.camera_calibration_dir, "ref.npy")

    def save_reference(self, path=None):
        if self._ref is None:
            raise RuntimeError("No reference captured yet; call capture_reference() first.")
        path = path or self.reference_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self._ref)
        return path

    def load_reference(self, path=None):
        """Auto-load a previously saved reference (author's hint: the saved
        artifacts are reused across runs). Returns True on success."""
        path = path or self.reference_path()
        if os.path.exists(path):
            self._ref = np.load(path)
            print("[dcontact] Loaded saved reference from {}".format(path))
            return True
        return False

    # ---------------- segmentation ----------------

    def detect(self, raw=None):
        """Full-frame 0/255 contact mask (paper III-D)."""
        if self._ref is None:
            raise RuntimeError("Reference image missing; call capture_reference() first.")
        raw = self.get_frame(raw)
        return self.cam.get_contact_area(raw, self._ref)

    def detect_rectified(self, raw=None):
        """(mask_rectified, image_rectified) using the saved npy mapping."""
        if not self.calibrated:
            raise RuntimeError("Not calibrated; run calibration first "
                               "(_1_Camera_Calibration.py).")
        if self._ref is None:
            raise RuntimeError("Reference image missing; call capture_reference() first.")
        raw = self.get_frame(raw)
        return self.cam.rectify_contact(raw, self._ref)

    def get_frame(self, raw=None):
        return self.cam.get_raw_image() if raw is None else np.asarray(raw)

    # ---------------- contact state ----------------

    def _valid_area(self):
        """Number of rectified pixels inside the calibrated valid region."""
        if self.cam.valid_mask is None:
            return int(np.prod(self.cam.rectified_shape))
        valid_rect = self.cam.valid_mask[self.cam.row_index, self.cam.col_index]
        return int((valid_rect > 0).sum())

    def contact_ratio(self, mask_rect=None, raw=None):
        """Fraction of the valid rectified sensing area in contact, in [0, 1]."""
        if mask_rect is None:
            mask_rect, _ = self.detect_rectified(raw)
        area = self._valid_area()
        return float((mask_rect > 0).sum()) / area if area > 0 else 0.0

    def in_contact(self, raw=None):
        return self.contact_ratio(raw=raw) > self.min_contact_ratio

    # ---------------- calibration ----------------

    def calibrate(self, headless=False, save_debug=True, exposure_ms=None):
        """Run the 5x5-grid camera calibration (saves the npy mapping)."""
        run_calibration(self.cfg_path, save_debug=save_debug,
                        headless=headless, exposure_ms=exposure_ms)

    # ---------------- misc ----------------

    def release(self):
        self.cam.release()

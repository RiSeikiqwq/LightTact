"""mock_camera.py -- synthetic LightTact camera for development/testing.

Mimics the optical properties reported in the LightTact paper: a near-black
background (per-channel Gaussian noise, mean ~0.5, std ~1.0), bright contact
regions, and -- for camera calibration -- a 5x5 dot grid (3 mm spacing at
~15.5 px/mm) warped by a mild rotation + radial distortion so that the RBF
rectification is actually exercised.

Shares the same public API as `Camera` (same config keys, same segmentation
via `segment_contact`, same rectification via the saved npy maps) so the whole
pipeline -- calibration, demo, robot-facing DContact -- runs without hardware.
"""

import os
import numpy as np
import cv2

from camera_2 import Camera, segment_contact

# Physical mock geometry (calibrated "truth" of the simulator):
MOCK_PX_PER_MM = 15.5
MOCK_GRID_MM = 3.0                 # 5x5 dots, 3 mm spacing (paper III-D)
MOCK_ROT_DEG = 2.0                 # slight grid rotation (exercises sorting)
MOCK_RADIAL_K = 0.12               # radial distortion strength
MOCK_DOT_RADIUS = 7                # px


class MockCamera(Camera):
    """Camera-compatible simulator. Inherits rectify_image/rectify_mask/
    rectify_contact/get_contact_area from Camera (attribute-compatible),
    overrides capture/render and exposure control."""

    def __init__(self, cfg, calibrated=False, exposure_ms=None, seed=0):
        self.cfg = cfg
        self.headless = bool(cfg.get("headless", False))
        self.rng = np.random.default_rng(seed)

        camera_setting = cfg["camera_setting"]
        camera_calibration = cfg["camera_calibration"]
        res = camera_setting.get("resolution", [640, 480])
        self.width = int(res[0])
        self.height = int(res[1])
        self.fps = float(camera_setting.get("fps", 30))
        self.cap = None  # no hardware

        # -- exposure (record-only; renders are exposure-independent) ------
        self.default_exposure_ms = float(camera_setting.get("exposure_ms", 20.0))
        self.auto_exposure = bool(camera_setting.get("auto_exposure", False))
        self.exposure_units = str(camera_setting.get("exposure_units", "auto"))
        if exposure_ms is not None:
            self.default_exposure_ms = float(exposure_ms)
        self.exposure_ms = self.default_exposure_ms
        self._backend = "MOCK"
        self.exposure_ms_applied = self.exposure_ms
        self.exposure_control_ok = True

        # -- calibration paths (same layout as Camera) ---------------------
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

        # -- ROI + segmentation params --------------------------------------
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

        # -- render state ----------------------------------------------------
        self._contact_mask = None      # ground-truth contact (bool full frame)
        self._board = False            # calibration board mode
        self._auto_board = False       # board appears right after ref capture
        self._ideal_dots_xy = None     # ground-truth dot grid (full frame, px)
        self._grid_center = (self.width / 2.0, (self.ROI_Y0 + self.ROI_Y1) / 2.0)

    # ---------------- ground truth control ----------------

    def set_contact_mask(self, mask):
        """Set the ground-truth contact region (full-frame 0/255 or bool)."""
        self._contact_mask = (np.asarray(mask) > 0)
        self._board = False

    def set_calibration_board(self, on=True):
        self._board = bool(on)
        if on:
            self._contact_mask = None
            self._build_ideal_dots()

    def enable_auto_board(self):
        """Model the calibration workflow: the board is pressed onto the
        sensor AFTER the reference image is captured, so the board appears
        automatically in all frames following get_raw_avg_image()."""
        self._auto_board = True

    def _build_ideal_dots(self):
        rows, cols = (int(self.cfg["camera_calibration"]["row_points"]),
                      int(self.cfg["camera_calibration"]["col_points"]))
        pitch = MOCK_GRID_MM * MOCK_PX_PER_MM
        cx, cy = self._grid_center
        pts = []
        for r in range(rows):
            for c in range(cols):
                pts.append([cx + (c - cols // 2) * pitch,
                            cy + (r - rows // 2) * pitch])
        self._ideal_dots_xy = np.array(pts, dtype=np.float64)  # (N, 2) (x, y)

    @property
    def gt_dot_xy(self):
        """Ground-truth (undistorted) dot positions in full-frame pixels."""
        if self._ideal_dots_xy is None:
            self._build_ideal_dots()
        return self._ideal_dots_xy.copy()

    # ---------------- rendering ----------------

    def _warp(self, pts):
        """Rotate + radially distort ideal positions into the observed grid."""
        cx, cy = self._grid_center
        # 1) rotation around grid center
        ang = np.deg2rad(MOCK_ROT_DEG)
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]])
        p = (pts - np.array([cx, cy])) @ R.T + np.array([cx, cy])
        # 2) radial distortion (normalized coordinates)
        nx = (p[:, 0] - cx) / (self.width / 2.0)
        ny = (p[:, 1] - cy) / ((self.ROI_Y1 - self.ROI_Y0) / 1.2)
        r2 = nx ** 2 + ny ** 2
        out = (p - np.array([cx, cy])) * (1.0 + MOCK_RADIAL_K * r2)[:, None] \
            + np.array([cx, cy])
        return out

    def _render(self):
        # near-black background, per-channel Gaussian noise (paper: mean < 3)
        img = np.clip(self.rng.normal(0.5, 1.0,
                                      (self.height, self.width, 3)),
                      0, 255).astype(np.float32)
        if self._board and self._ideal_dots_xy is not None:
            pts_w = self._warp(self._ideal_dots_xy)
            for (x, y) in pts_w:
                cv2.circle(img, (int(round(x)), int(round(y))),
                           MOCK_DOT_RADIUS, (205.0, 205.0, 215.0), -1)
        elif self._contact_mask is not None:
            mask_f = self._contact_mask.astype(np.float32)
            mask_blur = cv2.GaussianBlur(mask_f, (7, 7), 0)  # soft edges
            tex = self.rng.uniform(60.0, 200.0,
                                   (self.height, self.width, 3))
            img += mask_blur[..., None] * tex
        return np.clip(img, 0, 255).astype(np.uint8)

    # ---------------- capture (overrides Camera) ----------------

    def get_raw_image(self):
        return self._render()

    def get_raw_avg_image(self, n_frames=None, interactive=True):
        """Average n_frames renders (no interactive preview in the simulator;
        the averaging itself matches the real Camera's behavior)."""
        n_frames = n_frames if n_frames is not None else self.reference_frames
        img_add = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for _ in range(n_frames):
            img_add += self.get_raw_image()
        if self._auto_board and not self._board:
            self.set_calibration_board(True)   # board pressed after the ref
        return (img_add / n_frames).astype(np.uint8)

    capture_reference = get_raw_avg_image

    def get_contact_area(self, sample, ref):
        # deliberately identical to Camera: shared segment_contact code path
        return segment_contact(sample, ref, self.thresholds)

    # ---------------- exposure (record-only) ----------------

    def get_exposure(self):
        return self.exposure_ms

    def set_exposure(self, exposure_ms):
        self.exposure_ms = float(np.clip(exposure_ms, 1.0, 100.0))
        self.exposure_ms_applied = self.exposure_ms
        return True

    def adjust_exposure(self, delta_ms):
        self.set_exposure(self.exposure_ms + delta_ms)

    def toggle_auto_exposure(self):
        self.auto_exposure = not self.auto_exposure

    def release(self):
        pass

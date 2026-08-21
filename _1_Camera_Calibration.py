"""_1_Camera_Calibration.py -- LightTact camera calibration entry point.

CLI wrapper around the author's `run_calibration` (calibration_2.py):

  1. capture a reference image (average of N no-contact frames),
  2. press the 5x5 cylindrical-bump calibration board on the sensor and
     capture a sample image,
  3. detect the imprint centers, build the rectified grid, and save the
     raw<->rectified mapping as npy files (reused by every later run).

Run:
    python _1_Camera_Calibration.py                    # real camera
    python _1_Camera_Calibration.py --camera mock --headless   # no hardware

Use the '+/-' keys in the preview window to adjust the exposure time
before confirming the capture with 'y' (paper: fixed 20 ms default).
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_2 import make_camera  # noqa: E402
from calibration_2 import run_calibration  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="LightTact camera calibration (5x5 grid rectification).")
    parser.add_argument("--config", default="shape_config.yaml",
                        help="path to the config yaml (default: shape_config.yaml)")
    parser.add_argument("--camera", choices=["real", "mock"], default="real",
                        help="'real' for the UVC camera, 'mock' for the "
                             "synthetic simulator (no hardware)")
    parser.add_argument("--headless", action="store_true",
                        help="run without any GUI windows (no imshow)")
    parser.add_argument("--exposure", type=float, default=None,
                        help="override the fixed exposure time in ms "
                             "(default: config camera_setting.exposure_ms, 20 ms)")
    parser.add_argument("--save-debug", action="store_true", default=True,
                        help="save debug/preview images (default: true)")
    parser.add_argument("--no-save-debug", action="store_false", dest="save_debug",
                        help="disable saving debug/preview images")
    args = parser.parse_args()

    cfg_path = args.config

    if args.camera == "mock":
        # Simulator: model the human pressing the calibration board on the
        # sensor right after the reference image is captured.
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        cam = make_camera(cfg, calibrated=False, backend="mock",
                          exposure_ms=args.exposure, headless=args.headless)
        cam.enable_auto_board()
        run_calibration(cfg_path, save_debug=args.save_debug,
                        headless=args.headless, exposure_ms=args.exposure,
                        camera=cam)
    else:
        run_calibration(cfg_path, save_debug=args.save_debug,
                        headless=args.headless, exposure_ms=args.exposure,
                        backend=args.camera)
    return 0


if __name__ == "__main__":
    sys.exit(main())

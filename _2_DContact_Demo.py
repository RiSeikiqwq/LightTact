"""_2_DContact_Demo.py -- live LightTact contact-segmentation demo.

Flow:
  1. capture the reference image (average of N no-contact frames),
  2. loop: raw frame -> contact mask (paper III-D) -> rectified view
     (if calibration files exist), print the contact ratio,
  3. keys: '+'/'-' adjust exposure, 'e' reset exposure, 'y' re-capture the
     reference, 'r' save debug images, 'q' quit.

Run:
    python _2_DContact_Demo.py                    # real camera
    python _2_DContact_Demo.py --camera mock --headless --frames 20
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from camera_2 import handle_preview_key, imwrite_unicode  # noqa: E402
from dcontact import DContact  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="LightTact DContact live demo.")
    parser.add_argument("--config", default="shape_config.yaml")
    parser.add_argument("--camera", choices=["real", "mock"], default="real")
    parser.add_argument("--headless", action="store_true",
                        help="run without GUI windows")
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument("--frames", type=int, default=0,
                        help="number of frames to process (0 = infinite)")
    parser.add_argument("--save-debug", action="store_true", default=True,
                        help="save debug images on 'r' / at exit in headless mode")
    args = parser.parse_args()

    dc = DContact(cfg_path=args.config, calibrated=True, backend=args.camera,
                  exposure_ms=args.exposure)
    dc.cam.headless = args.headless

    # 1) reference image
    if dc.reference is None:
        print("[demo] DON'T touch the sensor. Press 'y' to capture the "
              "reference image, 'q' to quit. (+/-: exposure)")
        dc.capture_reference(interactive=not args.headless)
        dc.save_reference()
    else:
        print("[demo] Using saved reference image.")

    # 2) live loop
    print("[demo] '+/-' adjust exposure, 'e' reset, 'y' re-capture ref, "
          "'r' save debug, 'q' quit.")
    frame_idx = 0
    while True:
        raw = dc.get_frame()
        if raw is None:
            continue
        mask = dc.detect(raw)
        ratio = None
        if dc.calibrated:
            mask_rect, img_rect = dc.detect_rectified(raw)
            ratio = dc.contact_ratio(mask_rect)
        if not args.headless:
            cv2.imshow("raw", raw)
            cv2.imshow("contact_mask", mask)
            if dc.calibrated:
                cv2.imshow("rectified", img_rect)
                cv2.imshow("rectified_contact", mask_rect)
        if ratio is not None:
            print("frame {:4d}  contact_ratio = {:.4f}  contact = {}".format(
                frame_idx, ratio, ratio > dc.min_contact_ratio))
        else:
            print("frame {:4d}  (not calibrated)".format(frame_idx))

        if not args.headless:
            key = cv2.waitKey(1) & 0xFF
            if handle_preview_key(key, dc.cam):
                continue
            if key == ord("y"):
                print("[demo] Re-capturing reference...")
                dc.capture_reference(interactive=True)
                dc.save_reference()
                continue
            if key == ord("r"):
                out = dc.cam.camera_calibration_dir
                os.makedirs(out, exist_ok=True)
                imwrite_unicode(os.path.join(out, "demo_raw.png"), raw)
                imwrite_unicode(os.path.join(out, "demo_mask.png"), mask)
                if dc.calibrated:
                    imwrite_unicode(os.path.join(out, "demo_rectified.png"), img_rect)
                    imwrite_unicode(os.path.join(out, "demo_rectified_mask.png"),
                                    mask_rect)
                print("[demo] Saved debug images to {}".format(out))
            if key == ord("q"):
                break

        frame_idx += 1
        if args.frames and frame_idx >= args.frames:
            break

    if args.headless and args.save_debug and dc.calibrated:
        out = dc.cam.camera_calibration_dir
        os.makedirs(out, exist_ok=True)
        imwrite_unicode(os.path.join(out, "demo_raw.png"), raw)
        imwrite_unicode(os.path.join(out, "demo_mask.png"), mask)
        imwrite_unicode(os.path.join(out, "demo_rectified.png"), img_rect)
        imwrite_unicode(os.path.join(out, "demo_rectified_mask.png"), mask_rect)

    dc.release()
    if not args.headless:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

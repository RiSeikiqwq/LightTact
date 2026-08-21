import os
import cv2
import yaml
import numpy as np
from scipy.interpolate import Rbf

from camera_2 import make_camera, imwrite_unicode, handle_preview_key  # import your tuned camera module


# ---------- headless UI helpers ----------

def _wait(ms, headless: bool):
    """cv2.waitKey that returns -1 immediately in headless mode."""
    if headless:
        return -1
    return cv2.waitKey(ms)


# ---------- small helpers ----------

def _find_dot_centroids(mask: np.ndarray, min_area: float = 1.0):
    """
    Return Nx2 array of centroids (x, y) from a binary (0/255) mask.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        pts.append([cx, cy])
    return np.array(pts, dtype=np.float32)


def _sort_grid_points(pts_xy: np.ndarray, rows: int, cols: int):
    """
    Sort scattered points into a (rows x cols) grid order.
    Strategy: sort by y, split into row groups, then sort each group by x.
    Returns:
      init_position: (rows*cols, 2) array in (row-major) order of (y, x) as floats.
    """
    if pts_xy.shape[0] != rows * cols:
        raise RuntimeError(f"Expected {rows*cols} dots, found {pts_xy.shape[0]}")

    # sort by y (then by x)
    idx = np.lexsort((pts_xy[:, 0], pts_xy[:, 1]))  # sort by y primary, x secondary
    pts_sorted = pts_xy[idx]

    # split into 'rows' bands by y
    bands = np.array_split(pts_sorted, rows)
    ordered = []
    for band in bands:
        band_sorted = band[np.argsort(band[:, 0])]  # sort by x within that band
        ordered.append(band_sorted)
    grid = np.vstack(ordered)  # (rows*cols, 2) in row-major order

    # Return as (y, x) to match image-row/col convention downstream
    grid_yx = np.stack([grid[:, 1], grid[:, 0]], axis=1)  # (y, x)
    return grid_yx.astype(np.float32)


def _convex_hull_mask(h: int, w: int, pts_xy: np.ndarray) -> np.ndarray:
    """
    Binary mask (uint8) of the convex hull of given (x,y) points in CROPPED image space.
    """
    hull = cv2.convexHull(pts_xy.reshape(-1, 1, 2).astype(np.float32))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask


# ---------- main calibration routine ----------

# Extra sensing area margins (in mm) on EACH side:
#   - vertical: top and bottom
#   - horizontal: left and right
VERTICAL_ADD_MM   = 1.0   # 1 mm above + 1 mm below grid
HORIZONTAL_ADD_MM = 0.2   # 0.2 mm left + 0.2 mm right


def run_calibration(cfg_path: str = "shape_config.yaml", save_debug: bool = True,
                    headless: bool = False, exposure_ms: float = None,
                    backend: str = "real", camera=None):
    # Load YAML & camera (uncalibrated)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # ROI (vertical crop of the raw frame) comes from the config; the
    # defaults match the author's original hardcoded values.
    ROI_Y0 = int(cfg["camera_calibration"].get("roi_y0", 255))
    ROI_Y1 = int(cfg["camera_calibration"].get("roi_y1", 470))

    if camera is None:
        camera = make_camera(cfg, calibrated=False, backend=backend,
                             exposure_ms=exposure_ms, headless=headless)
    cam = camera
    cam.headless = headless

    cam_dir = cam.camera_calibration_dir
    os.makedirs(cam_dir, exist_ok=True)

    rows = int(cfg["camera_calibration"]["row_points"])   # e.g., 5
    cols = int(cfg["camera_calibration"]["col_points"])   # e.g., 5

    # legacy single spacing (mm), kept as fallback
    grid_mm = float(cfg["camera_calibration"].get("grid_distance", 3.0))

    # separate physical spacing in mm for rows and columns (can be different)
    row_mm = float(cfg["camera_calibration"].get("row_distance_mm", grid_mm))
    col_mm = float(cfg["camera_calibration"].get("col_distance_mm", grid_mm))

    # 1) Capture REFERENCE (avg over N)
    print("DON'T touch the sensor. Press 'y' to capture REFERENCE (avg). 'q' to quit.")
    ref = cam.get_raw_avg_image(interactive=not headless)
    imwrite_unicode(os.path.join(cam_dir, "ref_full.png"), ref)
    print("[calib] Reference captured.")

    # 2) Capture SAMPLE (with calibration board in contact)
    print("Press/hold the calibration board on the sensor. Press 'y' to capture SAMPLE.")
    while True:
        sample = cam.get_raw_image()
        if sample is None:
            continue
        prev = sample.copy()
        if not headless:
            cv2.imshow("sample_preview_full", prev)
        key = _wait(1, headless) & 0xFF
        if handle_preview_key(key, cam):  # '+/-' adjust exposure, 'e' reset
            continue
        if headless or key == ord("y"):
            if not headless:
                cv2.destroyWindow("sample_preview_full")
            break
        if key == ord("q"):
            if not headless:
                cv2.destroyAllWindows()
            raise SystemExit

    imwrite_unicode(os.path.join(cam_dir, "sample_full.png"), sample)

    # 3) Build contact mask on full frame
    contact_full = cam.get_contact_area(sample, ref)  # 0/255
    imwrite_unicode(os.path.join(cam_dir, "contact_mask_full.png"), contact_full)

    # 3b) Crop to active region BEFORE dot detection
    contact = contact_full[ROI_Y0:ROI_Y1, :]
    H_c, W_c = contact.shape[:2]
    imwrite_unicode(os.path.join(cam_dir, "contact_mask_cropped.png"), contact)
    print(f"[calib] Cropped size (H_c, W_c) = ({H_c}, {W_c})")

    # Optional clean-up (helps split blobs)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    contact_clean = cv2.morphologyEx(contact, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4) Detect dot centroids in CROPPED contact mask
    pts_xy = _find_dot_centroids(contact_clean, min_area=1.0)  # (N,2) (x,y) in cropped coordinates
    print(f"[calib] Found {pts_xy.shape[0]} dots in cropped ROI.")

    if pts_xy.shape[0] != rows * cols:
        # Show to help adjust threshold/lighting if needed
        if not headless:
            dbg = cv2.cvtColor(contact_clean, cv2.COLOR_GRAY2BGR)
            for (x, y) in pts_xy.astype(int):
                cv2.circle(dbg, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("detected_dots_raw_cropped", dbg)
            cv2.waitKey(0)
            cv2.destroyWindow("detected_dots_raw_cropped")
        raise RuntimeError(f"Expected {rows*cols} dots, got {pts_xy.shape[0]} — adjust lighting/thresholds.")

    # 5) Sort to a (rows x cols) grid order (row-major), produce init_position (image space, CROPPED)
    init_position_yx = _sort_grid_points(pts_xy, rows, cols)  # (N,2) of (y,x) in cropped image space

    # 6) Compute center index
    center_index = (rows * cols) // 2
    cy_cx = init_position_yx[center_index]          # (y, x) in cropped coordinates
    cy, cx = float(cy_cx[0]), float(cy_cx[1])

    center_r = center_index // cols
    center_c = center_index % cols

    # --- for DEBUG: neighbors of center (still used to draw lines later) ---
    vert_neighbor_indices = []
    if center_r - 1 >= 0:      # up
        vert_neighbor_indices.append(center_index - cols)
    if center_r + 1 < rows:    # down
        vert_neighbor_indices.append(center_index + cols)

    horiz_neighbor_indices = []
    if center_c - 1 >= 0:      # left
        horiz_neighbor_indices.append(center_index - 1)
    if center_c + 1 < cols:    # right
        horiz_neighbor_indices.append(center_index + 1)

    # 7) Use the FOUR CORNERS of the grid in CROPPED space as a trapezoid
    # indices in row-major:
    #   UL: row 0, col 0
    #   UR: row 0, col cols-1
    #   BL: row rows-1, col 0
    #   BR: row rows-1, col cols-1
    idx_ul = 0
    idx_ur = cols - 1
    idx_bl = (rows - 1) * cols
    idx_br = rows * cols - 1

    p_ul = init_position_yx[idx_ul]  # (y,x)
    p_ur = init_position_yx[idx_ur]
    p_bl = init_position_yx[idx_bl]
    p_br = init_position_yx[idx_br]

    # lengths of the 4 edges of the trapezoid (in pixels)
    top_len    = float(np.linalg.norm(p_ur - p_ul))  # spans 4 column gaps
    bottom_len = float(np.linalg.norm(p_br - p_bl))  # spans 4 column gaps
    left_len   = float(np.linalg.norm(p_bl - p_ul))  # spans 4 row  gaps
    right_len  = float(np.linalg.norm(p_br - p_ur))  # spans 4 row  gaps

    # Since it's a 5x5 grid, each side has (rows-1) or (cols-1) = 4 equal gaps.
    # Average pixel spacing between adjacent columns:
    dis_col_px = (top_len + bottom_len) / (2.0 * (cols - 1))
    # Average pixel spacing between adjacent rows:
    dis_row_px = (left_len + right_len) / (2.0 * (rows - 1))

    print(f"[calib] top={top_len:.3f}, bottom={bottom_len:.3f}, "
          f"left={left_len:.3f}, right={right_len:.3f}")
    print(f"[calib] dis_row_px={dis_row_px:.3f}, dis_col_px={dis_col_px:.3f}")

    # 7b) mm per pixel (for info) and per-axis pixels per mm
    pixel_per_mm_row = row_mm / dis_row_px   # mm per pixel vertically
    pixel_per_mm_col = col_mm / dis_col_px   # mm per pixel horizontally

    px_per_mm_row = dis_row_px / row_mm      # pixels per mm vertically
    px_per_mm_col = dis_col_px / col_mm      # pixels per mm horizontally

    # ---- NEW: single, isotropic pixels-per-mm scale ----
    px_per_mm = 0.5 * (px_per_mm_row + px_per_mm_col)

    # Save position_scale = [center_row, center_col, pixel_per_mm_row, pixel_per_mm_col]
    # NOTE: center_row/col are in CROPPED coordinates (0 at ROI_Y0).
    position_scale = np.array([cy, cx, pixel_per_mm_row, pixel_per_mm_col], dtype=np.float32)
    np.save(cam.position_scale_path, position_scale)

    print(f"[calib] row_mm={row_mm:.3f} mm, col_mm={col_mm:.3f} mm")
    print(f"[calib] pixel_per_mm_row={pixel_per_mm_row:.6f}, pixel_per_mm_col={pixel_per_mm_col:.6f}")
    print(f"[calib] px_per_mm (avg)={px_per_mm:.6f}")
    print(f"[calib] center(cropped) = ({cy:.1f},{cx:.1f})")
    print(f"[calib] Saved position_scale.npy → {cam.position_scale_path}")

    # 8) Build "real" sampling coordinates in PHYSICAL space (mm), centered at (0,0)
    #    real_position_yx[:,0] = real_y_mm, real_position_yx[:,1] = real_x_mm
    real_position_yx = np.zeros_like(init_position_yx, dtype=np.float32)
    for ri in range(rows):
        for ci in range(cols):
            idx = ri * cols + ci
            # offsets from center row/col in grid units (negative up/left)
            drow = ri - rows // 2
            dcol = ci - cols // 2
            real_y_mm = drow * row_mm
            real_x_mm = dcol * col_mm
            real_position_yx[idx, 0] = real_y_mm
            real_position_yx[idx, 1] = real_x_mm

    # 9) Fit RBF mapping: (real_y_mm, real_x_mm) -> (img_row, img_col) in CROPPED coordinates
    ry = real_position_yx[:, 0]  # in mm
    rx = real_position_yx[:, 1]  # in mm
    iy = init_position_yx[:, 0]  # image rows (cropped, in pixels)
    ix = init_position_yx[:, 1]  # image cols (cropped, in pixels)

    itp_row = Rbf(ry, rx, iy, function="cubic")
    itp_col = Rbf(ry, rx, ix, function="cubic")

    # 10) Compute physical extents (mm) of the grid + margins, then convert to pixels

    # Physical size of the 5x5 grid (4 gaps between dots) in mm:
    grid_height_mm = (rows - 1) * row_mm
    grid_width_mm  = (cols - 1) * col_mm

    # Total physical size including extra margins:
    total_height_mm = grid_height_mm + 2.0 * VERTICAL_ADD_MM
    total_width_mm  = grid_width_mm  + 2.0 * HORIZONTAL_ADD_MM

    # Desired output resolution in pixels, using a SINGLE isotropic px_per_mm
    H_out = int(round(total_height_mm * px_per_mm))
    W_out = int(round(total_width_mm  * px_per_mm))
    H_out = max(H_out, 8)
    W_out = max(W_out, 8)

    print(f"[calib] total_height_mm={total_height_mm:.3f}, total_width_mm={total_width_mm:.3f}")
    print(f"[calib] => rectified resolution (H_out x W_out) = {H_out} x {W_out}")

    # half extents in mm (around center)
    half_h_mm = total_height_mm / 2.0
    half_w_mm = total_width_mm  / 2.0

    # We define canonical domain in mm: [-half_h_mm, +half_h_mm] x [-half_w_mm, +half_w_mm]
    y_min = -half_h_mm
    y_max = +half_h_mm
    x_min = -half_w_mm
    x_max = +half_w_mm

    # Build rectified meshgrid over that domain (in mm)
    y_coords = np.linspace(y_min, y_max, H_out, dtype=np.float32)
    x_coords = np.linspace(x_min, x_max, W_out, dtype=np.float32)
    col_mesh, row_mesh = np.meshgrid(x_coords, y_coords)  # x→cols (mm), y→rows (mm)

    # evaluate maps: rectified (row_mesh_mm, col_mesh_mm) -> raw cropped indices (pixels)
    row_index_f = itp_row(row_mesh, col_mesh)
    col_index_f = itp_col(row_mesh, col_mesh)

    # 11) Clamp to bounds of CROPPED RAW region (0..H_c-1, 0..W_c-1)
    row_index = np.clip(row_index_f, 0, H_c - 1).astype(np.int32)
    col_index = np.clip(col_index_f, 0, W_c - 1).astype(np.int32)

    # Save dense maps (rectified space -> CROPPED raw space)
    np.save(cam.row_index_path, row_index)
    np.save(cam.col_index_path, col_index)
    print(f"[calib] Saved row_index.npy → {cam.row_index_path}")
    print(f"[calib] Saved col_index.npy → {cam.col_index_path}")

    # 12) Save a valid-region mask (convex hull in CROPPED space)
    hull_mask_raw = _convex_hull_mask(H_c, W_c, pts_xy)
    valid_mask_path = os.path.join(cam.camera_calibration_dir, "valid_mask_cropped.npy")
    np.save(valid_mask_path, hull_mask_raw)
    print(f"[calib] Saved valid_mask_cropped.npy → {valid_mask_path}")

    # ---------- optional previews ----------
    if save_debug:
        # ========= 1) RAW-CROPPED SPACE: trapezoid + all grid pts =========
        dbg_raw = np.zeros((H_c, W_c, 3), dtype=np.uint8)

        # ---- draw ALL grid points in white (including center) ----
        for idx in range(rows * cols):
            py, px = init_position_yx[idx]  # (y, x)
            iy2 = int(round(py))
            ix2 = int(round(px))
            if 0 <= iy2 < H_c and 0 <= ix2 < W_c:
                cv2.circle(dbg_raw, (ix2, iy2), 8, (255, 255, 255), -1)  # larger white dots

        # corner points in CROPPED RAW space (blue)
        for py, px in (p_ul, p_ur, p_br, p_bl):
            iy2 = int(round(py))
            ix2 = int(round(px))
            if 0 <= iy2 < H_c and 0 <= ix2 < W_c:
                cv2.circle(dbg_raw, (ix2, iy2), 11, (0, 255, 255), -1)  # green vertices

        # trapezoid edges in RED
        pts_poly_raw = np.array(
            [[int(round(p_ul[1])), int(round(p_ul[0]))],
             [int(round(p_ur[1])), int(round(p_ur[0]))],
             [int(round(p_br[1])), int(round(p_br[0]))],
             [int(round(p_bl[1])), int(round(p_bl[0]))]],
            dtype=np.int32
        )
        cv2.polylines(dbg_raw, [pts_poly_raw], isClosed=True, color=(0, 0, 255), thickness=5)

        if not headless:
            cv2.imshow("raw_contact_with_dots_cropped", dbg_raw)

        # ========= 2) RECTIFIED SPACE: rectangle + all grid pts =========
        # Now real_y/real_x and y_min/x_min are all in mm.
        def _real_to_rect(real_y: float, real_x: float) -> tuple[float, float]:
            denom_y = (y_max - y_min) if (y_max > y_min) else 1.0
            denom_x = (x_max - x_min) if (x_max > x_min) else 1.0
            ry_norm = (real_y - y_min) / denom_y
            rx_norm = (real_x - x_min) / denom_x
            py = ry_norm * (H_out - 1)
            px = rx_norm * (W_out - 1)
            return py, px

        dbg_rect = np.zeros((H_out, W_out, 3), dtype=np.uint8)

        # precompute rectified coordinates for all points
        rect_coords = np.zeros_like(init_position_yx, dtype=np.float32)  # (N,2) (y,x)
        for idx in range(rows * cols):
            real_y, real_x = real_position_yx[idx]  # in mm
            py, px = _real_to_rect(real_y, real_x)
            rect_coords[idx, 0] = py
            rect_coords[idx, 1] = px

            iy2 = int(round(py))
            ix2 = int(round(px))
            if 0 <= iy2 < H_out and 0 <= ix2 < W_out:
                cv2.circle(dbg_rect, (ix2, iy2), 8, (255, 255, 255), -1)  # larger white dots

        # corners in rectified space (blue)
        ul_rect_y, ul_rect_x = rect_coords[idx_ul]
        ur_rect_y, ur_rect_x = rect_coords[idx_ur]
        bl_rect_y, bl_rect_x = rect_coords[idx_bl]
        br_rect_y, br_rect_x = rect_coords[idx_br]

        corner_rect_pts = []
        for py, px in ((ul_rect_y, ul_rect_x),
                       (ur_rect_y, ur_rect_x),
                       (br_rect_y, br_rect_x),
                       (bl_rect_y, bl_rect_x)):
            iy2 = int(round(py))
            ix2 = int(round(px))
            if 0 <= iy2 < H_out and 0 <= ix2 < W_out:
                cv2.circle(dbg_rect, (ix2, iy2), 8, (255, 255, 255), -1)  # blue vertices
                corner_rect_pts.append([ix2, iy2])

        corner_rect_pts = np.array(corner_rect_pts, dtype=np.int32)
        #if corner_rect_pts.shape[0] == 4:
            # rectangle edges in RED
            #cv2.polylines(dbg_rect, [corner_rect_pts], isClosed=True, color=(0, 0, 255), thickness=5)

        if not headless:
            cv2.imshow("rectified_contact_cropped (calibrated)", dbg_rect)

        # ========= 3) RECTIFIED + VALID MASK =========
        valid_mapped = hull_mask_raw[row_index, col_index]  # shape: (H_out, W_out)

        dbg_rect_masked = dbg_rect.copy()
        dbg_rect_masked[valid_mapped == 0] = 0
        if not headless:
            cv2.imshow("rectified_contact_masked_cropped", dbg_rect_masked)

        # ========= 4) SAVE all three debug images =========
        raw_trap_path         = os.path.join(cam_dir, "raw_trapezoid_debug.png")
        rect_trap_path        = os.path.join(cam_dir, "rectified_trapezoid_debug.png")
        rect_trap_masked_path = os.path.join(cam_dir, "rectified_trapezoid_masked_debug.png")

        imwrite_unicode(raw_trap_path, dbg_raw)
        imwrite_unicode(rect_trap_path, dbg_rect)
        imwrite_unicode(rect_trap_masked_path, dbg_rect_masked)

        print("[calib] Saved debug images to:")
        print(f"  {raw_trap_path}")
        print(f"  {rect_trap_path}")
        print(f"  {rect_trap_masked_path}")
        if not headless:
            print("[calib] Close the preview windows or press any key to finish.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# ---------- entry point ----------

if __name__ == "__main__":
    run_calibration("shape_config.yaml", save_debug=True)

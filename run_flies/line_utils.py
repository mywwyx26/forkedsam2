"""
line_utils.py

Geometry helpers for the line stack: ordering line pixels into paths, building
filled "between two lines" masks, and generating every (i, j) combo mask plus
its negative prompt points.

Line index convention (shared with run_sam2_lines.py):
    0                  -> original boundary line 1
    1 .. n_lines       -> interpolated lines
    n_lines + 1        -> original boundary line 2

The two boundary lines are never mask bounds; they only supply negative
points for the outermost combos.
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree


def order_points_by_path(label_img, label_value):
    """
    Extract points for a given label and order them by walking along the
    skeleton path (nearest-neighbor traversal from one endpoint to the
    other). Works for both the original (possibly thick/smoothed) boundary
    lines and the thin rasterized interpolated lines.
    """
    binary = (label_img == label_value)
    skel = skeletonize(binary)
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return np.empty((0, 2))
    pts = np.stack([xs, ys], axis=1).astype(np.float64)

    tree = cKDTree(pts)
    neighbor_counts = tree.query_ball_point(pts, r=1.5, return_length=True)
    start_idx = np.argmin(neighbor_counts)

    visited = np.zeros(len(pts), dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    current = start_idx
    for _ in range(len(pts) - 1):
        _, idxs = tree.query(pts[current], k=len(pts))
        next_idx = next(i for i in idxs if not visited[i])
        order.append(next_idx)
        visited[next_idx] = True
        current = next_idx

    return pts[order]


def get_line_points(idx, boundary_mask, label_value1, label_value2,
                    lines_label_img, start_label, n_lines):
    """
    Unified accessor across the full line stack:
      idx == 0            -> original boundary line 1 (in boundary_mask)
      idx == n_lines + 1   -> original boundary line 2 (in boundary_mask)
      1 <= idx <= n_lines  -> interpolated line (in lines_label_img)
    """
    if idx == 0:
        return order_points_by_path(boundary_mask, label_value1)
    elif idx == n_lines + 1:
        return order_points_by_path(boundary_mask, label_value2)
    else:
        label_value = start_label + (idx - 1)
        return order_points_by_path(lines_label_img, label_value)


def build_between_mask(pts_i, pts_j, shape):
    """
    Fill the region between two ordered curves. The polygon is built by
    walking forward along curve i, then backward along curve j -- the
    straight edges connecting curve_i's endpoints to curve_j's endpoints
    are exactly the "sides" of the shape, formed automatically since
    cv2.fillPoly closes the last point back to the first.

    Requires pts_i and pts_j to be ordered in the SAME direction (e.g. both
    top-to-bottom); otherwise the polygon will twist/self-intersect.
    """
    if len(pts_i) < 2 or len(pts_j) < 2:
        return None
    polygon = np.concatenate([pts_i, pts_j[::-1]], axis=0).astype(np.int32)
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [polygon], color=1)
    return canvas


def sample_points_from_ordered(pts, num_points):
    """Evenly-spaced subsample of an already-ordered point array."""
    if len(pts) == 0:
        return np.empty((0, 2))
    if len(pts) <= num_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, num_points).astype(int)
    return pts[idx]


def generate_all_combo_masks(boundary_mask, label_value1, label_value2,
                             lines_label_img, start_label, n_lines,
                             shape, min_gap=3, max_gap=10, points_per_adjacent_line=5):
    """
    Build a filled mask for every pair of INTERIOR interpolated lines (i, j)
    with min_gap <= j - i <= max_gap, where i, j both range over 1..n_lines (the
    interpolated lines only). The two original boundary lines (index 0 and index
    n_lines + 1) are deliberately never used as mask bounds -- they exist only to
    provide negative prompts for the outermost combos, so every mask is "surrounded"
    on both sides even at the very edges of the line stack.

    For each combo mask, also build a set of negative points: a sample of
    points from the line immediately outside the mask on each side (i - 1
    and j + 1 -- which may be a boundary line for the outermost combos).

    Returns
    -------
    combo_masks : dict {(i, j): binary_mask}
    combo_neg_points : dict {(i, j): np.ndarray of shape (n_points, 2)}
    """
    n_total = n_lines + 2

    # cache ordered points per line index (0..n_total-1) so each line is
    # only extracted once, including the two boundary lines used for negatives
    line_points_cache = {
        idx: get_line_points(idx, boundary_mask, label_value1, label_value2,
                             lines_label_img, start_label, n_lines)
        for idx in range(n_total)
    }

    combo_masks = {}
    combo_neg_points = {}

    for i in range(1, n_lines + 1):
        j_min = i + min_gap
        j_max = min(i + max_gap, n_lines)
        for j in range(j_min, j_max + 1):
            pts_i = line_points_cache[i]
            pts_j = line_points_cache[j]
            mask = build_between_mask(pts_i, pts_j, shape)
            if mask is None:
                continue
            combo_masks[(i, j)] = mask

            neg_points_list = []

            # points from the line immediately outside the mask on each side
            left_adj_idx = i - 1
            right_adj_idx = j + 1
            for adj_idx in (left_adj_idx, right_adj_idx):
                adj_pts = line_points_cache.get(adj_idx)
                if adj_pts is not None and len(adj_pts) > 0:
                    neg_points_list.append(sample_points_from_ordered(adj_pts, points_per_adjacent_line))

            neg_points_list = [p for p in neg_points_list if len(p) > 0]
            combo_neg_points[(i, j)] = (
                np.concatenate(neg_points_list, axis=0) if neg_points_list else np.empty((0, 2))
            )

    return combo_masks, combo_neg_points

"""
run_sam2_lines.py

Run SAM2 using MASK inputs plus NEGATIVE POINT prompts. For every pair of
INTERIOR interpolated lines (i, j) at most `max_gap` positions apart, build
a filled mask of the region between them (closing the polygon by connecting
the two lines' endpoints directly -- no curve fitting on the sides), and
use that filled region as a SAM2 mask prompt.

The two original boundary lines are never used as mask bounds themselves --
they exist only to supply negative point prompts for the outermost combos,
so every mask (including ones right at the edge of the line stack) has
negative points surrounding it on both sides.

For each combo mask, negative points come from a sample of points along the
line immediately outside the mask on each side (which may be a boundary
line for the outermost combos).

No positive point prompts are used -- the mask_input itself is the positive prior.
"""

import math
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from contextlib import nullcontext
from pathlib import Path
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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


def mask_to_sam2_input(binary_mask, logit_scale=15.0, low_res_size=256):
    """
    Convert a binary region mask into the low-resolution logit format SAM2
    expects for `mask_input` (shape (1, 256, 256), float32). Foreground
    pixels get pushed to +logit_scale, background to -logit_scale, so SAM2
    treats it as a confident prior rather than a soft suggestion.
    """
    resized = cv2.resize(binary_mask.astype(np.float32), (low_res_size, low_res_size),
                          interpolation=cv2.INTER_NEAREST)
    logits = (resized - 0.5) * 2 * logit_scale
    return logits[None, :, :].astype(np.float32)


def run_sam2_on_combo_masks(image_rgb, combo_masks, combo_neg_points, checkpoint, model_cfg, device='cuda'):
    """
    Run SAM2 once per combo mask, using the mask as the primary prompt PLUS
    negative point prompts (from combo_neg_points) marking the adjacent
    lines and every spanned line's endpoints as background.

    Returns
    -------
    dict {(i, j): (prior_binary_mask, predicted_mask)}
    """
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_rgb)

    results = {}
    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if device == 'cuda'
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        for (i, j), binary_mask in combo_masks.items():
            mask_input = mask_to_sam2_input(binary_mask)
            neg_points = combo_neg_points.get((i, j), np.empty((0, 2)))

            if len(neg_points) > 0:
                point_coords = neg_points
                point_labels = np.zeros(len(neg_points), dtype=np.int32)  # all negative
            else:
                point_coords = None
                point_labels = None

            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=False,
            )
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            results[(i, j)] = (binary_mask, best_mask)

    return results


def compute_iou(mask_a, mask_b):
    """Intersection-over-union between two binary masks."""
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def deduplicate_by_prior_agreement(results, iou_threshold=0.9):
    """
    Group predicted masks that are near-duplicates of each other (IoU above
    `iou_threshold`), then keep only ONE result per group: whichever member's
    predicted mask has the highest IoU against its OWN prior (input polygon)
    mask -- i.e. whichever combo's prediction stuck closest to what was asked
    for, rather than trusting SAM2's own confidence score.

    Uses the FULL pairwise IoU matrix + union-find (connected components)
    rather than greedy first-match clustering. This guarantees any two masks
    with IoU >= threshold end up in the same group -- including transitively,
    through a chain of near-duplicates -- regardless of iteration order,
    which greedy clustering can miss or fragment.

    Returns
    -------
    dict, same shape as `results` but with duplicates collapsed.
    """
    keys = sorted(results.keys())
    n = len(keys)
    pred_masks = [np.asarray(results[k][1]).astype(bool) for k in keys]

    # full pairwise IoU matrix (n x n, symmetric, diagonal unused)
    iou_matrix = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            iou_val = compute_iou(pred_masks[a], pred_masks[b])
            iou_matrix[a, b] = iou_val
            iou_matrix[b, a] = iou_val

    # union-find so IoU-linked masks merge transitively, order-independent
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a in range(n):
        for b in range(a + 1, n):
            if iou_matrix[a, b] >= iou_threshold:
                union(a, b)

    clusters = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    deduped = {}
    for members in clusters.values():
        best_idx = None
        best_agreement = -1.0
        for idx in members:
            key = keys[idx]
            prior_mask, pred_mask = results[key]
            agreement = compute_iou(pred_mask, prior_mask)
            if agreement > best_agreement:
                best_agreement = agreement
                best_idx = idx
        best_key = keys[best_idx]
        deduped[best_key] = results[best_key]

    return deduped


def filter_results_by_coverage(results, min_fraction=0.01, max_fraction=0.5):
    """
    Drop any (i, j) result whose predicted mask covers less than
    `min_fraction` or more than `max_fraction` of the total image pixels --
    masks below min_fraction are usually degenerate/empty-ish predictions,
    and masks above max_fraction are almost always over-segmentations
    rather than a real bounded strip between the lines.
    """
    filtered = {}
    for key, (prior_mask, pred_mask) in results.items():
        if pred_mask is None:
            continue
        pred_mask_bool = np.asarray(pred_mask).astype(bool)
        coverage = np.count_nonzero(pred_mask_bool) / pred_mask_bool.size
        if min_fraction <= coverage <= max_fraction:
            filtered[key] = (prior_mask, pred_mask)
    return filtered


def plot_grid_dynamic(image_rgb, results, max_per_figure=100):
    """
    Plot all results in a roughly-square grid sized to fit however many
    combos there are. If there are more than `max_per_figure` results,
    splits across multiple figures so matplotlib doesn't choke on one huge
    grid -- with max_gap=10 across 32 lines you'll have ~265 combos total.
    """
    items = list(results.items())
    n_figures = math.ceil(len(items) / max_per_figure)

    for fig_idx in range(n_figures):
        chunk = items[fig_idx * max_per_figure: (fig_idx + 1) * max_per_figure]
        n = len(chunk)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
        axes = np.array(axes).reshape(-1)

        for k, ((i, j), (prior_mask, pred_mask)) in enumerate(chunk):
            ax = axes[k]
            ax.imshow(image_rgb, cmap='gray')
            if pred_mask is not None:
                overlay = np.ma.masked_where(pred_mask == 0, pred_mask)
                ax.imshow(overlay, cmap='autumn', alpha=0.5)
                coverage_pct = 100 * np.count_nonzero(pred_mask) / pred_mask.size
                ax.set_title(f"{i}-{j} ({coverage_pct:.0f}%)", fontsize=6)
            else:
                ax.set_title(f"{i}-{j}", fontsize=6)
            ax.axis('off')

        for k in range(len(chunk), len(axes)):
            axes[k].axis('off')

        plt.tight_layout()
        plt.show()


def sam2_main(number, clahe_image, smoothed_data):
    # make it easier to call from main file instead of changing both places
    lines_label_img = np.load(f'lines\\{number}.npy')  # labels 3..32 (30 interpolated lines)

    image_uint8 = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    # smoothed_data / lines_label_img can legitimately be a different (larger)
    # shape than the image, since lines are sometimes drawn outside the image
    # bounds to get better interpolation near the edges. That's fine --
    # generate_all_combo_masks always builds the final polygon canvas at
    # image_shape, and cv2.fillPoly clips any out-of-bounds points naturally.
    image_shape = image_rgb.shape[:2]

    checkpoint = str((Path(__file__).resolve().parent).parent / 'checkpoints' / 'sam2.1_hiera_large.pt')
    model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'

    combo_masks, combo_neg_points = generate_all_combo_masks(
        boundary_mask=smoothed_data, label_value1=1, label_value2=2,
        lines_label_img=lines_label_img, start_label=3, n_lines=30,
        shape=image_shape, min_gap=3, max_gap=10, points_per_adjacent_line=5,
    )
    print(f"Generated {len(combo_masks)} combo masks")

    results = run_sam2_on_combo_masks(
        image_rgb, combo_masks, combo_neg_points,
        checkpoint=checkpoint, model_cfg=model_cfg, device='cuda',
    )

    filtered_results = filter_results_by_coverage(results, min_fraction=0.01, max_fraction=0.1)
    print(f"Kept {len(filtered_results)} of {len(results)} results after coverage filter")
    #plot_grid_dynamic(image_rgb, filtered_results)

    deduped_results = deduplicate_by_prior_agreement(filtered_results, iou_threshold=0.85)
    print(f"Kept {len(deduped_results)} of {len(filtered_results)} results after deduplication")
    plot_grid_dynamic(image_rgb, deduped_results)


if __name__ == "__main__":
    number = '659'
    clahe_image = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\clahe\\{number}_clahe.npy')
    smoothed_data = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\bounds\\{number}_smoothed.npy')
    sam2_main(number=number, clahe_image=clahe_image, smoothed_data=smoothed_data)
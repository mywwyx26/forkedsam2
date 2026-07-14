"""
run_sam2_lines.py

Run SAM2 using MASK inputs instead of point prompts. For every pair of
lines (from the full set: original boundary line 1, the 30 interpolated
lines, and original boundary line 2) that are at most `max_gap` positions
apart, build a filled mask of the region between them (closing the polygon
by connecting the two lines' endpoints directly -- no curve fitting on the
sides), and use that filled region as a SAM2 mask prompt.

No point prompts (positive or negative) are used at this stage.
"""

import math
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from contextlib import nullcontext
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from pathlib import Path

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


def generate_all_combo_masks(boundary_mask, label_value1, label_value2,
                              lines_label_img, start_label, n_lines,
                              shape, max_gap=10):
    """
    Build a filled mask for every pair of lines (i, j) with 1 <= j - i <= max_gap,
    across the full stack of n_lines + 2 lines (boundary1, interpolated..., boundary2).

    Returns
    -------
    dict {(i, j): binary_mask}
    """
    n_total = n_lines + 2

    # cache ordered points per line index so each line is only extracted once
    line_points_cache = {
        idx: get_line_points(idx, boundary_mask, label_value1, label_value2,
                              lines_label_img, start_label, n_lines)
        for idx in range(n_total)
    }

    combo_masks = {}
    for i in range(n_total):
        j_max = min(i + max_gap, n_total - 1)
        for j in range(i + 1, j_max + 1):
            mask = build_between_mask(line_points_cache[i], line_points_cache[j], shape)
            if mask is not None:
                combo_masks[(i, j)] = mask

    return combo_masks


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


def run_sam2_on_combo_masks(image_rgb, combo_masks, checkpoint, model_cfg, device='cuda'):
    """
    Run SAM2 once per combo mask, using ONLY the mask as the prompt
    (no point or box prompts).

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
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                mask_input=mask_input,
                multimask_output=False,
            )
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            results[(i, j)] = (binary_mask, best_mask)

    return results


def filter_results_by_coverage(results, max_fraction=0.5):
    """
    Drop any (i, j) result whose predicted mask covers more than
    `max_fraction` of the total image pixels -- these are almost always
    over-segmentations rather than a real bounded strip between the lines.
    """
    filtered = {}
    for key, (prior_mask, pred_mask) in results.items():
        if pred_mask is None:
            continue
        pred_mask_bool = np.asarray(pred_mask).astype(bool)
        coverage = np.count_nonzero(pred_mask_bool) / pred_mask_bool.size
        if coverage <= max_fraction:
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


if __name__ == "__main__":
    number = '659'

    clahe_image = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\clahe\\{number}_clahe.npy')
    smoothed_data = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\bounds\\{number}.npy')  # labels 1, 2
    lines_label_img = np.load(f'lines\\{number}.npy')  # labels 3..32 (30 interpolated lines)

    image_uint8 = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    checkpoint = str((Path(__file__).resolve().parent).parent / 'checkpoints' / 'sam2.1_hiera_large.pt')
    model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'

    combo_masks = generate_all_combo_masks(
        boundary_mask=smoothed_data, label_value1=1, label_value2=2,
        lines_label_img=lines_label_img, start_label=3, n_lines=30,
        shape=smoothed_data.shape, max_gap=10,
    )
    print(f"Generated {len(combo_masks)} combo masks")

    results = run_sam2_on_combo_masks(
        image_rgb, combo_masks,
        checkpoint=checkpoint, model_cfg=model_cfg, device='cuda',
    )

    filtered_results = filter_results_by_coverage(results, max_fraction=0.2)
    print(f"Kept {len(filtered_results)} of {len(results)} results after coverage filter")

    plot_grid_dynamic(image_rgb, filtered_results)
"""
run_sam2_lines.py

Run SAM2 on an image, using points sampled along each interpolated line
(from interpolate_lines.py) as point prompts -- one SAM2 prediction per line.
Displays all resulting segmentations overlaid on the base image in a
5 rows x 6 cols grid (intended for n=30 interpolated lines).
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from contextlib import nullcontext
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def sample_points_from_line(lines_label_img, label_value, num_points=100):
    """
    Extract `num_points` evenly-spaced (x, y) points along a single labeled
    line in a rasterized label image, ordered along the line's path (not
    raw pixel scan order) so the spread is even along the curve rather than
    clustered.
    """
    ys, xs = np.where(lines_label_img == label_value)
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
    ordered_pts = pts[order]

    idx = np.linspace(0, len(ordered_pts) - 1, num_points).astype(int)
    return ordered_pts[idx]


def run_sam2_on_lines(image_rgb, lines_label_img, checkpoint, model_cfg,
                       start_label=3, n_lines=30, points_per_line=100,
                       device='cuda'):
    """
    Run SAM2 once per interpolated line, using sampled points from that line
    as positive point prompts.

    Returns
    -------
    list of (points, mask) tuples, one per line, in line order.
    `mask` is None if the line had no pixels (e.g. fully clipped out).
    """
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_rgb)

    results = []
    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if device == 'cuda'
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        for i in range(n_lines):
            label_value = start_label + i
            points = sample_points_from_line(lines_label_img, label_value, points_per_line)

            if len(points) == 0:
                results.append((points, None))
                continue

            point_labels = np.ones(len(points), dtype=np.int32)  # all positive prompts
            if len(points) >= 2:
                point_labels[0] = 0   # start of line -> negative prompt
                point_labels[-1] = 0  # end of line -> negative prompt
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=False,
            )
            best_mask = masks[0]  # multimask_output=False -> single mask
            results.append((points, best_mask))

    return results


def plot_grid(image_rgb, results, rows=5, cols=6):
    """
    Plot every line's segmentation overlaid on the base image in a
    rows x cols grid. Mask is drawn with masking (not opacity) so the
    underlying image still shows through everywhere the mask is 0.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten()

    for i, (points, mask) in enumerate(results):
        ax = axes[i]
        ax.imshow(image_rgb, cmap='gray')
        if mask is not None:
            mask_overlay = np.ma.masked_where(mask == 0, mask)
            ax.imshow(mask_overlay, cmap='autumn', alpha=0.5)
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], s=4, c='cyan')
        ax.set_title(f"line {i + 1}", fontsize=8)
        ax.axis('off')

    # hide any unused subplots if there are fewer results than grid cells
    for j in range(len(results), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    number = '659'

    clahe_image = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\clahe\\{number}_clahe.npy')
    lines_label_img = np.load(f'lines\\{number}.npy')

    # SAM2 expects a 3-channel uint8 image
    image_uint8 = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    checkpoint = str((Path(__file__).resolve().parent).parent / 'checkpoints' / 'sam2.1_hiera_large.pt')
    model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'

    results = run_sam2_on_lines(
        image_rgb, lines_label_img,
        checkpoint=checkpoint, model_cfg=model_cfg,
        start_label=3, n_lines=30, points_per_line=10,
        device='cuda',
    )

    plot_grid(image_rgb, results, rows=5, cols=6)
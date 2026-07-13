"""
interpolate_lines.py

Generate N lines interpolated between two curved lines, such that:
  - each generated line is "parallel" to the two input curves (same arc-length
    parameterization), not just linearly interpolated in raw pixel index
  - at x% of the way along BOTH input curves, the corresponding point on
    intermediate line i is y% of the way from curve 1 to curve 2, where
    y = i / (n + 1)

"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def order_points_by_path(mask, label_value):
    """
    Extract points for a given label and order them by walking along the
    skeleton path (nearest-neighbor traversal from one endpoint to the other).
    Same approach as used for smoothing -- reused here so this file can be
    used standalone on a raw label mask.
    """
    binary = (mask == label_value)
    skel = skeletonize(binary)
    ys, xs = np.where(skel)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)

    tree = cKDTree(pts)
    neighbor_counts = tree.query_ball_point(pts, r=1.5, return_length=True)
    start_idx = np.argmin(neighbor_counts)

    visited = np.zeros(len(pts), dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    current = start_idx
    for _ in range(len(pts) - 1):
        dists, idxs = tree.query(pts[current], k=len(pts))
        next_idx = next(i for i in idxs if not visited[i])
        order.append(next_idx)
        visited[next_idx] = True
        current = next_idx

    return pts[order]


def resample_curve_by_arclength(points, num_samples=500):
    """
    Resample an ordered set of (x, y) points so they are evenly spaced by
    arc-length PERCENTAGE along the curve (0% to 100%), rather than by raw
    pixel index. This is what lets point i on curve1 and point i on curve2
    both represent "the same x% along the line" even if the two curves have
    different pixel lengths or uneven point spacing.
    """
    points = np.asarray(points, dtype=np.float64)
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]
    if total_length == 0:
        raise ValueError("Curve has zero length -- check input points")

    percent_along = cumulative / total_length  # 0.0 to 1.0

    target_percent = np.linspace(0.0, 1.0, num_samples)
    x_resampled = np.interp(target_percent, percent_along, points[:, 0])
    y_resampled = np.interp(target_percent, percent_along, points[:, 1])
    return np.stack([x_resampled, y_resampled], axis=1)


def generate_intermediate_lines(line1_pts, line2_pts, n, num_samples=500):
    """
    Generate n lines interpolated between line1_pts and line2_pts.

    For a point at x% along curve1 and the corresponding point at x% along
    curve2, intermediate line i sits y = i/(n+1) percent of the way from
    curve1 to curve2 at that SAME x. Moving along the line changes x while
    y (the 0..1 blend fraction) stays fixed for that line -- exactly your spec.

    Parameters
    ----------
    line1_pts : array-like, shape (M, 2)
        Ordered (x, y) points along the first curve.
    line2_pts : array-like, shape (K, 2)
        Ordered (x, y) points along the second curve.
        IMPORTANT: line1_pts and line2_pts must be ordered in the SAME
        direction (e.g. both top-to-bottom). If the generated lines look
        twisted/crossed, reverse one of the point arrays with points[::-1].
    n : int
        Number of intermediate lines to generate (must be >= 1).
    num_samples : int
        Number of points representing each curve/generated line.

    Returns
    -------
    list of np.ndarray, each shape (num_samples, 2)
        n intermediate lines, ordered from closest-to-line1 to closest-to-line2.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    curve1 = resample_curve_by_arclength(line1_pts, num_samples)
    curve2 = resample_curve_by_arclength(line2_pts, num_samples)

    intermediate_lines = []
    for i in range(1, n + 1):
        y = i / (n + 1)
        interp_curve = (1 - y) * curve1 + y * curve2
        intermediate_lines.append(interp_curve)

    return intermediate_lines


def rasterize_lines(number, lines, shape, start_label=3, thickness=1):
    """
    Draw a list of (x, y) point-array lines into a single label image.

    Parameters
    ----------
    lines : list of np.ndarray, each shape (num_points, 2)
    shape : tuple (height, width)
    start_label : int
        Label value assigned to the first line; subsequent lines increment
        by 1 (so they don't collide with your original label values, e.g.
        if line1/line2 are labels 1 and 2, start_label=3 avoids overlap).
    thickness : int
        Line thickness in pixels.

    Returns
    -------
    np.ndarray, shape `shape`, dtype uint16
        Label image with each interpolated line drawn as its own label value.
    """
    canvas = np.zeros(shape, dtype=np.uint16)
    for i, pts in enumerate(lines):
        label_value = start_label + i
        int_pts = pts.astype(np.int32)
        cv2.polylines(canvas, [int_pts], isClosed=False, color=int(label_value), thickness=thickness)
    np.save(f'lines\\{number}.npy', canvas)


if __name__ == "__main__":
    number = '746' # change number for each file, but should find a better way to do this
    clahe_image = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\clahe\\{number}_clahe.npy')
    smoothed_data = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\bounds\\{number}.npy')

    label_value1, label_value2 = 1, 2
    line1_pts = order_points_by_path(smoothed_data, label_value1)
    line2_pts = order_points_by_path(smoothed_data, label_value2)

    n = 30
    intermediate_lines = generate_intermediate_lines(line1_pts, line2_pts, n)
    rasterize_lines(number, intermediate_lines, smoothed_data.shape) # np.save

    # Draw everything for a quick visual check
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(clahe_image, cmap='gray')
    ax.plot(line1_pts[:, 0], line1_pts[:, 1], color='cyan', linewidth=1.5, label='line 1')
    ax.plot(line2_pts[:, 0], line2_pts[:, 1], color='yellow', linewidth=1.5, label='line 2')
    for idx, curve in enumerate(intermediate_lines):
        ax.plot(curve[:, 0], curve[:, 1], color='lime', linewidth=1.0)
    ax.axis('off')
    ax.legend(loc='upper right')
    plt.show()
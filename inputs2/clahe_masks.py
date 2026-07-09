import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops

def preprocess(file):
    fig, axes = plt.subplots(2,2)
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].axis('off')

    # tried a few numbers and have decided that these are best
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image_8bit)
    axes[0,1].imshow(clahe_image, cmap='gray')
    axes[0,1].axis('off')

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    clahe_image = clahe.apply(image_8bit)
    axes[1,0].imshow(clahe_image, cmap='gray')
    axes[1,0].axis('off')

    blurred = ndimage.gaussian_filter(clahe_image, sigma=3)
    axes[1,1].imshow(blurred, cmap='gray')
    axes[1,1].axis('off')

    plt.show()
    return blurred
    

def keep_n_largest_labels(labels, n):
    if labels.max() <= n:
        return labels
    props = regionprops(labels)
    props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
    keep = {p.label for p in props_sorted[:n]}
    relabel_map = {old: new for new, old in enumerate(sorted(keep), start=1)}
    out = np.where(np.isin(labels, list(keep)), labels, 0)
    return np.vectorize(lambda v: relabel_map.get(v, 0))(out)


def rotate_mask(mask, angle_deg):
    h, w = mask.shape
    diag = int(np.ceil(np.hypot(h, w))) + 4
    canvas = np.zeros((diag, diag), dtype=mask.dtype)
    y0, x0 = (diag - h) // 2, (diag - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = mask
    center = (diag / 2, diag / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_NEAREST)
    return rotated, M, (y0, x0), diag


def _contiguous_runs(idx):
    if len(idx) == 0:
        return []
    runs = []
    start = idx[0]
    prev = idx[0]
    for v in idx[1:]:
        if v == prev + 1:
            prev = v
        else:
            runs.append((start, prev))
            start = v
            prev = v
    runs.append((start, prev))
    return runs


def _offset_line_for_rotation(rotated_mask, frac, max_jump):
    pts = []
    last_y = None
    for x in range(rotated_mask.shape[1]):
        rows = np.where(rotated_mask[:, x] > 0)[0]
        if len(rows) == 0:
            continue
        runs = _contiguous_runs(rows)
        if last_y is None:
            top, bottom = max(runs, key=lambda r: r[1] - r[0])
        else:
            candidates = [(t, b, abs((t + (b - t) * frac) - last_y)) for t, b in runs]
            top, bottom, _ = min(candidates, key=lambda c: c[2])
        y = top + (bottom - top) * frac
        if last_y is not None and abs(y - last_y) > max_jump:
            continue
        pts.append((x, y))
        last_y = y
    return np.array(pts)


def _arc_length(pts):
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return np.sum(np.sqrt((d ** 2).sum(axis=1)))


def find_best_rotation_multi(mask, fracs, angle_step=2, max_jump=25):
    best_total, best_theta = -1, None
    for theta in np.arange(0, 180, angle_step):
        rotated, _, _, _ = rotate_mask(mask, theta)
        total = 0.0
        for f in fracs:
            pts = _offset_line_for_rotation(rotated, f, max_jump)
            total += _arc_length(pts)
        if total > best_total:
            best_total, best_theta = total, theta
    return best_theta, best_total


def smooth_line(pts, window=21, polyorder=3):
    if len(pts) < polyorder + 2:
        return pts
    w = min(window, len(pts) - (1 - len(pts) % 2))
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        return pts
    out = pts.copy()
    out[:, 0] = savgol_filter(pts[:, 0], w, polyorder)
    out[:, 1] = savgol_filter(pts[:, 1], w, polyorder)
    return out


def lines_at_best_rotation(mask, percentages, angle_step=2, max_jump=25):
    fracs = np.cumsum(percentages) / 100.0
    theta, total = find_best_rotation_multi(mask, fracs, angle_step, max_jump)
    theta_final = (theta + 180) % 360

    rotated, M, (y0, x0), diag = rotate_mask(mask, theta_final)
    Minv = cv2.invertAffineTransform(M)

    lines_original = []
    for f in fracs:
        pts = _offset_line_for_rotation(rotated, f, max_jump)
        if len(pts) < 2:
            lines_original.append(np.empty((0, 2)))
            continue
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_canvas = (Minv @ pts_h.T).T
        pts_original = pts_canvas - np.array([x0, y0])
        lines_original.append(smooth_line(pts_original))
    return theta_final, total, lines_original


def boundaries_for_rotation(rotated_mask, max_jump):
    entries = []
    last_mid = None
    for x in range(rotated_mask.shape[1]):
        rows = np.where(rotated_mask[:, x] > 0)[0]
        if len(rows) == 0:
            continue
        runs = _contiguous_runs(rows)
        if last_mid is None:
            top, bottom = max(runs, key=lambda r: r[1] - r[0])
        else:
            candidates = [(t, b, abs((t + b) / 2 - last_mid)) for t, b in runs]
            top, bottom, _ = min(candidates, key=lambda c: c[2])
        mid = (top + bottom) / 2
        if last_mid is not None and abs(mid - last_mid) > max_jump:
            continue
        entries.append((x, top, bottom))
        last_mid = mid
    return entries


def build_layer_masks(mask, percentages, theta_final, max_jump=25):
    h, w = mask.shape
    rotated, M, (y0, x0), diag = rotate_mask(mask, theta_final)
    entries = boundaries_for_rotation(rotated, max_jump)
    edges = np.concatenate([[0.0], np.cumsum(percentages) / 100.0])
    n_layers = len(percentages)

    rotated_layers = [np.zeros((diag, diag), dtype=np.uint8) for _ in range(n_layers)]
    for x, top, bottom in entries:
        bounds = top + edges * (bottom - top)
        for i in range(n_layers):
            r0 = int(round(bounds[i]))
            r1 = int(round(bounds[i + 1]))
            r0, r1 = min(r0, r1), max(r0, r1)
            r1 = max(r1, r0 + 1)
            rotated_layers[i][r0:r1, x] = 1

    layer_masks_original = []
    for i in range(n_layers):
        unrotated = cv2.warpAffine(rotated_layers[i], M, (diag, diag),
                                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        cropped = unrotated[y0:y0 + h, x0:x0 + w]
        layer_masks_original.append(cropped)
    return layer_masks_original


def draw_blobs_and_lines(ax, background_img, labels_rounded, results, line_colors):
    ax.imshow(background_img, cmap='gray')
    for lbl_id, info in results.items():
        blob_mask = (labels_rounded == lbl_id).astype(np.uint8)
        color = line_colors[info["rank"] % len(line_colors)]
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color=color, linewidth=1.2)
        for pts in info["lines"]:
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.0)
    ax.axis('off')


def main(clahe_img, name, output_root='outputs'):
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    binarized = np.where(clahe_img > np.mean(clahe_img) * 0.8, 1, 0).astype(np.uint8)

    min_dot_area = 60
    lbl_speckles = label(binarized)
    sizes = ndimage.sum(binarized, lbl_speckles, range(1, lbl_speckles.max() + 1))
    dot_mask = np.isin(lbl_speckles, np.where(sizes >= min_dot_area)[0] + 1)
    binarized_clean = dot_mask.astype(np.uint8)

    binarized_filled = ndimage.binary_fill_holes(binarized_clean).astype(np.uint8)

    distance = ndimage.distance_transform_edt(binarized_filled)
    coordinates = peak_local_max(
        distance, min_distance=int(0.1 * np.mean(np.shape(clahe_img))), labels=binarized_filled
    )
    seed_mask = np.zeros(distance.shape, dtype=bool)
    seed_mask[tuple(coordinates.T)] = True
    markers, _ = ndimage.label(seed_mask)

    labels = watershed(-distance, markers, mask=binarized_filled)
    labels = keep_n_largest_labels(labels, n=3)
    n_found = labels.max()
    if n_found < 3:
        print(f"  [{name}] WARNING: only found {n_found} region(s) before rounding.")

    round_kernel_size = 1
    smooth_sigma = 4.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round_kernel_size, round_kernel_size))

    rounded_masks = {}
    for lbl_id in range(1, n_found + 1):
        m = (labels == lbl_id).astype(np.uint8)
        m = ndimage.binary_opening(m, structure=kernel).astype(np.uint8)
        m = ndimage.binary_closing(m, structure=kernel).astype(np.uint8)
        if smooth_sigma > 0:
            blurred = ndimage.gaussian_filter(m.astype(np.float32), sigma=smooth_sigma)
            m = (blurred > 0.5).astype(np.uint8)
        rounded_masks[lbl_id] = m

    union_rounded = np.zeros_like(binarized_filled)
    for m in rounded_masks.values():
        union_rounded = np.maximum(union_rounded, m)

    distance_rounded = ndimage.distance_transform_edt(union_rounded)
    markers_final = np.zeros_like(markers)
    for lbl_id in range(1, n_found + 1):
        m = rounded_masks[lbl_id]
        if m.sum() == 0:
            continue
        dist_m = ndimage.distance_transform_edt(m)
        py, px = np.unravel_index(np.argmax(dist_m), dist_m.shape)
        markers_final[py, px] = lbl_id
    labels_rounded = watershed(-distance_rounded, markers_final, mask=union_rounded)
    labels_rounded = keep_n_largest_labels(labels_rounded, n=3)
    final_binary_separated = labels_rounded > 0

    # ------------------------------------------------------------------
    # Diagnostic figure: raw binarized / split before rounding /
    # rounded union / final 3 labels rounded -- same style as the
    # other script's 2x2 diagnostic plot.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(binarized, cmap='gray')
    axes[0, 0].set_title('raw binarized')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(labels, cmap='jet')
    axes[0, 1].set_title('split BEFORE rounding')
    axes[0, 1].axis('off')
    axes[1, 0].imshow(union_rounded, cmap='gray')
    axes[1, 0].set_title('rounded (union)')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(labels_rounded, cmap='jet')
    axes[1, 1].set_title('final 3 labels, rounded')
    axes[1, 1].axis('off')
    fig.suptitle(name)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, 'diagnostic.png'), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Simple green mask overlay -- same style as the other script's
    # "overlay on everyone" plot, just for this single image.
    # ------------------------------------------------------------------
    cmap_green = LinearSegmentedColormap.from_list('cg', ['#000000', '#00FF00'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(clahe_img, cmap='gray')
    axes[0].set_title('clahe image')
    axes[0].axis('off')
    axes[1].imshow(clahe_img, cmap='gray')
    axes[1].imshow(final_binary_separated, cmap=cmap_green, alpha=0.3)
    axes[1].set_title('neuropil mask overlay (3 rounded blobs)')
    axes[1].axis('off')
    fig.suptitle(name)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, 'mask_overlay.png'), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Curved percentage-offset lines + blob outlines
    # ------------------------------------------------------------------
    percentages_by_rank = [
        [10, 10, 10, 5, 8, 11, 12, 7, 18, 9],
        [10, 6, 7, 29, 23, 25],
        [25, 25, 25, 25],
    ]

    areas = {lbl_id: (labels_rounded == lbl_id).sum() for lbl_id in range(1, labels_rounded.max() + 1)}
    ranked_label_ids = sorted(areas, key=lambda k: areas[k], reverse=True)
    print(f"  [{name}] blob areas: {areas}  ranked (largest->smallest): {ranked_label_ids}")

    line_colors = ['#00FF00', '#FF00FF', '#00CFFF']
    results = {}
    layer_masks_by_blob = {}
    for rank, lbl_id in enumerate(ranked_label_ids):
        if rank >= len(percentages_by_rank):
            break
        percentages = percentages_by_rank[rank]
        blob_mask = (labels_rounded == lbl_id).astype(np.uint8)

        theta_final, total, lines_original = lines_at_best_rotation(
            blob_mask, percentages, angle_step=2, max_jump=25
        )
        results[lbl_id] = {"theta": theta_final, "total_length": total, "lines": lines_original, "rank": rank}
        layer_masks_by_blob[lbl_id] = build_layer_masks(blob_mask, percentages, theta_final, max_jump=25)

        print(f"  [{name}] blob label={lbl_id} (rank {rank}, area={areas[lbl_id]}): "
              f"theta={theta_final}, combined length={total:.1f}, "
              f"{len(lines_original)} lines")

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_blobs_and_lines(ax, clahe_img, labels_rounded, results, line_colors)
    ax.set_title(name)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, 'blobs_and_lines.png'), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save per-layer masks for this image into its own subfolder --
    # important: this used to be a single shared 'layer_masks' folder
    # that every file in the loop would overwrite. Now nested under
    # out_dir so each input file keeps its own set.
    # ------------------------------------------------------------------
    layer_dir = os.path.join(out_dir, 'layer_masks')
    os.makedirs(layer_dir, exist_ok=True)
    blob_names = {0: 'largest', 1: 'second', 2: 'smallest'}
    for lbl_id, layer_masks in layer_masks_by_blob.items():
        rank = results[lbl_id]["rank"]
        blob_name = blob_names.get(rank, f"rank{rank}")
        for i, m in enumerate(layer_masks):
            out_path = os.path.join(layer_dir, f"{blob_name}_layer{i:02d}.png")
            cv2.imwrite(out_path, (m * 255).astype(np.uint8))
            np.save(os.path.join(layer_dir, f"{blob_name}_layer{i:02d}.npy"), m.astype(np.uint8))

    print(f"  [{name}] saved diagnostic.png, mask_overlay.png, blobs_and_lines.png, "
          f"and layer_masks/ into {out_dir}/")


if __name__ == '__main__':
    input_folder = 'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs2\\original'
    output_root = 'outputs'
    inputs = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) if f.endswith(".tif")]

    for file in inputs:
        name = os.path.splitext(os.path.basename(file))[0]
        print(f"Processing {name}...")

        image = preprocess(file)
        main(image, name, output_root)

    print(f"\nAll files processed. Results in ./{output_root}/<filename>/")
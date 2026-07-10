import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.signal import savgol_filter

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# config and predictor
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = str(SCRIPT_DIR.parent / 'checkpoints' / 'sam2.1_hiera_large.pt')
MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PERCENTAGES = [10, 10, 10, 5, 8, 11, 12, 7, 18, 9]

sam2_model = build_sam2(MODEL_CONFIG, CHECKPOINT_PATH, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

# get the 10 layers (same as combined3.py)
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

# sam2 helpers
def mask_array_to_sam2_input(mask_arr, logit_scale=20.0):
    m = mask_arr.astype(np.float32)
    if m.max() > 1:
        m = m / 255.0
    resized = cv2.resize(m, (256, 256), interpolation=cv2.INTER_LINEAR)
    logit_mask = (resized - 0.5) * logit_scale
    return logit_mask[None, :, :]


def mask_centroid_from_array(mask_arr):
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        return None
    return np.array([[xs.mean(), ys.mean()]]), np.array([1])


def run_sam2_on_mask(mask_arr):
    """Returns (bool_mask, score) or (None, None) if the input mask is empty."""
    centroid = mask_centroid_from_array(mask_arr)
    if centroid is None:
        return None, None
    point_coords, point_labels = centroid
    mask_input = mask_array_to_sam2_input(mask_arr)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=mask_input,
        multimask_output=False,
    )
    return masks[0].astype(bool), float(scores[0])

# main loop
input_folder = 'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs3\\original'
inputs = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) if f.endswith(".tif")]
output = 'layer_masks'
os.makedirs(output, exist_ok=True)

for i in range(len(inputs)):
    filename = os.path.splitext(os.path.splitext(os.path.basename(inputs[i]))[0])[0]
    clahe_image = np.load(os.path.join('clahe',f'{filename}_clahe.npy'))
    print(f"\nProcessing {filename}...")

    medulla_mask_path = os.path.join('medulla_masks',f'{filename}_medulla_mask.npy')
    if not os.path.exists(medulla_mask_path):
        print(f"  skipping -- no medulla mask found at {medulla_mask_path}")
        continue
    medulla_mask = np.load(medulla_mask_path)

    clahe_image_path = os.path.join('clahe',f'{filename}_clahe.npy')
    if not os.path.exists(clahe_image_path):
        print(f"  skipping -- could not find source image at {clahe_image_path}")
        continue
    clahe_image = np.load(clahe_image_path)
    predictor.set_image(cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB))

    sam2_medulla_mask, medulla_score = run_sam2_on_mask(medulla_mask)
    if sam2_medulla_mask is None:
        print(f"  skipping -- medulla mask was empty")
        continue
    print(f"  SAM2 medulla mask: score={medulla_score:.3f}")

    sam2_medulla_uint8 = sam2_medulla_mask.astype(np.uint8)
    theta_final, total_len, _ = lines_at_best_rotation(sam2_medulla_uint8, PERCENTAGES, angle_step=2, max_jump=25)
    hardcoded_layers = build_layer_masks(sam2_medulla_uint8, PERCENTAGES, theta_final, max_jump=25)
    print(f"  regenerated {len(hardcoded_layers)} layers from SAM2 medulla mask "
          f"(theta={theta_final}, combined length={total_len:.1f})")

    sam2_layers = []
    for lm in hardcoded_layers:
        sm, sc = run_sam2_on_mask(lm)
        sam2_layers.append((sm, sc))

    file_out_dir = os.path.join(output, filename)
    os.makedirs(file_out_dir, exist_ok=True)
    np.save(os.path.join(file_out_dir, "sam2_medulla_mask.npy"), sam2_medulla_mask.astype(np.uint8))
    for i, (lm, (sm, sc)) in enumerate(zip(hardcoded_layers, sam2_layers)):
        np.save(os.path.join(file_out_dir, f"hardcoded_layer{i:02d}.npy"), lm.astype(np.uint8))
        if sm is not None:
            np.save(os.path.join(file_out_dir, f"sam2_layer{i:02d}.npy"), sm.astype(np.uint8))

    n_layers = len(hardcoded_layers)
    fig, axes = plt.subplots(5,5, figsize=(12,12))

    axes[0, 0].imshow(clahe_image, cmap='gray')
    axes[0, 0].set_title('clahe image', fontsize=9)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(clahe_image, cmap='gray')
    axes[0, 1].imshow(np.ma.masked_where(~sam2_medulla_mask, sam2_medulla_mask),
                       cmap='autumn', alpha=0.5)
    axes[0, 1].set_title(f'SAM2 medulla mask\nscore={medulla_score:.2f}', fontsize=9)
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    axes[0, 4].axis('off')

    count = 5
    for i, (lm, (sm, sc)) in enumerate(zip(hardcoded_layers, sam2_layers)):
        row = count // 5
        col = count % 5
        axes[row, col].imshow(clahe_image, cmap='gray')
        lm_bool = lm.astype(bool)
        axes[row, col].imshow(np.ma.masked_where(~lm_bool, lm_bool), cmap='cool', alpha=0.5)
        axes[row, col].set_title(f'hardcoded layer {i:02d}', fontsize=8)
        axes[row, col].axis('off')

        axes[row+2, col].imshow(clahe_image, cmap='gray')
        if sm is not None:
            axes[row+2, col].imshow(np.ma.masked_where(~sm, sm), cmap='autumn', alpha=0.5)
            axes[row+2, col].set_title(f'SAM2 layer {i:02d}\nscore={sc:.2f}', fontsize=8)
        else:
            axes[row+2, col].set_title(f'SAM2 layer {i:02d}\n(empty)', fontsize=8)
        axes[row+2, col].axis('off')
        count = count + 1

    fig.suptitle(filename, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(file_out_dir, f"{filename}_medulla_and_layers.svg"), dpi=120)
    plt.show()
    plt.close(fig)
    print(f"  saved outputs to {file_out_dir}/")

print(f"\nDone. All results in ./{output}/<filename>/")
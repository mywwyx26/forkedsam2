"""
run_sam2_lines.py

Run SAM2 using MASK inputs plus NEGATIVE POINT prompts. For every pair of
INTERIOR interpolated lines (i, j) at most `max_gap` positions apart, a filled
mask of the region between them is used as a SAM2 mask prompt, with negative
points sampled from the lines just outside the mask on each side.
No positive point prompts are used -- the mask_input itself is the positive prior.

This file only holds the SAM2 inference and the top-level driver. The rest lives in:
    line_utils.py      -> building combo masks / negative points from the line stack
    mask_selection.py  -> coverage filter + NMS
    sam2_plots.py      -> result grid plot
    activity.py        -> dF/F traces from the saved masks

The selected masks are saved to outputs3/data/{number}_masks.npz for activity.py.
"""

import os
import numpy as np
import cv2
import torch
from contextlib import nullcontext
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from line_utils import generate_all_combo_masks
from mask_selection import filter_results_by_coverage, non_max_suppression_masks
from sam2_plots import plot_grid_dynamic

BASE = Path(__file__).resolve().parent


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
    dict {(i, j): (prior_binary_mask, predicted_mask, score)}
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
            best_score = float(scores[best_idx])
            results[(i, j)] = (binary_mask, best_mask, best_score)

    return results


def save_masks(number, results, out_dir=os.path.join('outputs3', 'data')):
    """
    Save the final predicted masks so activity.py can use them.

    File: {out_dir}/{number}_masks.npz with
        masks  : (N, H, W) bool   predicted masks
        keys   : (N, 2) int       the (i, j) line pair each mask came from
        scores : (N,) float       SAM2 confidence for each mask
    """
    if not results:
        print(f"no masks to save")
        return None
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted(results.keys())
    masks = np.stack([np.asarray(results[k][1]).astype(bool) for k in keys])
    scores = np.array([results[k][2] for k in keys], dtype=np.float32)
    path = os.path.join(out_dir, f'{number}_masks.npz')
    np.savez_compressed(path, masks=masks, keys=np.array(keys, dtype=np.int32), scores=scores)
    print(f"saved {len(keys)} masks -> {path}")
    return path


def sam2_main(number, clahe_image, smoothed_data, top_k=12):
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

    nms_results = non_max_suppression_masks(filtered_results, top_k=top_k, metric='iou')
    print(f"Kept {len(nms_results)} of {len(filtered_results)} results after NMS")

    plot_grid_dynamic(number, image_rgb, nms_results)
    save_masks(number, nms_results)
    return nms_results


if __name__ == "__main__":
    os.chdir(BASE)  # relative folders (bounds, lines, outputs3) resolve next to this file
    number = '659'
    clahe_image = np.load(BASE / 'clahe' / f'{number}_clahe.npy')
    smoothed_data = np.load(BASE / 'bounds' / f'{number}.npy')
    sam2_main(number=number, clahe_image=clahe_image, smoothed_data=smoothed_data)
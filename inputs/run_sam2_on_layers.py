"""
run_sam2_on_layers.py

Takes the per-layer binary masks saved by the neuropil segmentation
script (in ./layer_masks/) and feeds each one into SAM2 as a mask
prompt -- run separately over each of the 7 individual input frames
(NOT the averaged/combined image, since averaging smooths away the
texture/edge detail SAM2 needs to place layer boundaries well).
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = str(SCRIPT_DIR.parent / "checkpoints" / "sam2.1_hiera_large.pt")
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# The 7 individual frames, NOT the AVG/combined image -- the masks
# themselves (layer_masks/) were derived from the combined image's
# registration, but running SAM2 itself on each sharper individual
# frame should find the layer boundaries better than on the blurred
# average.
IMAGE_FILES = [
    'inputs_registered00000000.tif',
    'inputs_registered00000001.tif',
    'inputs_registered00000002.tif',
    'inputs_registered00000003.tif',
    'inputs_registered00000004.tif',
    'inputs_registered00000005.tif',
    'inputs_registered00000006.tif',
]

LAYER_MASK_DIR = "layer_masks"
OUTPUT_DIR = "sam2_outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------
# 2. Build SAM2 predictor (once, reused across all images/layers)
# ---------------------------------------------------------------
sam2_model = build_sam2(MODEL_CONFIG, CHECKPOINT_PATH, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)


# ---------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------
def load_as_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        norm = (rgb.astype(np.float32) - rgb.min())
        norm = norm / max(norm.max(), 1e-8)
        rgb = (norm * 255).astype(np.uint8)
    return img, rgb  # (original grayscale/native, RGB uint8 for SAM2)


def mask_png_to_sam2_input(mask_path, logit_scale=20.0):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    m_float = m.astype(np.float32) / 255.0
    resized = cv2.resize(m_float, (256, 256), interpolation=cv2.INTER_LINEAR)
    logit_mask = (resized - 0.5) * logit_scale
    return logit_mask[None, :, :]


def mask_centroid(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(m > 127)
    if len(xs) == 0:
        return None
    return np.array([[xs.mean(), ys.mean()]]), np.array([1])


layer_paths = sorted(glob.glob(os.path.join(LAYER_MASK_DIR, "*.png")))
print(f"Found {len(layer_paths)} layer masks in {LAYER_MASK_DIR}/")
print(f"Running SAM2 over {len(IMAGE_FILES)} individual input frames.")

# color per blob (matches the earlier segmentation script's line_colors)
blob_colors = {"largest": (1.0, 0.0, 0.0), "second": (0.0, 1.0, 0.0), "smallest": (0.0, 0.6, 1.0)}


def blob_name_from_layer(layer_name):
    return layer_name.split("_layer")[0]


# ---------------------------------------------------------------
# 4. Run SAM2 on every (image, layer mask) pair
# ---------------------------------------------------------------
results_by_image = {}  # image_file -> {layer_name: {"mask":..., "score":...}}

for image_path in IMAGE_FILES:
    image_gray, image_rgb = load_as_rgb(image_path)
    predictor.set_image(image_rgb)

    image_base = os.path.splitext(os.path.basename(image_path))[0]
    results_by_image[image_path] = {"gray": image_gray, "layers": {}}

    for layer_path in layer_paths:
        layer_name = os.path.splitext(os.path.basename(layer_path))[0]

        mask_input = mask_png_to_sam2_input(layer_path)
        centroid = mask_centroid(layer_path)
        if centroid is None:
            continue
        point_coords, point_labels = centroid

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=False,
        )

        results_by_image[image_path]["layers"][layer_name] = {
            "mask": masks[0].astype(bool),
            "score": float(scores[0]),
        }

        out_path = os.path.join(OUTPUT_DIR, f"{image_base}_{layer_name}_sam2.png")
        cv2.imwrite(out_path, (masks[0] * 255).astype(np.uint8))

    print(f"  {image_base}: done ({len(results_by_image[image_path]['layers'])} layers)")


# ---------------------------------------------------------------
# 5. Plot: one FIGURE per input frame, with one PANEL per layer
#    (the 20-panel style from before), saved into its own file.
# ---------------------------------------------------------------
figures_dir = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(figures_dir, exist_ok=True)

for image_path in IMAGE_FILES:
    image_base = os.path.splitext(os.path.basename(image_path))[0]
    info = results_by_image[image_path]
    layer_items = list(info["layers"].items())
    n = len(layer_items)
    if n == 0:
        continue

    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    for i, (layer_name, layer_info) in enumerate(layer_items):
        ax = axes[i // cols, i % cols]
        ax.imshow(info["gray"], cmap='gray')
        mask_bool = layer_info["mask"]
        ax.imshow(np.ma.masked_where(~mask_bool, mask_bool), cmap='autumn', alpha=0.5)
        ax.set_title(f"{layer_name}\nscore={layer_info['score']:.2f}", fontsize=8)
        ax.axis('off')

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis('off')

    fig.suptitle(image_base, fontsize=12)
    plt.tight_layout()
    out_fig_path = os.path.join(figures_dir, f"{image_base}_all_layers.svg")
    plt.savefig(out_fig_path, dpi=120)
    plt.close(fig)  # close each one so 7 figures don't all stay open in memory
    print(f"  saved {out_fig_path}")

print(f"\nDone. Per-layer masks in ./{OUTPUT_DIR}/, per-frame figures in ./{figures_dir}/")
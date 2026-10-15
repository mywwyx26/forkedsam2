"""
activity.py

Compute and plot dF/F activity for each SAM2 layer mask.

Takes the masks saved by run_sam2_lines.py (outputs3/data/{number}_masks.npz) and the
matching raw recording (TSeries-...-{number}.companion_registered.tif), and for
each mask:
    1. trace[t] = median fluorescence of the pixels inside the mask at frame t
    2. F0       = median of that trace over time
    3. dF/F     = (trace - F0) / F0
which is the same calculation findrois.py uses per ROI (median fluorescence,
median baseline), just applied to the layer masks instead of ROIs.

Outputs:
    outputs3/{number}_deltaf.svg       one stacked subplot of dF/F over time per mask
    outputs3/data/{number}_deltaf.npz  deltaf (N, T), time t (T,), keys (N, 2), so you don't
                         have to reload the (slow) video to re-plot
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tifffile

SAMPLE_RATE_HZ = 3.75  # all recordings were captured at 3.75 Hz
DATA_DIR = os.path.join('outputs3', 'data')  # masks in, dF/F .npz out
VIDEO_DIR = 'C:\\Users\\megan\\flies\\sams\\forkedsam2\\run_flies\\videos'


def find_video(number, video_dir=VIDEO_DIR):
    """Locate TSeries-<date>-<number>.companion_registered.tif without hardcoding the date."""
    pattern = os.path.join(video_dir, f'TSeries-*-{number}.companion_registered.tif')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No video found matching {pattern}')
    if len(matches) > 1:
        print(f'multiple videos match, using {matches[0]}')
    return matches[0]


def load_masks(number, masks_dir=DATA_DIR):
    """Load masks saved by run_sam2_lines.save_masks, ordered by position in the line stack."""
    d = np.load(os.path.join(masks_dir, f'{number}_masks.npz'))
    masks, keys, scores = d['masks'], d['keys'], d['scores']
    order = np.argsort(keys.sum(axis=1), kind='stable')  # (i + j) = layer centre, top to bottom
    return masks[order], keys[order], scores[order]


def compute_deltaf(data, masks):
    """
    data  : (T, H, W) recording
    masks : (N, H, W) bool
    Returns deltaf (N, T). Masks with no pixels, or F0 <= 0, come back as NaN.
    """
    n_frames = data.shape[0]
    traces = np.full((len(masks), n_frames), np.nan)
    for m, mask in enumerate(masks):
        if mask.any():
            traces[m] = np.median(data[:, mask], axis=1)  # median across the mask's pixels, per frame

    f0 = np.median(traces, axis=1, keepdims=True)
    f0 = np.where(f0 > 0, f0, np.nan)
    return (traces - f0) / f0


def plot_deltaf(number, deltaf, keys, output_dir='outputs3'):
    """One stacked subplot per mask, same layout/colors/labelling as findrois' cluster plot."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(deltaf)
    t = np.arange(deltaf.shape[1]) / SAMPLE_RATE_HZ

    set1 = [plt.cm.Set1(i) for i in range(8)]  # last of 9 is gray, skip it
    set3 = [plt.cm.Set3(i) for i in range(12)]
    colors = set1 + set3

    fig, axes = plt.subplots(n, 1, figsize=(20, max(6, 1.2 * n + 2)), sharex=True, squeeze=False)
    for idx, ax in enumerate(axes[:, 0]):
        i, j = keys[idx]
        ax.plot(t, deltaf[idx], color=colors[idx % len(colors)], linewidth=1.5)
        ax.set_ylabel(f'{i}-{j}', fontsize=7, rotation=0, labelpad=35)
        ax.tick_params(labelbottom=(idx == n - 1))
    axes[-1, 0].set_xlabel('time (s)')
    fig.suptitle(f'{number}: dF/F per layer mask')
    fig.savefig(os.path.join(output_dir, f'{number}_deltaf.svg'))
    plt.close(fig)


def activity_main(number, data_dir=DATA_DIR, video_dir=VIDEO_DIR, output_dir='outputs3'):
    masks, keys, _ = load_masks(number, data_dir)

    video_path = find_video(number, video_dir)
    data = tifffile.imread(video_path)
    if data.ndim != 3:
        raise ValueError(f'Expected a (T, H, W) video, got shape {data.shape}')
    if data.shape[1:] != masks.shape[1:]:
        raise ValueError(f'Video frames are {data.shape[1:]} but masks are {masks.shape[1:]}; '
                         'masks must be made from the same field of view')

    deltaf = compute_deltaf(data, masks)
    plot_deltaf(number, deltaf, keys, output_dir)

    t = np.arange(deltaf.shape[1]) / SAMPLE_RATE_HZ
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(os.path.join(data_dir, f'{number}_deltaf.npz'), deltaf=deltaf, t=t, keys=keys)
    print(f'dF/F for {len(deltaf)} masks -> {output_dir} (plot), {data_dir} (data)')
    return deltaf


if __name__ == "__main__":
    activity_main('659')
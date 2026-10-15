"""
sam2_plots.py

Grid plot of SAM2 mask results overlaid on the image.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt


def plot_grid_dynamic(number, image_rgb, results, output_dir='outputs3', max_per_figure=100):
    """
    Plot all results in a roughly-square grid sized to fit however many
    combos there are. If there are more than `max_per_figure` results,
    splits across multiple figures so matplotlib doesn't choke on one grid
    (extra figures get a _1, _2, ... suffix instead of overwriting each other).
    """
    os.makedirs(output_dir, exist_ok=True)
    items = list(results.items())
    n_figures = math.ceil(len(items) / max_per_figure)

    for fig_idx in range(n_figures):
        chunk = items[fig_idx * max_per_figure: (fig_idx + 1) * max_per_figure]
        n = len(chunk)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
        axes = np.array(axes).reshape(-1)

        for k, ((i, j), (prior_mask, pred_mask, score)) in enumerate(chunk):
            ax = axes[k]
            ax.imshow(image_rgb, cmap='gray')
            if pred_mask is not None:
                overlay = np.ma.masked_where(pred_mask == 0, pred_mask)
                ax.imshow(overlay, cmap='autumn', alpha=0.5)
                coverage_pct = 100 * np.count_nonzero(pred_mask) / pred_mask.size
                ax.set_title(f"{i}-{j} ({coverage_pct:.0f}%, s={score:.2f})", fontsize=6)
            else:
                ax.set_title(f"{i}-{j}", fontsize=6)
            ax.axis('off')

        for k in range(len(chunk), len(axes)):
            axes[k].axis('off')

        plt.tight_layout()
        suffix = '' if n_figures == 1 else f'_{fig_idx}'
        fig.savefig(os.path.join(output_dir, f'{number}_results{suffix}.svg'))
        plt.close(fig)  # avoid piling up open figures when looping over many files

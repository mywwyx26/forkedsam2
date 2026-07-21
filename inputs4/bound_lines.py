# convert bound lines tif to npy and smooth them out, view over clahe image
# napari -> add shapes layer -> draw lines (add path) -> convert to labels layer -> save layer as tif
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import splprep, splev

def bound_lines(number, data):
    labels_present = np.unique(data)
    labels_present = labels_present[labels_present != 0]  # drop background

    def smooth_line(mask, label_value, smoothing=1000, thickness=1):
        ys, xs = np.where(mask == label_value)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        # order points along the line's principal axis (PCA), since raw pixel coords aren't in path order
        mean = pts.mean(axis=0)
        centered = pts - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        proj = centered @ direction
        order = np.argsort(proj)
        ordered_pts = pts[order]
        canvas = np.zeros(mask.shape, dtype=np.uint8)

        # dedupe consecutive duplicate points, splprep needs strictly increasing param
        _, unique_idx = np.unique(ordered_pts, axis=0, return_index=True)
        clean_pts = ordered_pts[np.sort(unique_idx)]
        tck, _ = splprep([clean_pts[:, 0], clean_pts[:, 1]], s=smoothing)
        u_fine = np.linspace(0, 1, 500)
        x_fine, y_fine = splev(u_fine, tck)
        curve_pts = np.stack([x_fine, y_fine], axis=1).astype(np.int32)
        cv2.polylines(canvas, [curve_pts], isClosed=False, color=1, thickness=thickness)

        return canvas * label_value

    # rebuild the full label image from smoothed individual lines
    smoothed_data = np.zeros_like(data)
    for lbl in labels_present:
        smoothed_data += smooth_line(data, lbl)
    np.save(f'bounds\\{number}.npy', smoothed_data)
    return data, smoothed_data, labels_present

if __name__ == "__main__":
    number = '659' # change number for each file
    clahe_image = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\clahe\\{number}_clahe.npy')
    data = cv2.imread(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\bounds\\{number}.tif', cv2.IMREAD_UNCHANGED)
    data, smoothed_data, labels_present = bound_lines(number, data) # does np.save
    
    # plotting
    bright_colors = ['#00FFFF', '#FFFF00']  # cyan, magenta -- pick any high-contrast set
    cmap = ListedColormap(bright_colors[:len(labels_present)])
    bounds = np.concatenate([labels_present - 0.5, [labels_present[-1] + 0.5]])
    norm = BoundaryNorm(bounds, cmap.N)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(clahe_image, cmap='gray')
    ax1.imshow(np.ma.masked_where(data == 0, data), cmap=cmap, norm=norm)
    ax1.axis('off')
    ax2.imshow(clahe_image, cmap='gray')
    ax2.imshow(np.ma.masked_where(smoothed_data == 0, smoothed_data), cmap=cmap, norm=norm)
    ax2.axis('off')
    plt.show()


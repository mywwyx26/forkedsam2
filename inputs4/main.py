# pipeline: clahe -> napari -> bound_lines -> interpolate_lines -> run_sam2_lines
# there's nothing i can do about the napari step so we start from bound_lines

import cv2
import numpy as np
from pathlib import Path

from bound_lines import bound_lines
from interpolate_lines import order_points_by_path, generate_intermediate_lines, rasterize_lines
from run_sam2_lines import run_sam2_on_lines, plot_grid

numbers = ['659', '661', '663', '665', '668', '670', '746']
for number in numbers:
    clahe_image = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\clahe\\{number}_clahe.npy')

    # bound lines
    data = cv2.imread(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\bounds\\{number}.tif', cv2.IMREAD_UNCHANGED)
    bound_lines(number, data)
    print(f'bound_lines done for {number}')

    # interpolated lines
    smoothed_data = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\bounds\\{number}.npy')
    line1_pts = order_points_by_path(smoothed_data, 1)
    line2_pts = order_points_by_path(smoothed_data, 2)
    intermediate_lines = generate_intermediate_lines(line1_pts, line2_pts, 30)
    rasterize_lines(number, intermediate_lines, smoothed_data.shape)
    print(f'interpolate_lines done for {number}')

    # run sam2
    lines_label_img = np.load(f'lines\\{number}.npy')
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
    print(f'sam2 done for {number}')

    plot_grid(image_rgb, results, rows=5, cols=6)
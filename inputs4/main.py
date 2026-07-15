# pipeline: clahe -> napari -> bound_lines -> interpolate_lines -> run_sam2_lines
# there's nothing i can do about the napari step so we start from bound_lines

import cv2
import numpy as np
from pathlib import Path

from bound_lines import bound_lines
from interpolate_lines import order_points_by_path, generate_intermediate_lines, rasterize_lines
from run_sam2_lines_again import sam2_main

numbers = ['659', '661', '663', '665', '668', '670', '746']
for number in numbers:
    clahe_image = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\clahe\\{number}_clahe.npy')
    print(f'starting {number}')

    # bound lines
    data = cv2.imread(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\bounds\\{number}.tif', cv2.IMREAD_UNCHANGED)
    bound_lines(number, data)

    # interpolated lines
    smoothed_data = np.load(f'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\bounds\\{number}.npy')
    line1_pts = order_points_by_path(smoothed_data, 1)
    line2_pts = order_points_by_path(smoothed_data, 2)
    intermediate_lines = generate_intermediate_lines(line1_pts, line2_pts, 30)
    rasterize_lines(number, intermediate_lines, smoothed_data.shape)

    # run sam2 with mask prompts
    sam2_main(number=number, clahe_image=clahe_image, smoothed_data=smoothed_data)
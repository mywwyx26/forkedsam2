# pipeline: clahe -> napari -> bound_lines -> interpolate_lines -> run_sam2_lines -> activity
# there's nothing i can do about the napari step so we start from bound_lines

import os
from pathlib import Path

import cv2
import numpy as np

# everything (clahe/, bounds/, lines/, outputs3/, ...) lives next to this file.
# chdir so the relative paths inside the other scripts ('lines\\', 'bounds\\', 'outputs3') resolve
# here no matter which folder the script is launched from.
BASE = Path(__file__).resolve().parent
os.chdir(BASE)

from bound_lines import bound_lines
from interpolate_lines import order_points_by_path, generate_intermediate_lines, rasterize_lines
from run_sam2_lines import sam2_main
from activity import activity_main

numbers = ['659', '661', '663', '665', '668', '670', '746']
for number in numbers:
    clahe_image = np.load(BASE / 'clahe' / f'{number}_clahe.npy')
    print(f'\nstarting {number}')

    # bound lines
    data = cv2.imread(str(BASE / 'bounds' / f'{number}.tif'), cv2.IMREAD_UNCHANGED)
    bound_lines(number, data)

    # interpolated lines
    smoothed_data = np.load(BASE / 'bounds' / f'{number}.npy')
    line1_pts = order_points_by_path(smoothed_data, 1)
    line2_pts = order_points_by_path(smoothed_data, 2)
    intermediate_lines = generate_intermediate_lines(line1_pts, line2_pts, 30)
    rasterize_lines(number, intermediate_lines, smoothed_data.shape)

    # run sam2 with mask prompts (also saves outputs3/data/{number}_masks.npz)
    sam2_main(number=number, clahe_image=clahe_image, smoothed_data=smoothed_data)

    # dF/F per selected mask, from the raw video
    activity_main(number)
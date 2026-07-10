import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

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

'''
part 1: obtain outside lines
 - so first, consider that i don't actually know how to GET the lines in the first place
 - i could draw them and export as png? like i do with medibang???
 - however this should also not be my concern bc then there's the whole coding part
 - and there's no way it's actually hard to get automatically

part 2: get inside lines, the layers and columns
 - basically just need to get one of them first, then the other can be done by perpendicular
 - i'm sure there's a code to actually get the perpendicular one at every point...
 - problem is how to get the first parallel line
 - i assume it would be like, at x percent of the outside line for both lines, this point on the inside line
   makes it y percent of the way from the first line to the second line, move along x while keeping y the same
 - also this line will come to an abrupt stop at the end lines (the perpendicular ones)
 - an issue is that sometimes the recording cuts off the ends weirdly, so it's hard to draw end lines for columns
 - this may be solved by instead making the layer lines longer, and just drawing column line in the middle
 - there will be extra bits on the ends but that should be fine, and may be hard to continue layer blindly
 
part 3: convert to a form sam2 can read
 - the lines can be either masks or multiple points, worth trying both

part 4: filter out the correct masks
 - am stumped on this one.
 - i could say get rid of the ones that are 95%+ the same?
 - but that doesn't guarantee the remaining ones are right

'''

# main loop
input_folder = 'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\original'
inputs = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) if f.endswith(".tif")]
output = 'layer_masks'
os.makedirs(output, exist_ok=True)

for i in range(len(inputs)):
    filename = os.path.splitext(os.path.splitext(os.path.basename(inputs[i]))[0])[0]
    clahe_image = np.load(os.path.join('clahe',f'{filename}_clahe.npy'))
    print(f"\nProcessing {filename}...")

    clahe_image_path = os.path.join('clahe',f'{filename}_clahe.npy')
    if not os.path.exists(clahe_image_path):
        print(f"  skipping -- could not find source image at {clahe_image_path}")
        continue
    clahe_image = np.load(clahe_image_path)
    predictor.set_image(cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB))

    layer_mask_path = os.path.join('layer_masks',f'{filename}_layer_mask.npy')
    if not os.path.exists(layer_mask_path):
        print(f"  skipping -- no layer mask found at {layer_mask_path}")
        continue
    layer_mask = np.load(layer_mask_path)

    column_mask_path = os.path.join('column_masks',f'{filename}_column_mask.npy')
    if not os.path.exists(column_mask_path):
        print(f"  skipping -- no column mask found at {column_mask_path}")
        continue
    column_mask = np.load(column_mask_path)
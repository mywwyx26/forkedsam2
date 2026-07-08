import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_folder = 'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs2\\original'
inputs = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) if f.endswith(".tif")]

fig, axes = plt.subplots(2, len(inputs))

for i in range(len(inputs)):
    image = cv2.imread(inputs[i], cv2.IMREAD_UNCHANGED)
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[0,i].imshow(image, cmap='gray')
    axes[0,i].axis('off')

    # tried a few numbers and have decided that these are best
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(16,16))
    clahe_img = clahe.apply(image_8bit)
    axes[1,i].imshow(clahe_img, cmap='gray')
    axes[1,i].axis('off')

plt.show()
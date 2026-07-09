import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import label, regionprops

input_folder = 'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs3\\original'
inputs = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) if f.endswith(".tif")]
fig, axes = plt.subplots(6,len(inputs))

for i in range(len(inputs)):
    image = cv2.imread(inputs[i], cv2.IMREAD_UNCHANGED)
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[0,i].imshow(image, cmap='gray')
    axes[0,i].axis('off')

    blurred = ndimage.gaussian_filter(image, sigma=5)
    axes[1,i].imshow(blurred, cmap='gray')
    axes[1,i].axis('off')

    binarized = np.where(image > np.mean(image), 1, 0).astype(np.uint8)
    axes[2,i].imshow(binarized, cmap='gray')
    axes[2,i].axis('off')

    # remove small specks and fill holes (50px)
    lbl_speckles = label(binarized)
    sizes = ndimage.sum(binarized, lbl_speckles, range(1, lbl_speckles.max() + 1))
    binarized = np.isin(lbl_speckles, np.where(sizes >= 50)[0] + 1).astype(np.uint8)
    binarized = ndimage.binary_fill_holes(binarized).astype(np.uint8)
    axes[3,i].imshow(binarized, cmap='gray')
    axes[3,i].axis('off')

    # use erosion to break off extra parts and isolate medulla
    # disclaimer: this was only a problem for one of them, 32 may not always work
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32,32))
    opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
    axes[4,i].imshow(opened, cmap='gray')
    axes[4,i].axis('off')

    lbl = label(opened)
    props = regionprops(lbl)
    largest = max(props, key=lambda p: p.area)
    final = (lbl == largest.label).astype(np.uint8)
    axes[5,i].imshow(final, cmap='gray')
    axes[5,i].axis('off')

plt.show()
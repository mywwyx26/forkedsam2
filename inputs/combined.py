import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# binarize the combined registered image
combined = cv2.imread('AVG_inputs_registered.tif', cv2.IMREAD_UNCHANGED)
binarized = np.where(combined > np.mean(combined)*0.8, 1, 0)

# make the binarized neuropils rounder and remove small dots
kernel = [[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]]
#kernel = [[0,0,1,1,1,0,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0]]
#binarized_opening = ndimage.binary_opening(binarized, structure=kernel2)
#binarized_closing = ndimage.binary_closing(binarized_opening, structure=kernel1)

#kernel = np.ones((3,3), dtype=bool)
binarized_opening = ndimage.binary_opening(binarized, structure=kernel)
binarized_closing = ndimage.binary_closing(binarized_opening, structure=kernel)
distance = ndimage.distance_transform_edt(binarized_closing)
coordinates = peak_local_max(distance, min_distance=int(0.1*np.mean(np.shape(combined))), labels=binarized_closing)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coordinates.T)] = True
markers, _ = ndimage.label(mask)

# Labels will contain 0 for background, 1 for Blob A, 2 for Blob B, etc.
labels = watershed(-distance, markers, mask=binarized_closing)
final_binary_separated = labels > 0

# plot the binarized versions to see which strategy is best
fig, axes = plt.subplots(2,2)
axes[0,0].imshow(binarized, cmap='gray')
axes[0,0].axis('off')
axes[0,1].imshow(binarized_opening, cmap='gray')
axes[0,1].axis('off')
axes[1,0].imshow(binarized_closing, cmap='gray')
axes[1,0].axis('off')
axes[1,1].imshow(labels, cmap='jet')
axes[1,1].axis('off')
plt.tight_layout()
plt.show()

# plot with overlay on everyone
files = ['inputs_registered00000000.tif', 'inputs_registered00000001.tif',
         'inputs_registered00000002.tif', 'inputs_registered00000003.tif',
         'inputs_registered00000004.tif', 'inputs_registered00000005.tif',
         'inputs_registered00000006.tif']

fig, axes = plt.subplots(3,3)
axes[0,0].imshow(combined, cmap='gray')
axes[0,0].axis('off')
axes[0,1].imshow(binarized, cmap='gray')
axes[0,1].axis('off')

count = 2
cmap_green = LinearSegmentedColormap.from_list('cg', ['#000000', '#00FF00'])
for file in files:
    data = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    axes[count // 3, count % 3].imshow(data, cmap='gray')
    axes[count // 3, count % 3].imshow(binarized, cmap=cmap_green, alpha = 0.1)
    axes[count // 3, count % 3].axis('off')
    count = count + 1

plt.tight_layout
#plt.show()
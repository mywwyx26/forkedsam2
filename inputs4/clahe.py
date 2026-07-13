# apply clahe to all original images (only needs to be run once)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_folder = 'C:\\Users\\megan\\flies\\sams\\forkedsam2\\inputs4\\original'
inputs = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) if f.endswith(".tif")]
output = 'clahe'
os.makedirs(output, exist_ok=True)

fig, axes = plt.subplots(2,len(inputs))
for i in range(len(inputs)):
    image = cv2.imread(inputs[i], cv2.IMREAD_UNCHANGED)
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[0,i].imshow(image, cmap='gray')
    axes[0,i].axis('off')

    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(16,16))
    clahe_image = clahe.apply(image_8bit)
    axes[1,i].imshow(clahe_image, cmap='gray')
    axes[1,i].axis('off')

    filename = os.path.splitext(os.path.splitext(os.path.basename((inputs[i].split('-'))[-1]))[0])[0]
    np.save(os.path.join(output, f'{filename}_clahe.npy'), clahe_image)

plt.show()
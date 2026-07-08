import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops

# binarize the combined registered image
combined = cv2.imread('AVG_inputs_registered.tif', cv2.IMREAD_UNCHANGED)
binarized = np.where(combined > np.mean(combined) * 0.8, 1, 0).astype(np.uint8)


def keep_n_largest_labels(labels, n):
    """Keep only the n largest labels in an integer label image, relabeled
    1..n by descending area. Returns the cleaned label image."""
    if labels.max() <= n:
        return labels
    props = regionprops(labels)
    props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
    keep = {p.label for p in props_sorted[:n]}
    relabel_map = {old: new for new, old in enumerate(sorted(keep), start=1)}
    out = np.where(np.isin(labels, list(keep)), labels, 0)
    return np.vectorize(lambda v: relabel_map.get(v, 0))(out)


# --- 1. Remove small dots (specks), BEFORE splitting ---
# Dots still need to go before the split -- they can otherwise pollute the
# distance transform and generate a spurious extra peak_local_max seed.
min_dot_area = 60
lbl_speckles = label(binarized)
sizes = ndimage.sum(binarized, lbl_speckles, range(1, lbl_speckles.max() + 1))
dot_mask = np.isin(lbl_speckles, np.where(sizes >= min_dot_area)[0] + 1)
binarized_clean = dot_mask.astype(np.uint8)

# --- 2. Fill small holes, BEFORE splitting ---
# Holes also need to go before the split -- an internal gap distorts the
# distance transform locally and can shift or split a peak.
binarized_filled = ndimage.binary_fill_holes(binarized_clean).astype(np.uint8)

# --- 3. Split into (up to) 3 neuropils FIRST, on the un-rounded mask ---
# Rounding (opening/closing with a big kernel) BEFORE this step is what
# risks losing a blob entirely: a kernel bigger than a thin/small neuropil
# can erode it away completely, or shift/merge its distance-transform peak
# with a neighbor's, so peak_local_max never finds it. Splitting on the
# cleaned-but-unrounded mask guarantees the split reflects what's actually
# there.
distance = ndimage.distance_transform_edt(binarized_filled)
coordinates = peak_local_max(
    distance, min_distance=int(0.1 * np.mean(np.shape(combined))), labels=binarized_filled
)
seed_mask = np.zeros(distance.shape, dtype=bool)
seed_mask[tuple(coordinates.T)] = True
markers, _ = ndimage.label(seed_mask)

labels = watershed(-distance, markers, mask=binarized_filled)
labels = keep_n_largest_labels(labels, n=3)

n_found = labels.max()
if n_found < 3:
    print(f"WARNING: only found {n_found} region(s) before rounding -- "
          f"check min_dot_area / peak_local_max min_distance, rounding can't fix a missed split.")

# --- 4. NOW round each of the 3 labels INDEPENDENTLY ---
round_kernel_size = 21  # bump up for rounder, down to keep more real shape detail
smooth_sigma = 4.0      # optional extra rounding pass; set to 0 to skip
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round_kernel_size, round_kernel_size))

rounded_masks = {}
for lbl_id in range(1, n_found + 1):
    m = (labels == lbl_id).astype(np.uint8)
    m = ndimage.binary_opening(m, structure=kernel).astype(np.uint8)
    m = ndimage.binary_closing(m, structure=kernel).astype(np.uint8)
    if smooth_sigma > 0:
        blurred = ndimage.gaussian_filter(m.astype(np.float32), sigma=smooth_sigma)
        m = (blurred > 0.5).astype(np.uint8)
    rounded_masks[lbl_id] = m

# --- 5. Reconcile: independently-rounded blobs can now overlap at their
# shared boundary (closing can dilate a blob outward into space that used
# to belong to its neighbor). Re-run watershed on the UNION of the rounded
# masks, using the ORIGINAL seed markers -- this guarantees exactly the
# same 3 blobs (can't lose one, since the markers force them to exist) and
# cleanly resolves any boundary overlap based on rounded distance, giving
# an overall rounder final shape.
union_rounded = np.zeros_like(binarized_filled)
for m in rounded_masks.values():
    union_rounded = np.maximum(union_rounded, m)

distance_rounded = ndimage.distance_transform_edt(union_rounded)
# restrict markers to only the surviving n_found seeds, matching original label ids
markers_final = np.where(np.isin(markers, list(range(1, n_found + 1))), markers, 0)
labels_rounded = watershed(-distance_rounded, markers_final, mask=union_rounded)
labels_rounded = keep_n_largest_labels(labels_rounded, n=3)

final_binary_separated = labels_rounded > 0

# plot the binarized versions to see which strategy is best
fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(binarized, cmap='gray')
axes[0, 0].set_title('raw binarized')
axes[0, 0].axis('off')
axes[0, 1].imshow(labels, cmap='jet')
axes[0, 1].set_title('split BEFORE rounding')
axes[0, 1].axis('off')
axes[1, 0].imshow(union_rounded, cmap='gray')
axes[1, 0].set_title('rounded (union)')
axes[1, 0].axis('off')
axes[1, 1].imshow(labels_rounded, cmap='jet')
axes[1, 1].set_title('final 3 labels, rounded')
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()

# plot with overlay on everyone
files = ['inputs_registered00000000.tif', 'inputs_registered00000001.tif',
         'inputs_registered00000002.tif', 'inputs_registered00000003.tif',
         'inputs_registered00000004.tif', 'inputs_registered00000005.tif',
         'inputs_registered00000006.tif']

fig, axes = plt.subplots(3, 3)
axes[0, 0].imshow(combined, cmap='gray')
axes[0, 0].axis('off')
axes[0, 1].imshow(final_binary_separated, cmap='gray')
axes[0, 1].axis('off')

count = 2
cmap_green = LinearSegmentedColormap.from_list('cg', ['#000000', '#00FF00'])
for file in files:
    data = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    axes[count // 3, count % 3].imshow(data, cmap='gray')
    axes[count // 3, count % 3].imshow(final_binary_separated, cmap=cmap_green, alpha=0.1)
    axes[count // 3, count % 3].axis('off')
    count = count + 1

plt.tight_layout()
plt.show()
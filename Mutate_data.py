# # Adapted from Team Challenge Team 4 code

# import glob
# import numpy as np
# import os
# import random
# import scipy
# import SimpleITK as sitk

# from os import listdir
# from os.path import isfile, join
# from scipy import ndimage

# def mutate_data(images, masks, segmentations, noise_strength=32, new_records=80, seed=314159):
#     count = 0
#     random.seed(seed)

#     # Get how many original images we have
#     original_amount = images.shape[0]

#     # Preallocate array space for new images
#     new_shape = images.shape
#     new_shape = list(new_shape)
#     new_shape[0] += new_records
#     new_shape = tuple(new_shape)

#     res_im = np.zeros(new_shape)
#     res_ma = np.zeros(new_shape)
#     res_se = np.zeros(new_shape)

#     # Place original images into result array
#     res_im[0:images.shape[0], 0:images.shape[1], 0:images.shape[2]] = images
#     res_ma[0:masks.shape[0], 0:masks.shape[1], 0:masks.shape[2]] = masks
#     res_se[0:segmentations.shape[0], 0:segmentations.shape[1], 0:segmentations.shape[2]] = segmentations

#     print("Mutating Data...")
#     print("SETTINGS")
#     print("\tNoise Strength = {}".format(noise_strength))
#     print("\tNew Records = {}".format(new_records))
#     print("\tSeed = {}".format(seed))
#     while count < new_records:
#         # Fetch random patient images
#         print("{}/{}".format(count, new_records), end='\r')

#         idx = random.randint(0, images.shape[0] - 1)

#         image = images[idx]
#         mask = masks[idx]
#         segmentation = segmentations[idx]

#         # Mutate images
#         # Determine which operation to perform
#         operation = random.randint(0, 3)

#         # Random Rotation
#         if operation == 0:
#             # Determine new orientation
#             rot = random.randint(0, 2)

#             if rot == 0:
#                 # Flip over one axis
#                 image = np.flipud(image)
#                 mask = np.flipud(mask)
#                 segmentation = np.flipud(segmentation)
#             elif rot ==1:
#                 # Flip over one axis
#                 image = np.fliplr(image)
#                 mask = np.fliplr(mask)
#                 segmentation = np.fliplr(segmentation)
#             else:
#                 # Flip over both axes
#                 image = np.flipud(image)
#                 mask = np.flipud(mask)
#                 segmentation = np.flipud(segmentation)

#                 image = np.fliplr(image)
#                 mask = np.fliplr(mask)
#                 segmentation = np.fliplr(segmentation)

#         # Random noise (normal distribution)
#         elif operation == 1:
#             # Noise is only applied to the actual image
#             image = np.random.normal(image, noise_strength)
#         # Contrast change
#         elif operation == 2:
#             image = (float(random.randint(50, 150)) / 100.0) * image
#         # Blur Kernel
#         elif operation == 3:
#             image = ndimage.gaussian_filter(image, 2)

#         # Save
#         res_im[original_amount + count] = image
#         res_ma[original_amount + count] = mask
#         res_se[original_amount + count] = segmentation

#         count += 1

#     print("Mutated images")
#     return res_im, res_ma, res_se




import os
import numpy as np
import cv2
from tqdm import tqdm

# Define paths for HRF dataset
TRAIN_IMAGES_PATH = "HRF/train/enhanced"
TRAIN_MASKS_PATH = "HRF/train/av3"
TEST_IMAGES_PATH = "HRF/test/enhanced"
TEST_MASKS_PATH = "HRF/test/av3"

PATCHES_OUTPUT_PATH = "HRF/patches"
PATCH_SIZE = 128  # Size of patches
STRIDE = 64       # Stride for sliding window

# Create output directories
os.makedirs(PATCHES_OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(PATCHES_OUTPUT_PATH, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(PATCHES_OUTPUT_PATH, "train", "masks"), exist_ok=True)
os.makedirs(os.path.join(PATCHES_OUTPUT_PATH, "test", "images"), exist_ok=True)
os.makedirs(os.path.join(PATCHES_OUTPUT_PATH, "test", "masks"), exist_ok=True)

# Map colors to class IDs
COLOR_TO_CLASS = {
    (255, 0, 255): 1,  # Magenta -> Arteries
    (0, 255, 255): 2,  # Cyan -> Veins
    (0, 0, 128): 3,    # Dark Blue -> Unknown
    (0, 0, 0): 0       # Black -> Background
}

def convert_color_to_class(mask):
    """
    Convert color-coded mask to class ID mask.
    """
    h, w, _ = mask.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for color, class_id in COLOR_TO_CLASS.items():
        class_mask[(mask == color).all(axis=-1)] = class_id
    return class_mask

def create_patches(image, mask, output_path, prefix):
    """
    Create patches from images and masks and save them.
    """
    h, w, _ = image.shape
    patch_idx = 0
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            img_patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            
            # Save patches
            img_filename = os.path.join(output_path, "images", f"{prefix}_{patch_idx}.png")
            mask_filename = os.path.join(output_path, "masks", f"{prefix}_{patch_idx}.png")
            cv2.imwrite(img_filename, img_patch)
            cv2.imwrite(mask_filename, mask_patch)
            patch_idx += 1

def process_dataset(images_path, masks_path, output_path, prefix):
    """
    Process images and masks to create patches.
    """
    images = sorted(os.listdir(images_path))
    masks = sorted(os.listdir(masks_path))
    
    for img_file, mask_file in tqdm(zip(images, masks), total=len(images), desc=f"Processing {prefix}"):
        img_path = os.path.join(images_path, img_file)
        mask_path = os.path.join(masks_path, mask_file)
        
        # Read image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        
        # Convert mask to class IDs
        class_mask = convert_color_to_class(mask)
        
        # Create and save patches
        create_patches(image, class_mask, output_path, prefix)

# Process training and test datasets
process_dataset(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, os.path.join(PATCHES_OUTPUT_PATH, "train"), "train")
process_dataset(TEST_IMAGES_PATH, TEST_MASKS_PATH, os.path.join(PATCHES_OUTPUT_PATH, "test"), "test")

print("Patches created and saved successfully.")

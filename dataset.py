import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class RetinalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Image and mask directories
        self.image_dir = os.path.join(data_dir, 'enhanced')
        self.mask_dir = os.path.join(data_dir, 'av3')
        
        # List all image files
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Open the image and mask files
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        mask = Image.open(mask_path)  # Assuming the mask is in grayscale
        
        # Convert mask colors to class labels (if applicable)
        # For this case, magenta for arteries, cyan for veins, etc.
        mask = np.array(mask)
        
        # Mapping color values to label values (example)
        # Adjust depending on your mask color coding
        mask[mask == 255] = 0  # Background (if mask is in binary form)
        mask[mask == 128] = 1  # Arteries
        mask[mask == 64] = 2  # Veins
        
        mask = Image.fromarray(mask.astype(np.uint8))  # Convert to image format for consistency
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

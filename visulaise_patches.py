import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def visualize_patches(patches, labels=None, title="Patches Visualization", save_path=None):
    """
    Visualize patches in a grid format.

    Parameters:
    - patches: Numpy array of shape (N, patch_height, patch_width, channels).
    - labels: Optional. List of labels corresponding to patches.
    - title: Title of the visualization.
    - save_path: Path to save the visualization image. If None, it just displays.
    """
    num_patches = len(patches)
    grid_size = int(np.ceil(np.sqrt(num_patches)))  # Square grid
    
    plt.figure(figsize=(12, 12))
    gs = GridSpec(grid_size, grid_size)
    plt.suptitle(title, fontsize=16)
    
    for i in range(num_patches):
        ax = plt.subplot(gs[i])
        ax.axis('off')  # Remove axes
        
        patch = patches[i]
        if patch.ndim == 2:  # Grayscale
            ax.imshow(patch, cmap='gray')
        else:  # RGB
            ax.imshow(patch)
        
        if labels is not None:
            ax.set_title(f"Label: {labels[i]}", fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust to fit title
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved at: {save_path}")
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Path to the patches folder or dataset
    patches_folder = "./patches/"
    patches = []
    labels = []  # Optional: Add labels if available
    
    # Load example patches
    for patch_file in os.listdir(patches_folder):
        if patch_file.endswith(".npy"):  # Assuming patches are saved as .npy
            patch = np.load(os.path.join(patches_folder, patch_file))
            patches.append(patch)
            # Optionally, add label parsing here if filenames contain labels
    
    # Convert to numpy array
    patches = np.array(patches)
    
    # Visualize
    visualize_patches(patches, labels=None, title="Sample HRF Patches", save_path="./patch_visualization.png")

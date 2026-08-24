"""Script to load patches and plot them sorted by deforestation area."""

import sys
import os
import yaml
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_config(config_path: str = "../config.yaml") -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the config YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_patches_from_h5(h5_file_path: str):
    """Load patches from HDF5 file.
    
    Args:
        h5_file_path: Path to the input patches HDF5 file
        
    Returns:
        tuple: (patches, time_indices, num_patches_per_time)
    """
    if not Path(h5_file_path).exists():
        raise FileNotFoundError(
            f"Patches file not found at {h5_file_path}"
        )
    
    with h5py.File(h5_file_path, 'r') as f:
        patches = f['patches'][:]
        time_indices = f['time_indices'][:]
        
        print(f"Loaded patches from {h5_file_path}")
        print(f"Patches shape: {patches.shape}")
        print(f"Time indices shape: {time_indices.shape}")
        
        # Calculate number of patches per time
        num_patches_per_time = np.sum(time_indices == 0)
        print(f"Number of patches per time: {num_patches_per_time}")
        
        return patches, time_indices, num_patches_per_time


def get_deforestation_masks(patches, time_indices, num_patches_per_time):
    """Extract deforestation masks from patches.
    
    Args:
        patches: Array of patches with shape (N, C, H, W)
        time_indices: Array of time indices
        num_patches_per_time: Number of patches per time slice
        
    Returns:
        tuple: (deforestation_masks, valid_indices)
    """
    # Get the latest time index (most recent data)
    max_time_index = time_indices.max()
    
    # Get patches from the latest time
    latest_time_mask = time_indices == max_time_index
    latest_patches = patches[latest_time_mask]
    
    # Channel 1 is the target mask (deforestation mask)
    deforestation_masks = latest_patches[:, 1, :, :]
    
    # Find patches with some deforestation (non-zero pixels)
    valid_indices = []
    for i, mask in enumerate(deforestation_masks):
        if np.any(mask > 0):
            valid_indices.append(i)
    
    print(f"Found {len(valid_indices)} patches with deforestation out of "
          f"{len(deforestation_masks)} total")
    
    return deforestation_masks, valid_indices


def calculate_deforestation_area(masks):
    """Calculate deforestation area for each mask.
    
    Args:
        masks: Array of deforestation masks with shape (N, H, W)
        
    Returns:
        np.ndarray: Array of deforestation areas for each mask
    """
    deforestation_areas = masks.sum(axis=(1, 2))
    return deforestation_areas


def sort_patches_by_deforestation_area(masks, valid_indices):
    """Sort patches by deforestation area in descending order.
    
    Args:
        masks: Array of deforestation masks with shape (N, H, W)
        valid_indices: List of indices with deforestation
        
    Returns:
        tuple: (sorted_masks, sorted_indices, sorted_areas)
    """
    # Only consider patches with deforestation
    valid_masks = masks[valid_indices]
    
    # Calculate deforestation areas
    deforestation_areas = calculate_deforestation_area(valid_masks)
    
    # Get indices that would sort by deforestation area in descending order
    sort_indices = np.argsort(deforestation_areas)[::-1]
    
    # Sort all arrays using the indices
    sorted_masks = valid_masks[sort_indices]
    sorted_areas = deforestation_areas[sort_indices]
    
    # Map back to original indices
    sorted_original_indices = [valid_indices[i] for i in sort_indices]
    
    print("Sorted patches by deforestation area (descending):")
    print(f"Top 10 deforestation areas: {sorted_areas[:10]}")
    
    return sorted_masks, sorted_original_indices, sorted_areas


def plot_top_patches(masks, original_indices, areas, n_patches=4):
    """Plot the top n patches with highest deforestation area.
    
    Args:
        masks: Sorted deforestation masks array with shape (N, H, W)
        original_indices: Original indices of sorted patches
        areas: Sorted deforestation areas
        n_patches: Number of patches to plot
    """
    # Set color scale maximum (same as plots.py)
    vmax = 1.0
    
    # Create a figure with subplots for each patch (single column)
    fig, axes = plt.subplots(n_patches, 1, figsize=(8, 5*n_patches))
    fig.suptitle('Top 4 Patches with Highest Deforestation Area',
                 fontsize=16, fontweight='bold')
    
    # Handle case where n_patches = 1
    if n_patches == 1:
        axes = [axes]
    
    for i in range(n_patches):
        patch_idx = original_indices[i]
        deforestation_area = areas[i]
        mask = masks[i]  # Shape: (H, W)
        
        # Calculate some statistics
        total_pixels = mask.size
        deforested_pixels = np.sum(mask > 0)
        deforestation_percentage = (deforested_pixels / total_pixels) * 100
        
        # Plot target (same style as plots.py)
        im = axes[i].imshow(mask, cmap='coolwarm', vmin=0, vmax=vmax)
        axes[i].set_title(f'Ground Truth\n(Target Area: {deforestation_area:.0f})')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Add patch info as text (same style as plots.py)
        patch_info = (f"Patch {patch_idx}: Deforestation Area = "
                      f"{deforestation_area:.0f}")
        fig.text(0.02, 0.95 - i*0.23, patch_info, fontsize=10,
                 fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def main():
    """Main function to load patches and plot sorted patches."""
    try:
        # Load configuration
        config = load_config()
        
        # Fix path resolution - add .. prefix when running from notebooks/
        patches_path = config["data"]["h5_paths"]["input"]
        if not patches_path.startswith('/'):  # If it's a relative path
            patches_path = os.path.join('..', patches_path)
        
        # Load patches
        patches, time_indices, num_patches_per_time = load_patches_from_h5(
            patches_path)
        
        # Get deforestation masks
        masks, valid_indices = get_deforestation_masks(
            patches, time_indices, num_patches_per_time)
        
        # Debug: Check data statistics
        print("\n=== DATA ANALYSIS ===")
        print(f"Total number of patches: {len(patches)}")
        print(f"Patches shape: {patches.shape}")
        print(f"Patches data type: {patches.dtype}")
        print(f"Patches min: {patches.min()}, max: {patches.max()}")
        
        if len(valid_indices) == 0:
            print("No patches with deforestation found!")
            return
        
        # Calculate deforestation areas for all valid patches
        valid_masks = masks[valid_indices]
        all_deforestation_areas = calculate_deforestation_area(valid_masks)
        
        print("\nDeforestation area statistics:")
        print(f"  Min area: {all_deforestation_areas.min()}")
        print(f"  Max area: {all_deforestation_areas.max()}")
        print(f"  Mean area: {all_deforestation_areas.mean():.2f}")
        print(f"  Median area: {np.median(all_deforestation_areas):.2f}")
        
        # Sort by deforestation area
        (sorted_masks, sorted_indices,
         sorted_areas) = sort_patches_by_deforestation_area(
            masks, valid_indices
        )
        
        # Debug: Check individual patch statistics
        print("\n=== TOP 4 PATCHES ANALYSIS ===")
        for i in range(min(4, len(sorted_masks))):
            patch_idx = sorted_indices[i]
            mask = sorted_masks[i]
            
            print(f"Patch {patch_idx}:")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask unique values: {np.unique(mask)}")
            print(f"  Mask mean: {mask.mean():.4f}")
            print(f"  Deforested pixels: {np.sum(mask > 0)}")
            print(f"  Total pixels: {mask.size}")
        
        # Plot top 4 patches
        plot_top_patches(
            sorted_masks, sorted_indices, sorted_areas, 
            n_patches=min(4, len(sorted_masks))
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the patches HDF5 file exists. "
              "You might need to process the data first.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

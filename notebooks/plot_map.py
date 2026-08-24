"""Script to reconstruct and plot the full map from patches."""

import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import math

# Add src to path for imports
try:
    # Try to get the current file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    src_dir = os.path.join(parent_dir, 'src')
    sys.path.append(src_dir)
except NameError:
    # __file__ not available (e.g., in notebook), use current working directory
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

try:
    from src.utils.utils import load_config
except ImportError:
    # Try alternative import path
    sys.path.append('..')
    from src.utils.utils import load_config


def load_predictions_from_h5(h5_file_path: str):
    """Load predictions from HDF5 file.
    
    Args:
        h5_file_path: Path to the predictions HDF5 file
        
    Returns:
        tuple: (predictions, binary_predictions, targets, metadata)
    """
    if not Path(h5_file_path).exists():
        raise FileNotFoundError(
            f"Predictions file not found at {h5_file_path}"
        )
    
    with h5py.File(h5_file_path, 'r') as f:
        predictions = f['predictions'][:]
        binary_predictions = f['binary_predictions'][:]
        targets = f['targets'][:]
        
        # Load metadata
        metadata = {}
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]
            
        print(f"Loaded predictions from {h5_file_path}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Binary predictions shape: {binary_predictions.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Metadata: {metadata}")
        
        return predictions, binary_predictions, targets, metadata


def calculate_grid_size(num_patches: int) -> Tuple[int, int]:
    """Calculate optimal grid size for arranging patches.
    
    Args:
        num_patches: Total number of patches
        
    Returns:
        tuple: (grid_width, grid_height) in patches
    """
    # Try to make a roughly square grid
    sqrt_patches = int(math.sqrt(num_patches))
    
    # Find factors close to square root
    best_ratio = float('inf')
    best_w, best_h = sqrt_patches, sqrt_patches
    
    for w in range(sqrt_patches - 50, sqrt_patches + 50):
        if w <= 0:
            continue
        h = math.ceil(num_patches / w)
        if w * h >= num_patches:
            ratio = max(w, h) / min(w, h)
            if ratio < best_ratio:
                best_ratio = ratio
                best_w, best_h = w, h
    
    print(f"Grid size: {best_w} x {best_h} patches (total capacity: {best_w * best_h})")
    return best_w, best_h


def reconstruct_simple_map(
    predictions: np.ndarray,
    patch_size: int = 32
) -> np.ndarray:
    """Reconstruct a simple map by arranging patches in a grid.
    
    Args:
        predictions: Array of predictions with shape (N, 1, 1, H, W)
        patch_size: Size of each patch
        
    Returns:
        np.ndarray: Reconstructed full map
    """
    num_patches = len(predictions)
    grid_w, grid_h = calculate_grid_size(num_patches)
    
    # Initialize the full map
    full_map = np.zeros((grid_h * patch_size, grid_w * patch_size))
    
    # Place patches in the grid
    for i, prediction in enumerate(predictions):
        # Calculate grid position
        row = i // grid_w
        col = i % grid_w
        
        # Skip if we exceed the grid
        if row >= grid_h:
            break
            
        # Extract and process the patch
        if prediction.ndim == 4:  # (1, 1, H, W)
            patch = prediction.squeeze()
        elif prediction.ndim == 3:  # (1, H, W)
            patch = prediction.squeeze(0)
        else:
            patch = prediction
            
        # Place patch in the full map
        start_row = row * patch_size
        end_row = start_row + patch_size
        start_col = col * patch_size
        end_col = start_col + patch_size
        
        full_map[start_row:end_row, start_col:end_col] = patch
    
    return full_map


def plot_full_maps_simple(
    pred_map: np.ndarray,
    binary_pred_map: np.ndarray,
    target_map: np.ndarray,
    figsize: Tuple[int, int] = (20, 8)
):
    """Plot the full reconstructed maps.
    
    Args:
        pred_map: Raw prediction map
        binary_pred_map: Binary prediction map
        target_map: Target map
        figsize: Figure size
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    fig.suptitle('Full Amazon Deforestation Map Reconstruction', 
                 fontsize=20, fontweight='bold')
    
    # Raw predictions
    im1 = axes[0].imshow(pred_map, cmap='Greys', vmin=0, vmax=1)
    axes[0].set_title('Raw Predictions', fontsize=16)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Binary predictions
    im2 = axes[1].imshow(binary_pred_map, cmap='Greys', vmin=0, vmax=1)
    axes[1].set_title('Binary Predictions', fontsize=16)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Targets
    im3 = axes[2].imshow(target_map, cmap='Greys', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth', fontsize=16)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add statistics
    pred_area = np.sum(binary_pred_map > 0)
    target_area = np.sum(target_map > 0)
    accuracy = np.mean(binary_pred_map == target_map)
    
    stats_text = (f"Predicted Deforestation: {pred_area:,.0f} pixels\n"
                  f"Actual Deforestation: {target_area:,.0f} pixels\n"
                  f"Pixel Accuracy: {accuracy:.3f}\n"
                  f"Map Size: {pred_map.shape[0]} x {pred_map.shape[1]} pixels")
    
    fig.text(0.02, 0.02, stats_text, fontsize=12, fontweight='bold',
             verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                                                  facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def analyze_patches(predictions: np.ndarray, binary_predictions: np.ndarray, 
                   targets: np.ndarray):
    """Analyze the patches to understand the data better.
    
    Args:
        predictions: Raw predictions
        binary_predictions: Binary predictions
        targets: Ground truth targets
    """
    print("\n=== PATCH ANALYSIS ===")
    print(f"Total patches: {len(predictions)}")
    print(f"Patch size: {predictions.shape[-2:]} pixels")
    
    # Calculate deforestation statistics
    target_deforestation = np.sum(targets > 0, axis=(-2, -1))
    pred_deforestation = np.sum(binary_predictions > 0, axis=(-2, -1))
    
    print(f"\nDeforestation statistics:")
    print(f"  Patches with actual deforestation: {np.sum(target_deforestation > 0)}")
    print(f"  Patches with predicted deforestation: {np.sum(pred_deforestation > 0)}")
    print(f"  Average deforestation per patch (actual): {np.mean(target_deforestation.flatten()):.2f} pixels")
    print(f"  Average deforestation per patch (predicted): {np.mean(pred_deforestation.flatten()):.2f} pixels")
    
    # Find top deforestation patches
    top_indices = np.argsort(target_deforestation.flatten())[-10:][::-1]
    print(f"\nTop 10 patches by deforestation area:")
    for i, idx in enumerate(top_indices):
        actual = target_deforestation.flatten()[idx]
        predicted = pred_deforestation.flatten()[idx]
        print(f"  {i+1}. Patch {idx}: Actual={actual:.0f}, Predicted={predicted:.0f} pixels")


def main():
    """Main function to reconstruct and plot the full map."""
    try:
        # Load configuration
        config = load_config("../config.yaml")
        
        # Paths
        predictions_path = "../results/predictions_baseline_w7.h5"
        
        # Configuration
        patch_size = 32  # From the data shape we can see patches are 32x32
        
        print("=== LOADING DATA ===")
        # Load predictions
        predictions, binary_predictions, targets, metadata = (
            load_predictions_from_h5(predictions_path)
        )
        
        # Analyze the patches
        analyze_patches(predictions, binary_predictions, targets)
        
        print("\n=== RECONSTRUCTING MAPS ===")
        # Reconstruct maps using simple grid arrangement
        print("Reconstructing raw predictions...")
        pred_map = reconstruct_simple_map(predictions, patch_size)
        
        print("Reconstructing binary predictions...")
        binary_pred_map = reconstruct_simple_map(binary_predictions, patch_size)
        
        print("Reconstructing targets...")
        target_map = reconstruct_simple_map(targets, patch_size)
        
        print("\n=== PLOTTING RESULTS ===")
        print(f"Full map size: {pred_map.shape}")
        
        # Plot the results
        plot_full_maps_simple(
            pred_map, binary_pred_map, target_map, 
            figsize=(20, 8)
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model and the "
              "necessary files exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

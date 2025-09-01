#!/usr/bin/env python3
"""
Script to visualize SSL features using t-SNE
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path

def load_features(features_path):
    """Load features from numpy file"""
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    features = np.load(features_path)
    print(f"✅ Features loaded: {features.shape}")
    return features

def load_image_paths(paths_path):
    """Load image paths from text file"""
    if not os.path.exists(paths_path):
        print(f"⚠️  Image paths file not found: {paths_path}")
        return None
    
    with open(paths_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    print(f"✅ Image paths loaded: {len(image_paths)}")
    return image_paths

def run_tsne(features, n_components=2, perplexity=30, random_state=42):
    """Run t-SNE dimensionality reduction"""
    print(f"🔄 Running t-SNE with perplexity={perplexity}...")
    
    # Use compatible parameters for different sklearn versions
    tsne_params = {
        'n_components': n_components,
        'perplexity': perplexity,
        'random_state': random_state,
        'verbose': 1
    }
    
    # Add n_iter if supported (for older sklearn versions)
    try:
        tsne = TSNE(**tsne_params, n_iter=1000)
    except TypeError:
        # Remove n_iter for newer sklearn versions
        tsne = TSNE(**tsne_params)
    
    features_2d = tsne.fit_transform(features)
    print(f"✅ t-SNE completed: {features_2d.shape}")
    
    return features_2d

def run_pca(features, n_components=2):
    """Run PCA for comparison"""
    print(f"🔄 Running PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"✅ PCA completed: {features_pca.shape}")
    print(f"   Explained variance: {explained_variance}")
    
    return features_pca, explained_variance

def create_tsne_plot(features_2d, image_paths=None, save_path=None, title="t-SNE Visualization"):
    """Create t-SNE visualization plot"""
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot
    scatter = plt.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        alpha=0.6, 
        s=30,
        c=np.arange(len(features_2d)),  # Color by index
        cmap='viridis'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sample Index', rotation=270, labelpad=20)
    
    # Customize plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Features: {features_2d.shape[0]} samples\nDimensions: {features_2d.shape[1]}D → 2D"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.show()

def create_pca_plot(features_pca, explained_variance, save_path=None, title="PCA Visualization"):
    """Create PCA visualization plot"""
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot
    scatter = plt.scatter(
        features_pca[:, 0], 
        features_pca[:, 1], 
        alpha=0.6, 
        s=30,
        c=np.arange(len(features_pca)),
        cmap='plasma'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sample Index', rotation=270, labelpad=20)
    
    # Customize plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({explained_variance[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_variance[1]:.1%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Features: {features_pca.shape[0]} samples\nExplained variance: {sum(explained_variance):.1%}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.show()

def analyze_feature_quality(features):
    """Analyze feature quality metrics"""
    print("\n📊 Feature Quality Analysis:")
    print("=" * 50)
    
    # Basic statistics
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_norm = np.linalg.norm(features, axis=1)
    
    print(f"Feature Statistics:")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {np.mean(feature_mean):.4f}")
    print(f"  Std: {np.mean(feature_std):.4f}")
    print(f"  Norm mean: {np.mean(feature_norm):.4f}")
    print(f"  Norm std: {np.std(feature_norm):.4f}")
    
    # Feature distribution
    print(f"\nFeature Distribution:")
    print(f"  Min: {np.min(features):.4f}")
    print(f"  Max: {np.max(features):.4f}")
    print(f"  Range: {np.max(features) - np.min(features):.4f}")
    
    # Sparsity analysis
    zero_features = np.sum(features == 0, axis=1)
    print(f"\nSparsity Analysis:")
    print(f"  Zero features per sample (mean): {np.mean(zero_features):.2f}")
    print(f"  Zero features per sample (std): {np.std(zero_features):.2f}")
    
    # Feature correlation
    feature_corr = np.corrcoef(features.T)
    print(f"\nFeature Correlation:")
    print(f"  Correlation matrix shape: {feature_corr.shape}")
    print(f"  Mean correlation: {np.mean(feature_corr[np.triu_indices_from(feature_corr, k=1)]):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Visualize SSL features using t-SNE and PCA")
    parser.add_argument("--features_path", type=str, required=True,
                       help="Path to features .npy file")
    parser.add_argument("--paths_path", type=str, default=None,
                       help="Path to image paths .txt file (optional)")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="Directory to save visualization plots")
    parser.add_argument("--perplexity", type=int, default=30,
                       help="t-SNE perplexity parameter")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--no_pca", action="store_true",
                       help="Skip PCA visualization")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 SSL Features Visualization Tool")
    print("=" * 50)
    
    # Load features
    features = load_features(args.features_path)
    
    # Load image paths if provided
    image_paths = None
    if args.paths_path:
        image_paths = load_image_paths(args.paths_path)
    
    # Analyze feature quality
    analyze_feature_quality(features)
    
    # Run t-SNE
    print("\n🎨 Generating Visualizations:")
    print("=" * 50)
    
    # t-SNE visualization
    features_tsne = run_tsne(
        features, 
        perplexity=args.perplexity, 
        random_state=args.random_state
    )
    
    tsne_save_path = os.path.join(args.output_dir, "tsne_visualization.png")
    create_tsne_plot(
        features_tsne, 
        image_paths, 
        tsne_save_path,
        title="t-SNE: SSL Features Visualization"
    )
    
    # PCA visualization (optional)
    if not args.no_pca:
        features_pca, explained_variance = run_pca(features)
        
        pca_save_path = os.path.join(args.output_dir, "pca_visualization.png")
        create_pca_plot(
            features_pca, 
            explained_variance, 
            pca_save_path,
            title="PCA: SSL Features Visualization"
        )
    
    print(f"\n🎉 Visualization completed!")
    print(f"📁 Plots saved to: {args.output_dir}")
    
    # Save 2D coordinates for further analysis
    tsne_coords_path = os.path.join(args.output_dir, "tsne_coordinates.npy")
    np.save(tsne_coords_path, features_tsne)
    print(f"💾 t-SNE coordinates saved to: {tsne_coords_path}")

if __name__ == "__main__":
    main()


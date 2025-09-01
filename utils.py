#!/usr/bin/env python3
"""
Utility functions for SSL feature extraction and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

def visualize_features(features, save_path=None):
    """Visualize features using t-SNE"""
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
    plt.title("t-SNE visualization of SSL features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def compute_similarity_matrix(features):
    """Compute cosine similarity matrix"""
    return cosine_similarity(features)

def find_similar_images(similarity_matrix, query_idx, top_k=5):
    """Find most similar images to a query image"""
    similarities = similarity_matrix[query_idx]
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    top_similarities = similarities[top_indices]
    return list(zip(top_indices, top_similarities))


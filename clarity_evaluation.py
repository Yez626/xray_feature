"""
Evaluation framework for Clarity project
Implements Recall@1 evaluation and anomaly classification metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class ClarityEvaluator:
    """
    Comprehensive evaluation framework for Clarity project
    """
    
    def __init__(self, anomaly_types: List[str] = None):
        """
        Initialize evaluator
        
        Args:
            anomaly_types: List of anomaly types for classification
        """
        if anomaly_types is None:
            self.anomaly_types = ["Grade A", "Incomplete", "Counterfeit", "Defective"]
        else:
            self.anomaly_types = anomaly_types
        
        self.results = {}
    
    def evaluate_recall_at_1(
        self,
        similarity_matrices: Dict[str, np.ndarray],
        ground_truth_mapping: Dict[str, str],
        barcode_galleries: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate Recall@1 for each barcode
        
        Args:
            similarity_matrices: Dictionary mapping barcodes to similarity matrices
            ground_truth_mapping: Dictionary mapping query paths to ground truth gallery paths
            barcode_galleries: Dictionary mapping barcodes to gallery image paths
        
        Returns:
            Dictionary with Recall@1 scores per barcode
        """
        recall_scores = {}
        
        for barcode, similarity_matrix in similarity_matrices.items():
            if barcode not in barcode_galleries:
                continue
            
            gallery_paths = barcode_galleries[barcode]
            correct = 0
            total = 0
            
            # For each query in the ground truth mapping
            for query_path, gt_path in ground_truth_mapping.items():
                if gt_path not in gallery_paths:
                    continue
                
                # Find query index in similarity matrix
                query_idx = None
                for i, path in enumerate(gallery_paths):
                    if path == query_path:
                        query_idx = i
                        break
                
                if query_idx is None:
                    continue
                
                # Get most similar item (excluding self)
                similarities = similarity_matrix[query_idx]
                # Set self-similarity to -1 to exclude it
                similarities[query_idx] = -1
                
                most_similar_idx = np.argmax(similarities)
                most_similar_path = gallery_paths[most_similar_idx]
                
                if most_similar_path == gt_path:
                    correct += 1
                
                total += 1
            
            recall_scores[barcode] = correct / total if total > 0 else 0.0
        
        return recall_scores
    
    def evaluate_anomaly_classification(
        self,
        predictions: List[str],
        ground_truth: List[str],
        confidence_scores: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Evaluate anomaly classification performance
        
        Args:
            predictions: Predicted anomaly types
            ground_truth: Ground truth anomaly types
            confidence_scores: Confidence scores for predictions (optional)
        
        Returns:
            Dictionary with classification metrics
        """
        # Convert to numerical labels
        label_to_idx = {label: idx for idx, label in enumerate(self.anomaly_types)}
        
        y_true = [label_to_idx[label] for label in ground_truth]
        y_pred = [label_to_idx[label] for label in predictions]
        
        # Compute metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Compute per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1': f1,
            'confusion_matrix': cm,
            'per_class_precision': precision_per_class.tolist(),
            'per_class_recall': recall_per_class.tolist(),
            'per_class_f1': f1_per_class.tolist(),
            'class_names': self.anomaly_types
        }
        
        # Add confidence analysis if provided
        if confidence_scores is not None:
            results['mean_confidence'] = np.mean(confidence_scores)
            results['confidence_std'] = np.std(confidence_scores)
            
            # Confidence by correctness
            correct_mask = np.array(y_true) == np.array(y_pred)
            if np.any(correct_mask):
                results['correct_confidence_mean'] = np.mean(np.array(confidence_scores)[correct_mask])
            if np.any(~correct_mask):
                results['incorrect_confidence_mean'] = np.mean(np.array(confidence_scores)[~correct_mask])
        
        return results
    
    def evaluate_similarity_thresholds(
        self,
        similarity_scores: List[float],
        ground_truth_labels: List[str],
        thresholds: List[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance at different similarity thresholds
        
        Args:
            similarity_scores: List of similarity scores
            ground_truth_labels: List of ground truth labels
            thresholds: List of thresholds to evaluate (optional)
        
        Returns:
            Dictionary with metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = {}
        
        for threshold in thresholds:
            # Predict anomaly based on threshold
            predictions = []
            for score in similarity_scores:
                if score < threshold:
                    predictions.append("Anomaly")  # Low similarity = anomaly
                else:
                    predictions.append("Normal")  # High similarity = normal
            
            # Convert to binary classification
            y_true = [1 if label != "Grade A" else 0 for label in ground_truth_labels]
            y_pred = [1 if pred == "Anomaly" else 0 for pred in predictions]
            
            # Compute metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            results[f'threshold_{threshold:.1f}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.anomaly_types,
            yticklabels=self.anomaly_types
        )
        plt.title('Confusion Matrix - Anomaly Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_similarity_distribution(
        self,
        similarity_scores: List[float],
        labels: List[str],
        save_path: str = None
    ):
        """
        Plot distribution of similarity scores by label
        
        Args:
            similarity_scores: List of similarity scores
            labels: List of corresponding labels
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 6))
        
        for label in self.anomaly_types:
            scores = [score for score, lab in zip(similarity_scores, labels) if lab == label]
            if scores:
                plt.hist(scores, alpha=0.7, label=label, bins=20)
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarity Scores by Anomaly Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(
        self,
        recall_scores: Dict[str, float],
        classification_metrics: Dict[str, float],
        similarity_metrics: Dict[str, Dict[str, float]] = None
    ) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            recall_scores: Recall@1 scores per barcode
            classification_metrics: Classification performance metrics
            similarity_metrics: Similarity threshold metrics (optional)
        
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append("=" * 60)
        report.append("CLARITY PROJECT EVALUATION REPORT")
        report.append("=" * 60)
        
        # Recall@1 Results
        report.append("\n1. RECALL@1 EVALUATION")
        report.append("-" * 30)
        for barcode, score in recall_scores.items():
            report.append(f"Barcode {barcode}: {score:.3f}")
        
        avg_recall = np.mean(list(recall_scores.values()))
        report.append(f"Average Recall@1: {avg_recall:.3f}")
        
        # Classification Results
        report.append("\n2. ANOMALY CLASSIFICATION")
        report.append("-" * 30)
        report.append(f"Overall Precision: {classification_metrics['overall_precision']:.3f}")
        report.append(f"Overall Recall: {classification_metrics['overall_recall']:.3f}")
        report.append(f"Overall F1-Score: {classification_metrics['overall_f1']:.3f}")
        
        # Per-class results
        report.append("\nPer-class Performance:")
        for i, class_name in enumerate(classification_metrics['class_names']):
            precision = classification_metrics['per_class_precision'][i]
            recall = classification_metrics['per_class_recall'][i]
            f1 = classification_metrics['per_class_f1'][i]
            report.append(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # Similarity threshold results
        if similarity_metrics:
            report.append("\n3. SIMILARITY THRESHOLD ANALYSIS")
            report.append("-" * 30)
            for threshold, metrics in similarity_metrics.items():
                report.append(f"{threshold}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def run_comprehensive_evaluation(
    model,
    test_loader,
    gallery_loader,
    ground_truth_mapping: Dict[str, str],
    barcode_mapping: Dict[str, str]
) -> Dict:
    """
    Run comprehensive evaluation of the Clarity model
    
    Args:
        model: Trained Clarity model
        test_loader: DataLoader for test images
        gallery_loader: DataLoader for gallery images
        ground_truth_mapping: Mapping from query to ground truth gallery paths
        barcode_mapping: Mapping from image paths to barcodes
    
    Returns:
        Dictionary with all evaluation results
    """
    evaluator = ClarityEvaluator()
    
    # Build gallery
    from clarity_models import SimilarityMatcher
    matcher = SimilarityMatcher(model)
    matcher.build_gallery(gallery_loader, barcode_mapping)
    
    # Extract features for test set
    model.eval()
    test_features = []
    test_paths = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, paths = batch
                labels = ["Unknown"] * len(paths)
            
            features = model.extract_features(images)
            test_features.append(features.cpu())
            test_paths.extend(paths)
            test_labels.extend(labels)
    
    test_features = torch.cat(test_features, dim=0)
    
    # Group by barcode for Recall@1 evaluation
    barcode_groups = defaultdict(list)
    for i, path in enumerate(test_paths):
        barcode = barcode_mapping.get(path, "unknown")
        barcode_groups[barcode].append(i)
    
    # Evaluate Recall@1 for each barcode
    recall_scores = {}
    for barcode, indices in barcode_groups.items():
        if len(indices) < 2:  # Need at least 2 items for evaluation
            continue
        
        # Compute similarity matrix for this barcode
        barcode_features = test_features[indices]
        similarity_matrix = torch.mm(barcode_features, barcode_features.t()).numpy()
        
        # Create ground truth mapping for this barcode
        barcode_gt_mapping = {
            test_paths[i]: ground_truth_mapping.get(test_paths[i], "")
            for i in indices
        }
        
        # Evaluate Recall@1
        barcode_recall = evaluator.evaluate_recall_at_1(
            {barcode: similarity_matrix},
            barcode_gt_mapping,
            {barcode: [test_paths[i] for i in indices]}
        )
        
        recall_scores.update(barcode_recall)
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(recall_scores, {})
    
    return {
        'recall_scores': recall_scores,
        'evaluation_report': report,
        'test_features': test_features,
        'test_paths': test_paths,
        'test_labels': test_labels
    }

"""
Evaluation Metrics for Deepfake Detection

Includes ROC AUC, EER, accuracy, precision, recall, F1, and confusion matrix.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        labels: True labels (0=real, 1=fake)
        scores: Prediction scores (higher = more likely fake)
        
    Returns:
        Tuple of (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find where FPR and FNR are closest
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold


def compute_metrics(labels: np.ndarray,
                   predictions: np.ndarray,
                   probabilities: np.ndarray = None) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        probabilities: Prediction probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='binary', zero_division=0),
        'recall': recall_score(labels, predictions, average='binary', zero_division=0),
        'f1': f1_score(labels, predictions, average='binary', zero_division=0),
    }
    
    # Add AUC and EER if probabilities provided
    if probabilities is not None:
        # Use probability of fake class (class 1)
        if probabilities.ndim == 2:
            scores = probabilities[:, 1]
        else:
            scores = probabilities
        
        try:
            metrics['auc'] = roc_auc_score(labels, scores)
            metrics['eer'], metrics['eer_threshold'] = compute_eer(labels, scores)
        except Exception as e:
            metrics['auc'] = 0.0
            metrics['eer'] = 0.0
            metrics['eer_threshold'] = 0.5
    
    return metrics


def plot_confusion_matrix(labels: np.ndarray,
                         predictions: np.ndarray,
                         class_names: list = None,
                         normalize: bool = False,
                         save_path: str = None,
                         title: str = 'Confusion Matrix') -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: Class names for labels
        normalize: Whether to normalize
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    if class_names is None:
        class_names = ['Real', 'Fake']
    
    cm = confusion_matrix(labels, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(labels: np.ndarray,
                  scores: np.ndarray,
                  save_path: str = None,
                  title: str = 'ROC Curve') -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        labels: True labels
        scores: Prediction scores
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(history: Dict[str, list],
                         metrics: list = None,
                         save_path: str = None) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics over epochs
        metrics: Metrics to plot (default: ['loss', 'accuracy'])
        save_path: Path to save plot
        
    Returns:
        matplotlib Figure
    """
    if metrics is None:
        metrics = ['loss', 'accuracy']
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if f'train_{metric}' in history:
            ax.plot(history[f'train_{metric}'], label='Train', marker='o')
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label='Validation', marker='s')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


class MetricsTracker:
    """Helper class to track metrics during training."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auc': [],
            'val_eer': []
        }
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_accuracy': 0.0,
            'best_val_auc': 0.0,
            'best_epoch': 0
        }
    
    def update(self, epoch: int, metrics: Dict[str, float], phase: str = 'train'):
        """Update metrics for current epoch."""
        for key, value in metrics.items():
            history_key = f'{phase}_{key}'
            if history_key in self.history:
                self.history[history_key].append(value)
        
        # Update best metrics (only for validation)
        if phase == 'val':
            if 'loss' in metrics and metrics['loss'] < self.best_metrics['best_val_loss']:
                self.best_metrics['best_val_loss'] = metrics['loss']
                self.best_metrics['best_epoch'] = epoch
            
            if 'accuracy' in metrics and metrics['accuracy'] > self.best_metrics['best_val_accuracy']:
                self.best_metrics['best_val_accuracy'] = metrics['accuracy']
            
            if 'auc' in metrics and metrics['auc'] > self.best_metrics['best_val_auc']:
                self.best_metrics['best_val_auc'] = metrics['auc']
    
    def get_history(self) -> Dict[str, list]:
        """Get full training history."""
        return self.history
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best validation metrics."""
        return self.best_metrics
    
    def save(self, path: str):
        """Save metrics to file."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'history': self.history,
                'best_metrics': self.best_metrics
            }, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.history = data['history']
            self.best_metrics = data['best_metrics']

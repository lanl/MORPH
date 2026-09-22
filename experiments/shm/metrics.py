import torch
import torch.nn as nn

class Metrics():
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    # PyTorch Functions for recall, precision and f1 score
    def recall_pytorch(self, y_true, y_pred, threshold=0.5):
        """Calculate recall for binary classification"""
        y_pred_binary = (y_pred > threshold).float()
        true_positives = torch.sum((y_true * y_pred_binary))
        possible_positives = torch.sum(y_true)
        recall = true_positives / (possible_positives + 1e-8)
        return recall

    def precision_pytorch(self, y_true, y_pred, threshold=0.5):
        """Calculate precision for binary classification"""
        y_pred_binary = (y_pred > threshold).float()
        true_positives = torch.sum((y_true * y_pred_binary))
        predicted_positives = torch.sum(y_pred_binary)
        precision = true_positives / (predicted_positives + 1e-8)
        return precision

    def f1_score_pytorch(self, y_true, y_pred, threshold=0.5):
        """Calculate F1 score for binary classification"""
        precision = self.precision_pytorch(y_true, y_pred, threshold)
        recall = self.recall_pytorch(y_true, y_pred, threshold)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1

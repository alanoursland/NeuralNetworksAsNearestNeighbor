# evaluation/evaluator.py

import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, model, test_data, test_targets, criterion, device):
        self.model = model
        self.test_data = test_data
        self.test_targets = test_targets
        self.criterion = criterion
        self.device = device

    def evaluate(self):
        # save the criterion reduction attribute and set it to none
        old_criterion_reduction = self.criterion.reduction
        self.criterion.reduction = 'none'

        # Switch the model to evaluation mode
        self.model.eval()

        # Move the data and targets to the appropriate device
        test_data = self.test_data.to(self.device)
        self.all_targets = self.test_targets.to(self.device)
        
        # Get the total number of samples in the data loader
        total_samples = test_data.size(0)  # Assuming test_data is a tensor of shape (N, ...)
        # Get the shape of outputs (assuming fixed output size per batch)

        # Preallocate tensors for all outputs, losses, and targets
        self.all_losses = torch.empty((total_samples,), device=self.device)

        with torch.no_grad():
            self.all_outputs = self.model(test_data).detach()
            self.all_losses = self.criterion(self.all_outputs, self.all_targets).detach()

        # restore the old criterion reduction
        self.criterion.reduction = old_criterion_reduction

        return self.all_losses, self.all_outputs, self.all_targets

    def accuracy(self):
        predicted_classes = torch.argmax(self.all_outputs, dim=1)
        predicted_classes_np = predicted_classes.cpu().numpy()
        all_targets_np = self.all_targets.cpu().numpy()
        accuracy = accuracy_score(all_targets_np, predicted_classes_np)
        return accuracy

    def generate_stats(self):
        # Average loss
        avg_loss = self.all_losses.mean().item()

        # Convert predictions (all_outputs) to class labels by taking the argmax along the last dimension
        predicted_classes = torch.argmax(self.all_outputs, dim=1)

        # Convert tensors to numpy for sklearn metrics
        predicted_classes_np = predicted_classes.cpu().numpy()
        all_targets_np = self.all_targets.cpu().numpy()

        # Accuracy
        accuracy = accuracy_score(all_targets_np, predicted_classes_np)

        # Precision, Recall, and F1-Score (assuming this is a classification problem)
        precision = precision_score(all_targets_np, predicted_classes_np, average='weighted', zero_division=0)
        recall = recall_score(all_targets_np, predicted_classes_np, average='weighted', zero_division=0)
        f1 = f1_score(all_targets_np, predicted_classes_np, average='weighted')

        # Confusion Matrix
        confusion = confusion_matrix(all_targets_np, predicted_classes_np)

        # Return stats as a dictionary
        stats = {
            'average_loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion
        }

        return stats

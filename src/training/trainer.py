# training/trainer.py

import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, data_loader, criterion, optimizer, scheduler, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, num_epochs, evaluator=None):
        dataset_size = len(self.data_loader.dataset)
        epoch_losses = torch.zeros(num_epochs)  # List to store loss for each epoch
        if evaluator is not None:
            test_accuracy = torch.zeros(num_epochs)  # List to store accuracy for each epoch
        self.model.train()
        max_accuracy = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, targets in self.data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                epoch_loss += loss.item()

            accuracy = None
            if evaluator is not None:
                evaluator.evaluate()
                accuracy = evaluator.accuracy()
                test_accuracy[epoch] = accuracy

            average_loss = epoch_loss / dataset_size
            epoch_losses[epoch] = average_loss  # Save the loss for the current epoch
            if (epoch+1) % 10 == 0 or accuracy > max_accuracy:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}, Accuracy {accuracy:.6f}')
            if accuracy > max_accuracy:
                max_accuracy = accuracy
        return epoch_losses, test_accuracy  # Return the list of losses

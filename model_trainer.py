import torch
import torch.nn as nn
import torch.optim as optim
import time


class ModelTrainer:
    def __init__(self, models, dataloaders, device=None, learning_rate=0.001, epochs=5):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.models = models
        self.train_loader, self.val_loader = dataloaders
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

        # Initialize optimizers for each model
        self.optimizers = {
            model_name: optim.Adam(model.parameters(), lr=learning_rate)
            for model_name, model in self.models.items()
        }

    def fine_tune(self, callback=None):
        """Fine-tune all models using the integrated dataset."""
        results = {}

        for model_name, model in self.models.items():
            print(f"\nFine-tuning {model_name}...")
            model.train()
            optimizer = self.optimizers[model_name]

            best_val_accuracy = 0.0
            best_model_state_dict = None

            for epoch in range(self.epochs):
                start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0

                # Training phase
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                # Validation phase
                val_loss, val_accuracy = self.validate_model(model)

                epoch_loss = running_loss / len(self.train_loader.dataset)
                epoch_time = time.time() - start_time

                print(
                    f"Epoch {epoch + 1}/{self.epochs} | Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%} | Time: {epoch_time:.2f}s")

                if callback:
                    callback(model_name, epoch + 1, self.epochs, val_accuracy)

                # Save the best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model_state_dict = model.state_dict().copy()

            # Save the best model
            if best_model_state_dict is not None:
                model.load_state_dict(best_model_state_dict)

                # Save model checkpoint
                checkpoint = {
                    'model_state_dict': best_model_state_dict,
                    'val_accuracy': best_val_accuracy,
                    'fine_tuned_date': time.strftime("%Y-%m-%d %H:%M:%S")
                }

                torch.save(checkpoint, f"model/{model_name}_fine_tuned.pth")
                print(f"Model {model_name} saved with validation accuracy: {best_val_accuracy:.4f}")

            results[model_name] = best_val_accuracy

        return results

    def validate_model(self, model):
        """Validate a model on the validation set."""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(self.val_loader.dataset)
        val_accuracy = correct / total

        model.train()
        return val_loss, val_accuracy
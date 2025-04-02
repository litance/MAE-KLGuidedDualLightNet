import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms
import os
import scipy.stats


class DatasetManager:
    def __init__(self, main_dataset_path, temp_dataset_path, batch_size=64):
        self.main_dataset_path = main_dataset_path
        self.temp_dataset_path = temp_dataset_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Initialize datasets
        self.main_dataset = self._load_dataset(main_dataset_path)
        self.temp_dataset = self._load_dataset(temp_dataset_path)

    def _load_dataset(self, path):
        """Load a dataset from the given path."""
        if os.path.exists(path):
            try:
                # For ImageFolder format datasets
                return datasets.ImageFolder(path, transform=self.transform)
            except:
                # If not ImageFolder format, you may need to implement custom dataset loading
                print(f"Warning: Could not load dataset from {path} as ImageFolder.")
                return None
        else:
            print(f"Warning: Dataset path {path} does not exist.")
            return None

    def calculate_kl_divergence(self):
        """Calculate KL divergence between temp_dataset and main_dataset class distributions."""
        if self.main_dataset is None or self.temp_dataset is None:
            return 0.0

        # Get class distributions
        main_classes = [label for _, label in self.main_dataset.samples]
        temp_classes = [label for _, label in self.temp_dataset.samples]

        # Count classes
        main_class_counts = np.bincount(main_classes, minlength=10)
        temp_class_counts = np.bincount(temp_classes, minlength=10)

        # Convert to probabilities
        main_probs = main_class_counts / np.sum(main_class_counts)
        temp_probs = temp_class_counts / np.sum(temp_class_counts)

        # Replace zeros to avoid division by zero in KL divergence calculation
        main_probs = np.where(main_probs == 0, 1e-10, main_probs)
        temp_probs = np.where(temp_probs == 0, 1e-10, temp_probs)

        # Calculate KL divergence
        kl_div = scipy.stats.entropy(temp_probs, main_probs)

        return kl_div

    def calculate_integration_proportion(self):
        """Calculate what percentage of mainDataset the tempDataset should be."""
        kl_div = self.calculate_kl_divergence()

        # Convert KL divergence to a proportion using a simple formula
        # Higher KL divergence means more dissimilar distributions, so we may want less integration
        # This is just one possible approach - adjust as needed for your specific case
        p = 1.0 / (1.0 + kl_div)

        # Limit p to a reasonable range (e.g., 5% to 30%)
        p = max(0.05, min(0.3, p))

        return p

    def create_integrated_dataset(self):
        """Create an integrated dataset based on calculated proportion."""
        if self.main_dataset is None or self.temp_dataset is None:
            return self.main_dataset

        p = self.calculate_integration_proportion()

        # Calculate how many samples from tempDataset to include
        main_size = len(self.main_dataset)
        temp_size = len(self.temp_dataset)
        target_temp_size = int(p * main_size)

        # Limit samples from temp_dataset
        actual_temp_size = min(temp_size, target_temp_size)
        if actual_temp_size < temp_size:
            # Take a random subset of temp_dataset
            indices = torch.randperm(temp_size)[:actual_temp_size]
            temp_subset = Subset(self.temp_dataset, indices)
        else:
            temp_subset = self.temp_dataset

        # Combine datasets
        integrated_dataset = ConcatDataset([self.main_dataset, temp_subset])

        print(
            f"Integration completed: Added {actual_temp_size} samples from tempDataset, which is {p:.1%} of mainDataset size ({main_size}).")

        return integrated_dataset

    def get_dataloaders(self):
        """Get DataLoaders for training."""
        integrated_dataset = self.create_integrated_dataset()

        # Split for training and validation (e.g., 80% training, 20% validation)
        dataset_size = len(integrated_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            integrated_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
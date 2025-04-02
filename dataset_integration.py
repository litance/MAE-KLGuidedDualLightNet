import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms
import os
import scipy.stats
import pickle


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
        if not os.path.exists(path):
            print(f"Warning: Dataset path {path} does not exist.")
            return None

        # 检查是否为CIFAR-10格式
        if 'cifar-10-batches-py' in path:
            try:
                # 使用CIFAR10类加载数据集
                parent_dir = os.path.dirname(os.path.dirname(path))
                print(f"Loading CIFAR-10 dataset from {parent_dir}")
                return datasets.CIFAR10(
                    root=parent_dir,  # 包含cifar-10-batches-py文件夹的父目录
                    train=True,  # 使用训练数据进行微调
                    download=True,  # 如果不存在则下载
                    transform=self.transform
                )
            except Exception as e:
                print(f"Error loading CIFAR-10 dataset: {e}")
                return self._create_dummy_dataset()
        else:
            # 尝试以ImageFolder格式加载
            try:
                return datasets.ImageFolder(path, transform=self.transform)
            except Exception as e:
                print(f"Warning: Could not load dataset from {path} as ImageFolder: {e}")

                # 创建虚拟数据集作为备用
                return self._create_dummy_dataset()

    def _create_dummy_dataset(self):
        """创建一个简单的虚拟数据集用于测试"""
        print("创建虚拟数据集，每类10个样本...")

        class DummyDataset(Dataset):
            def __init__(self, transform=None):
                self.transform = transform
                self.samples = [(None, i % 10) for i in range(100)]  # 每类10个样本
                self.targets = [i % 10 for i in range(100)]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                # 生成随机32x32图像
                image = torch.rand(3, 32, 32)
                label = self.targets[idx]
                return image, label

        return DummyDataset(transform=self.transform)

    def calculate_kl_divergence(self):
        """Calculate KL divergence between temp_dataset and main_dataset class distributions."""
        if self.main_dataset is None or self.temp_dataset is None:
            return 0.0

        # 获取类别分布 - 处理不同的数据集格式
        if hasattr(self.main_dataset, 'samples'):
            main_classes = [label for _, label in self.main_dataset.samples]
        elif hasattr(self.main_dataset, 'targets'):
            main_classes = self.main_dataset.targets
        else:
            print("Warning: Cannot get class distribution from main_dataset")
            return 0.0

        if hasattr(self.temp_dataset, 'samples'):
            temp_classes = [label for _, label in self.temp_dataset.samples]
        elif hasattr(self.temp_dataset, 'targets'):
            temp_classes = self.temp_dataset.targets
        else:
            print("Warning: Cannot get class distribution from temp_dataset")
            return 0.0

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
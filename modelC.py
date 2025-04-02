import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

from sympy import false
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datetime import datetime

torch.manual_seed(42)


def load_datasets():
    print("Loading CIFAR-10 datasets...")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


log_dir = f"logs/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, channels)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1)
        return x * y


class MobileNetLSTMSTAM(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetLSTMSTAM, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.mobilenet.features[-1] = nn.Identity()
        self.stam = SEBlock(160)
        self.lstm = nn.LSTM(160, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.stam(x)
        x = x.mean([2, 3])
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.fc(self.dropout(x[:, -1, :]))
        return x


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset = load_datasets()

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MobileNetLSTMSTAM(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    num_epochs = 100
    best_val_accuracy = 0.0
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                tepoch.set_postfix(loss=f"{total_loss / (tepoch.n + 1):.4f}",
                                   acc=f"{correct / total:.4f}")

        train_avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_avg_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        writer.add_scalar("Loss/train", train_avg_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        epoch_end = time.time()
        elapsed_time = epoch_end - epoch_start

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_avg_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed_time:.2f}s\n")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_accuracy
            }, "model/modelC.pth")
            print(f"New best model saved with val accuracy: {val_accuracy:.4f}")

        if train_accuracy > 0.99:
            print("Training accuracy reached 99%, stopping early.")
            break

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results - Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.4f}")

    writer.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curvesC.png")
    plt.show()


if __name__ == "__main__":
    main()
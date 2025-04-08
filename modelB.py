import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
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


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            stride=stride, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class STAMBlock(nn.Module):

    def __init__(self, in_channels, reduction=8):
        super(STAMBlock, self).__init__()
        # Spatial attention
        self.spatial_pool = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        spatial_att = self.spatial_pool(x)
        x_spatial = x * spatial_att

        avg_pool = self.channel_avg_pool(x_spatial)
        max_pool = self.channel_max_pool(x_spatial)

        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        channel_att = self.sigmoid(avg_out + max_out)

        return x_spatial * channel_att


class LightweightCNN(nn.Module):

    def __init__(self, input_channels=3, base_channels=32):
        super(LightweightCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True)
        )

        self.block1 = self._make_layer(base_channels, base_channels * 2, stride=2)
        self.block2 = self._make_layer(base_channels * 2, base_channels * 4, stride=2)
        self.block3 = self._make_layer(base_channels * 4, base_channels * 8, stride=2)

        self.stam1 = STAMBlock(base_channels * 2)
        self.stam2 = STAMBlock(base_channels * 4)
        self.stam3 = STAMBlock(base_channels * 8)

        self.output_channels = base_channels * 8

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(DepthwiseSeparableConv(in_channels, out_channels, stride))
        layers.append(DepthwiseSeparableConv(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.block1(x)
        x = self.stam1(x)

        x = self.block2(x)
        x = self.stam2(x)

        x = self.block3(x)
        x = self.stam3(x)

        return x


class LightCNN_STAM(nn.Module):
    def __init__(self, num_classes=10):
        super(LightCNN_STAM, self).__init__()

        self.cnn = LightweightCNN(input_channels=3, base_channels=32)
        feature_dim = self.cnn.output_channels

        # Removed LSTM

        self.final_stam = STAMBlock(feature_dim)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feature_dim, num_classes)  # Changed from 512 to feature_dim

    def forward(self, x):
        batch_size = x.size(0)

        x = self.cnn(x)

        x = self.final_stam(x)

        x = x.mean([2, 3])  # Global average pooling

        # Removed LSTM processing

        # Direct connection to FC layer
        x = self.fc(self.dropout(x))

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

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = LightCNN_STAM(num_classes=10).to(device)  # Updated class name
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    num_epochs = 50
    best_val_accuracy = 0.0
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

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
            }, "model/modelB.pth")
            print(f"New best model saved with val accuracy: {val_accuracy:.4f}")

        if train_accuracy > 0.8:
            print(
                "Training accuracy reached 80%, stopping early.")  # Changed from 99% to 80% to match your original code
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
    plt.savefig("training_curvesD.png")
    plt.show()


if __name__ == "__main__":
    main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision.transforms as transforms
import torchvision.models as models
#import kagglehub
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datetime import datetime
from dataload import ASLDataset, transform
#from PIL import Image

#path = kagglehub.dataset_download("ayuraj/asl-dataset")
path = "../dataset/asl_dataset"

log_dir = f"logs/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


dataset = ASLDataset(path, transform=transform)


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
    def __init__(self, num_classes=len(dataset.classes)):
        super(MobileNetLSTMSTAM, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.mobilenet.features[-1] = nn.Identity()
        self.stam = SEBlock(160)
        self.lstm = nn.LSTM(160, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.stam(x)
        x = x.mean([2, 3])
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.fc(self.dropout(x[:, -1, :]))
        return x


device = torch.device("cuda")
model = MobileNetLSTMSTAM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_losses, train_accuracies = [], []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(dataloader)

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                avg_loss = total_loss / (tepoch.n + 1)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                accuracy = correct / total

                tepoch.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.4f}")

        epoch_end = time.time()
        elapsed_time = epoch_end - epoch_start

        current_lr = optimizer.param_groups[0]['lr']
        avg_epoch_loss = total_loss / num_batches
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(accuracy)

        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        print(f"\nEpoch {epoch + 1}/{num_epochs} | "
              f"Avg Loss: {avg_epoch_loss:.4f} | "
              f"Accuracy: {accuracy:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed_time:.2f}s\n")

        if accuracy > 0.99:
            break
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': len(dataset.classes)
        }, "model/model.pth")

        writer.close()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label="Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy Curve")
        plt.legend()
        plt.show()

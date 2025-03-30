import os
import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision.transforms as transforms
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from dataload import ASLDataset, transform
#from PIL import Image

path = "dataset/test_dataset/asl_dataset"

log_dir = f"logs/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

dataset = ASLDataset(path, transform=transform)


class ESNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ESNetBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ESNet(nn.Module):
    def __init__(self):
        super(ESNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            ESNetBlock(32, 64, stride=1),
            ESNetBlock(64, 128, stride=2),
            ESNetBlock(128, 256, stride=2),
            ESNetBlock(256, 320, stride=2)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        return x


class DepthwiseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DepthwiseLSTM, self).__init__()
        self.pointwise_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.depthwise_conv(x.permute(0, 2, 1))  # Change shape for depthwise conv
        x = x.permute(0, 2, 1)  # Restore shape
        x, _ = self.lstm(x)
        return x[:, -1, :]


class ESNetLSTM(nn.Module):
    def __init__(self, num_classes=len(dataset.classes)):
        super(ESNetLSTM, self).__init__()
        self.esnet = ESNet()
        self.lstm = DepthwiseLSTM(320, 320)
        self.fc = nn.Linear(320, num_classes)  # Change from 512 to 320

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.esnet(x)
        x = x.mean([2, 3])
        x = x.view(batch_size, 1, -1)
        x = self.lstm(x)
        x = self.fc(x)
        return x


device = torch.device("cuda")
model = ESNetLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_losses, train_accuracies = [], []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss, correct, total = 0.0, 0, 0
        num_batches = len(dataloader)

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                tepoch.set_postfix(loss=f"{total_loss / (tepoch.n + 1):.4f}", acc=f"{correct / total:.4f}")

        train_losses.append(total_loss / num_batches)
        train_accuracies.append(correct / total)

        writer.add_scalar("Loss/train", total_loss / num_batches, epoch)
        writer.add_scalar("Accuracy/train", correct / total, epoch)
        print(f"Epoch {epoch + 1} | Loss: {total_loss / num_batches:.4f} | Accuracy: {correct / total:.4f}")
        if correct / total > 0.99:
            break
        torch.save({'model_state_dict': model.state_dict()}, "model/modelB.pth")

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

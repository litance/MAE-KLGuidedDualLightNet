import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import kagglehub
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from PIL import Image

path = "C:\\Users\\User\\PycharmProjects\\signTest\\dataset\\asl_dataset"

log_dir = f"logs/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

class ASLDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted([cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))])
        for label, cls in enumerate(self.classes):
            class_path = os.path.join(root, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

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
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.features[18] = nn.Identity()
        self.stam = SEBlock(320)
        self.lstm = nn.LSTM(320, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.mobilenet.features(x)
        x = self.stam(x)
        x = x.mean([2, 3])
        x = x.view(batch_size, 1, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


device = torch.device("cuda")
model = MobileNetLSTMSTAM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_losses = []
train_accuracies = []

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

    torch.save(model.state_dict(), "model.pth")

writer.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")
plt.legend()
plt.show()
import os
import torch
import torch.nn as nn
import torchvision.models as models
#import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datetime import datetime
from dataload import ASLDataset, transform
import mediapipe as mp
import numpy as np

path = "dataset/test_dataset/asl_dataset"

log_dir = f"logs/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

dataset = ASLDataset(path, transform=transform)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def extract_keypoints(image):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
    image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image

    results = hands.process(image)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        return torch.tensor(keypoints, dtype=torch.float32)
    return torch.zeros(21 * 3)


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


class MultiModalModel(nn.Module):
    def __init__(self, num_classes=len(dataset.classes)):
        super(MultiModalModel, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        self.mobilenet_features = nn.Sequential(
            *list(self.mobilenet.children())[:-2],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.keypoint_fc = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(960 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, keypoints):
        image_features = self.mobilenet_features(image)
        keypoint_features = self.keypoint_fc(keypoints)
        features = torch.cat((image_features, keypoint_features), dim=1)
        output = self.fc(features)
        return output


device = torch.device("cuda")
model = MultiModalModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

num_epochs = 100
batch_size = 256
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    train_losses = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        keypoints_list = []
        for img in images:
            kp = extract_keypoints(img)
            keypoints_list.append(kp)

        keypoints = torch.stack(keypoints_list).to(device)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, keypoints)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch+1} | Loss: {train_losses/len(dataloader):.4f} | Accuracy: {accuracy:.4f}")
    writer.add_scalar("Loss/train", train_losses/len(dataloader), epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)

    if accuracy > 0.99:
        break

    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': len(dataset.classes)
    }, "model/modelC.pth")

writer.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracy) + 1), accuracy, marker='o', label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")
plt.legend()
plt.show()
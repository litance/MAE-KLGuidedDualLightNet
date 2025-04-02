import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from modelC import MobileNetLSTMSTAM  # 确保modelC.py在同一目录下

# 固定随机种子保证可复现性
torch.manual_seed(42)


def load_test_dataset():
    """加载CIFAR-10测试集（与训练相同的预处理）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    return test_dataset


def evaluate_model(model, test_loader, device):
    """评估模型并返回预测结果和真实标签"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def generate_classification_report(labels, preds, class_names):
    """生成并打印分类报告"""
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print("=" * 80)
    print("Classification Report:")
    print("=" * 80)
    print(report)

    # 保存报告到文件
    with open("classification_report.txt", "w") as f:
        f.write(report)


def plot_confusion_matrix(labels, preds, class_names):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()


def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载测试集
    test_dataset = load_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 初始化模型并加载预训练权重
    model = MobileNetLSTMSTAM(num_classes=10).to(device)
    checkpoint = torch.load("model/modelC.pth", map_location=device)  # 修改为你的模型路径
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val accuracy {checkpoint['val_accuracy']:.4f}")

    # 评估模型
    preds, labels = evaluate_model(model, test_loader, device)

    # 计算并打印准确率
    accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
    print(f"\nAbsolute Test Accuracy: {accuracy * 100:.2f}%")

    # 生成详细报告
    class_names = test_dataset.classes
    generate_classification_report(labels, preds, class_names)
    plot_confusion_matrix(labels, preds, class_names)

    # 保存预测结果（可选）
    results_df = pd.DataFrame({"True Label": labels, "Predicted Label": preds})
    results_df.to_csv("test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from main import MobileNetLSTMSTAM, ASLDataset, transform  # 确保 import 你的模型 & 数据集

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
modelA = MobileNetLSTMSTAM().to(device)
modelA.load_state_dict(torch.load("model.pth", map_location=device))
modelA.eval()

modelB = MobileNetLSTMSTAM().to(device)
modelB.load_state_dict(torch.load("model.pth", map_location=device))
modelB.eval()

# 加载测试数据集
test_path = "C:\\Users\\User\\PycharmProjects\\signTest\\dataset\\asl_dataset"
test_dataset = ASLDataset(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 获取类别数量
num_classes = len(test_dataset.classes)

# 存储真实标签和预测分数
y_true = []
scores_A = []
scores_B = []

# 遍历测试数据
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs_A = modelA(images)
        outputs_B = modelB(images)

        prob_A = torch.softmax(outputs_A, dim=1)  # 转换为概率
        prob_B = torch.softmax(outputs_B, dim=1)  # 转换为概率

        y_true.extend(labels.cpu().numpy())
        scores_A.extend(prob_A.cpu().numpy())
        scores_B.extend(prob_B.cpu().numpy())

# 转换为 NumPy 数组
y_true = np.array(y_true)
scores_A = np.array(scores_A)
scores_B = np.array(scores_B)

# 进行 One-vs-Rest 处理
y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

# 存储 AUC 分数
auc_scores_A = []
auc_scores_B = []

# 画 ROC 曲线
plt.figure(figsize=(10, 6))

for i in range(num_classes):
    if len(np.unique(y_true)) == 1:
        print("Only one class in y_true, skipping ROC curve calculation.")
    else:
        fpr_A, tpr_A, _ = roc_curve(y_true_bin[:, i], scores_A[:, i])
        fpr_B, tpr_B, _ = roc_curve(y_true_bin[:, i], scores_B[:, i])

    roc_auc_A = auc(fpr_A, tpr_A)
    roc_auc_B = auc(fpr_B, tpr_B)

    auc_scores_A.append(roc_auc_A)
    auc_scores_B.append(roc_auc_B)

    plt.plot(fpr_A, tpr_A, label=f"Model A - Class {i} (AUC = {roc_auc_A:.3f})")
    plt.plot(fpr_B, tpr_B, '--', label=f"Model B - Class {i} (AUC = {roc_auc_B:.3f})")

# 计算 Macro-Averaged AUC
macro_auc_A = np.mean(auc_scores_A)
macro_auc_B = np.mean(auc_scores_B)

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (Macro AUC: ModelA={macro_auc_A:.3f}, ModelB={macro_auc_B:.3f})")
plt.legend()
plt.show()

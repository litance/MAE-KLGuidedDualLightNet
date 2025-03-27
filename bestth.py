import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from main import MobileNetLSTMSTAM, ASLDataset, transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelA = MobileNetLSTMSTAM().to(device)
modelA.load_state_dict(torch.load("model.pth", map_location=device))
modelA.eval()

modelB = MobileNetLSTMSTAM().to(device)
modelB.load_state_dict(torch.load("model.pth", map_location=device))
modelB.eval()

test_path = "C:\\Users\\User\\PycharmProjects\\signTest\\dataset\\asl_dataset"
test_dataset = ASLDataset(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(test_dataset.classes)

y_true = []
scores_A = []
scores_B = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs_A = modelA(images)
        outputs_B = modelB(images)

        prob_A = torch.softmax(outputs_A, dim=1)
        prob_B = torch.softmax(outputs_B, dim=1)

        y_true.extend(labels.cpu().numpy())
        scores_A.extend(prob_A.cpu().numpy())
        scores_B.extend(prob_B.cpu().numpy())

y_true = np.array(y_true)
scores_A = np.array(scores_A)
scores_B = np.array(scores_B)

y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

auc_scores_A = []
auc_scores_B = []
best_thresholds_A = []
best_thresholds_B = []

plt.figure(figsize=(10, 6))


def get_best_threshold(fpr, tpr, thresholds):
    J_scores = tpr - fpr
    best_idx = np.argmax(J_scores)
    return thresholds[best_idx]


for i in range(num_classes):
    if len(np.unique(y_true)) == 1:
        print(f"Class {i}: Only one class in y_true, skipping ROC curve calculation.")
        continue

    fpr_A, tpr_A, thresholds_A = roc_curve(y_true_bin[:, i], scores_A[:, i])
    fpr_B, tpr_B, thresholds_B = roc_curve(y_true_bin[:, i], scores_B[:, i])

    roc_auc_A = auc(fpr_A, tpr_A)
    roc_auc_B = auc(fpr_B, tpr_B)

    auc_scores_A.append(roc_auc_A)
    auc_scores_B.append(roc_auc_B)

    best_threshold_A = get_best_threshold(fpr_A, tpr_A, thresholds_A)
    best_threshold_B = get_best_threshold(fpr_B, tpr_B, thresholds_B)

    best_thresholds_A.append(best_threshold_A)
    best_thresholds_B.append(best_threshold_B)

    plt.plot(fpr_A, tpr_A, label=f"Model A - Class {i} (AUC = {roc_auc_A:.3f})")
    plt.plot(fpr_B, tpr_B, '--', label=f"Model B - Class {i} (AUC = {roc_auc_B:.3f})")

macro_auc_A = np.mean(auc_scores_A)
macro_auc_B = np.mean(auc_scores_B)

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (Macro AUC: ModelA={macro_auc_A:.3f}, ModelB={macro_auc_B:.3f})")
plt.legend()
plt.show()

print("Best thresholds a (ModelA):", best_thresholds_A)
print("Best thresholds b (ModelB):", best_thresholds_B)

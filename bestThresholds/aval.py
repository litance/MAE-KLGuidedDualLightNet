import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from dataload import ASLDataset, transform
from modelA import MobileNetLSTMSTAM
from modelB import ESNetLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(model_configs):
    models = {}
    for name, config in model_configs.items():
        model_class, path, num_classes = config

        model = model_class(num_classes).to(device)

        original_state_dict = torch.load(path, map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in original_state_dict.items()
                           if k in model_state_dict and model_state_dict[k].shape == v.shape}

        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)
        model.eval()
        models[name] = model
    return models

test_path = "../dataset/test_dataset/asl_dataset"
test_dataset = ASLDataset(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(test_dataset.classes)

model_configs = {
    "ModelA": (MobileNetLSTMSTAM, "../model/modelA.pth", num_classes),
    "ModelB": (ESNetLSTM, "../model/modelB.pth", num_classes),
}

models = load_models(model_configs)

modelA = models["ModelA"]
modelB = models["ModelB"]


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
    if thresholds.size > 0:
        J_scores = tpr - fpr
        best_idx = np.argmax(J_scores)
        return thresholds[best_idx]
    return 0.5


for i in range(num_classes):
    if np.unique(y_true_bin[:, i]).size < 2:
        print(f"Class {i}: Only one class present in y_true, skipping ROC curve calculation.")
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

best_thresholds_A = [float(x) if np.isfinite(x) else "inf" for x in best_thresholds_A]
best_thresholds_B = [float(x) if np.isfinite(x) else "inf" for x in best_thresholds_B]

data = {
    "Best thresholds a (ModelA)": best_thresholds_A,
    "Best thresholds b (ModelB)": best_thresholds_B
}

json_path = "../bestThresholds/bestThresholds.json"
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Best thresholds saved to {json_path}")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (Macro AUC: ModelA={macro_auc_A:.3f}, ModelB={macro_auc_B:.3f})")
plt.legend()
plt.show()

print("Best thresholds a (ModelA):", best_thresholds_A)
print("Best thresholds b (ModelB):", best_thresholds_B)

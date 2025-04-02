import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from modelA import MobileNetLSTMSTAM
from modelB import LightCNN_LSTM_STAM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cifar10_testset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    testset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    return testset


def load_models(model_configs):
    models = {}
    for name, config in model_configs.items():
        model_creator, path, num_classes = config
        model = model_creator(num_classes).to(device)
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=False)
        model.eval()
        models[name] = model
    return models


def evaluate_models(models, test_loader):
    y_true = []
    results = {name: {'scores': []} for name in models}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            y_true.extend(labels.cpu().numpy())

            for name, model in models.items():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                results[name]['scores'].extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    for name in results:
        results[name]['scores'] = np.array(results[name]['scores'])
    return y_true, results


def calculate_auc(y_true, results, num_classes=10):
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    auc_scores = {name: [] for name in results}
    plt.figure(figsize=(10, 6))

    for name in results:
        for i in range(num_classes):
            if np.unique(y_true_bin[:, i]).size < 2:
                print(f"Class {i} - Only one class present in y_true, skipping.")
                continue

            fpr, tpr, _ = roc_curve(y_true_bin[:, i], results[name]['scores'][:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[name].append(roc_auc)
            plt.plot(fpr, tpr, label=f"{name} - Class {i} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for CIFAR-10 Models")
    plt.legend()
    plt.show()

    return auc_scores


if __name__ == "__main__":
    testset = load_cifar10_testset()
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    model_configs = {
        "ModelC": (MobileNetLSTMSTAM, "../model/modelC.pth", 10),
        "ModelD": (LightCNN_LSTM_STAM, "../model/modelD.pth", 10)
    }

    models = load_models(model_configs)
    y_true, results = evaluate_models(models, test_loader)
    auc_scores = calculate_auc(y_true, results)

    auc_dict = {name: {str(i): float(score) for i, score in enumerate(scores)} for name, scores in auc_scores.items()}
    with open("auc_scores_cifar10.json", "w") as f:
        json.dump(auc_dict, f, indent=4)

    for name in auc_scores:
        print(f"{name} - Macro AUC: {np.mean(auc_scores[name]):.3f}")
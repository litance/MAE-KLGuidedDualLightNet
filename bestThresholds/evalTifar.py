import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from modelC import create_model
from modelD import ESNetLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cifar10_testset():
    """加载CIFAR-10测试集"""
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


# 修改后的 load_models 函数
def load_models(model_configs):
    models = {}
    for name, config in model_configs.items():
        model_creator, path, num_classes = config

        # 初始化模型
        if name == "ModelC":
            model = create_model(num_classes).to(device)
        elif name == "ModelD":
            model = ESNetLSTM(num_classes).to(device)
        else:
            raise ValueError(f"Unknown model name: {name}")

        # 加载权重（兼容不同保存格式）
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 过滤掉不匹配的键
        model_state_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model_state_dict.update(filtered_dict)
        model.load_state_dict(model_state_dict, strict=False)

        model.eval()
        models[name] = model
    return models


def evaluate_models(models, test_loader):
    """评估模型并返回预测结果"""
    results = {name: {'scores': [], 'preds': []} for name in models}
    y_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            y_true.extend(labels.cpu().numpy())

            for name, model in models.items():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                results[name]['scores'].extend(probs.cpu().numpy())
                results[name]['preds'].extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    for name in results:
        results[name]['scores'] = np.array(results[name]['scores'])
        results[name]['preds'] = np.array(results[name]['preds'])

    return y_true, results


def calculate_roc(y_true, results, num_classes=10):
    """计算ROC曲线和AUC"""
    plt.figure(figsize=(10, 8))

    # 二值化真实标签
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    metrics = {}

    for name in results:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresholds = dict()

        # 计算每个类别的ROC
        for i in range(num_classes):
            if np.sum(y_true_bin[:, i]) == 0:
                continue  # 跳过没有样本的类别

            fpr[i], tpr[i], thresholds[i] = roc_curve(y_true_bin[:, i], results[name]['scores'][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            # 绘制ROC曲线
            plt.plot(fpr[i], tpr[i],
                     label=f'{name} - Class {i} (AUC = {roc_auc[i]:.2f})')

        # 计算宏观平均AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        macro_auc = auc(all_fpr, mean_tpr)

        metrics[name] = {
            'class_auc': roc_auc,
            'macro_auc': macro_auc,
            'thresholds': thresholds
        }

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for CIFAR-10 Models')
    plt.legend(loc="lower right")
    plt.show()

    return metrics


if __name__ == "__main__":
    # 加载测试集
    testset = load_cifar10_testset()
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # 模型配置
    model_configs = {
        "ModelC": (create_model, "../model/modelC.pth", 10),  # CIFAR-10有10类
        "ModelD": (ESNetLSTM, "../model/modelD.pth", 10)
    }

    # 加载模型
    models = load_models(model_configs)

    # 评估模型
    y_true, results = evaluate_models(models, test_loader)

    # 计算ROC和AUC
    metrics = calculate_roc(y_true, results)

    # 打印结果
    for name, metric in metrics.items():
        print(f"\n{name} Performance:")
        print(f"Macro AUC: {metric['macro_auc']:.4f}")
        for i in range(10):
            if i in metric['class_auc']:
                print(f"Class {i} AUC: {metric['class_auc'][i]:.4f}")

    # 保存最佳阈值
    best_thresholds = {
        "ModelC": {str(cls): float(thr[0]) for cls, thr in metrics["ModelC"]["thresholds"].items()},
        "ModelD": {str(cls): float(thr[0]) for cls, thr in metrics["ModelD"]["thresholds"].items()}
    }

    with open("best_thresholds_cifar10.json", "w") as f:
        json.dump(best_thresholds, f, indent=4)
    print("\nBest thresholds saved to best_thresholds_cifar10.json")
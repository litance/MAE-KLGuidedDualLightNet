import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from modelA import MobileNetSTAM
from modelB import LightCNN_STAM

# Fix random seed for reproducibility
torch.manual_seed(42)


def load_test_dataset():
    """Load CIFAR-10 test dataset with the same preprocessing as training"""
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


def evaluate_single_model(model, model_name, test_loader, device):
    """Evaluate a single model and return predictions and labels"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def generate_classification_report(labels, preds, class_names, model_name):
    """Generate and print classification report"""
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print("=" * 80)
    print(f"Classification Report for {model_name}:")
    print("=" * 80)
    print(report)

    # Save report to file
    with open(f"classification_report_{model_name}.txt", "w") as f:
        f.write(report)


def plot_confusion_matrix(labels, preds, class_names, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"confusion_matrix_{model_name}.png", bbox_inches="tight")
    plt.close()

    return cm


def load_models(device):
    """Load both models for evaluation"""
    os.makedirs("model", exist_ok=True)

    model_paths = {
        'ModelC': "model/modelA.pth",
        'ModelD': "model/modelB.pth"
    }

    models = {}

    # Load Model C
    print("Loading Model C...")
    try:
        modelC = MobileNetSTAM(num_classes=10)
        checkpoint = torch.load(model_paths['ModelC'], map_location=device)

        # Adjust the model if the state dict does not match
        state_dict = checkpoint['model_state_dict']
        model_dict = modelC.state_dict()

        # Filter out classifier weights to avoid size mismatch
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)

        modelC.load_state_dict(model_dict, strict=False)
        modelC.to(device)
        modelC.eval()
        models['ModelC'] = modelC
        print(f"Model C loaded successfully. Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error loading Model C: {str(e)}")
        raise e

    # Load Model D
    print("Loading Model D...")
    try:
        modelD = LightCNN_STAM(num_classes=10)
        checkpoint = torch.load(model_paths['ModelD'], map_location=device)

        # Adjust the model if the state dict does not match
        state_dict = checkpoint['model_state_dict']
        model_dict = modelD.state_dict()

        # Filter out classifier weights to avoid size mismatch
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)

        modelD.load_state_dict(model_dict, strict=False)
        modelD.to(device)
        modelD.eval()
        models['ModelD'] = modelD
        print(f"Model D loaded successfully. Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error loading Model D: {str(e)}")
        raise e

    return models


def load_auc_scores():
    """Load AUC scores from JSON file"""
    json_path = "bestThresholds/auc_scores_cifar10.json"
    try:
        if not os.path.exists(json_path):
            print(f"Warning: AUC scores file {json_path} not found. Using default AUC scores.")
            return {'ModelC': {str(i): 0.95 for i in range(10)},
                    'ModelD': {str(i): 0.95 for i in range(10)}}

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Validate the loaded data structure
        required_models = ['ModelC', 'ModelD']
        for model_name in required_models:
            if model_name not in data:
                print(f"Warning: {model_name} AUC scores not found in {json_path}. Using defaults.")
                data[model_name] = {str(i): 0.95 for i in range(10)}
            else:
                # Ensure all class indices are present
                for i in range(10):
                    if str(i) not in data[model_name]:
                        print(f"Warning: Class {i} AUC score not found for {model_name}. Using default.")
                        data[model_name][str(i)] = 0.95

        print(f"Successfully loaded AUC scores from {json_path}")

        # Print the Macro AUC values for reference
        if "Macro AUC" in data["ModelC"]:
            print(f"ModelC Macro AUC: {data['ModelC']['Macro AUC']:.4f}")
        if "Macro AUC" in data["ModelD"]:
            print(f"ModelD Macro AUC: {data['ModelD']['Macro AUC']:.4f}")

        return data
    except Exception as e:
        print(f"Error loading AUC scores: {str(e)}. Using default AUC scores.")
        return {'ModelC': {str(i): 0.95 for i in range(10)},
                'ModelD': {str(i): 0.95 for i in range(10)}}


def evaluate_ensemble(probaC, probaD, labels, auc_scores, classes):
    """Evaluate the ensemble model using the same logic as in detect.py"""
    predictions = []
    confidences = []    
    statuses = []

    for i in range(len(labels)):
        # Get the top predictions for each model
        max_probC = np.max(probaC[i])
        predC = np.argmax(probaC[i])

        max_probD = np.max(probaD[i])
        predD = np.argmax(probaD[i])

        # Get AUC scores for the predicted classes
        aucC = float(auc_scores['ModelC'].get(str(predC), 0.95))
        aucD = float(auc_scores['ModelD'].get(str(predD), 0.95))

        # Calculate confidences using AUC scores
        confidenceC = max_probC * aucC
        confidenceD = max_probD * aucD

        # Check if models agree on the prediction
        agree_on_class = (predC == predD)

        # Calculate thresholds based on AUC scores
        thresholdC = 0.5 / aucC if aucC > 0.5 else 0.5
        thresholdD = 0.5 / aucD if aucD > 0.5 else 0.5

        # Check confidence levels
        both_confident = (max_probC > thresholdC) and (max_probD > thresholdD)

        weighted_probC = max_probC * aucC
        weighted_probD = max_probD * aucD

        high_confidence = (weighted_probC > 0.97) and (weighted_probD > 0.97)
        moderate_agreement = (weighted_probC > 0.65 and weighted_probD > 0.35) or (
                weighted_probD > 0.65 and weighted_probC > 0.35)

        combined_confidence = (confidenceC + confidenceD) / 2

        # Determine final prediction based on ensemble rules
        condition = (agree_on_class and both_confident) or (agree_on_class and high_confidence)

        if condition:
            prediction = predC  # Since both models agree
            confidence = combined_confidence
            status = "Confident"
        elif agree_on_class and moderate_agreement:
            prediction = predC  # Both models agree but with moderate confidence
            confidence = combined_confidence
            status = "Moderate"
        else:
            # Choose the model with higher weighted probability
            prediction = predC if weighted_probC >= weighted_probD else predD
            confidence = max(weighted_probC, weighted_probD)
            status = "Uncertain"

        predictions.append(prediction)
        confidences.append(confidence)
        statuses.append(status)

    return np.array(predictions), np.array(confidences), np.array(statuses)


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    test_dataset = load_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    class_names = test_dataset.classes

    # Load models
    models = load_models(device)

    # Load AUC scores
    auc_scores = load_auc_scores()

    # Evaluate Model C
    predsC, probsC, labels = evaluate_single_model(models['ModelC'], "ModelC", test_loader, device)

    # Calculate and print accuracy for Model C
    accuracyC = accuracy_score(labels, predsC)
    print(f"\nTest Accuracy for Model C: {accuracyC * 100:.2f}%")

    # Generate classification report for Model C
    generate_classification_report(labels, predsC, class_names, "ModelC")

    # Plot confusion matrix for Model C
    cmC = plot_confusion_matrix(labels, predsC, class_names, "ModelC")

    # Evaluate Model D
    predsD, probsD, _ = evaluate_single_model(models['ModelD'], "ModelD", test_loader, device)

    # Calculate and print accuracy for Model D
    accuracyD = accuracy_score(labels, predsD)
    print(f"\nTest Accuracy for Model D: {accuracyD * 100:.2f}%")

    # Generate classification report for Model D
    generate_classification_report(labels, predsD, class_names, "ModelD")

    # Plot confusion matrix for Model D
    cmD = plot_confusion_matrix(labels, predsD, class_names, "ModelD")

    # Evaluate ensemble model
    preds_ensemble, confidences, statuses = evaluate_ensemble(probsC, probsD, labels, auc_scores, class_names)

    # Calculate and print accuracy for ensemble
    accuracy_ensemble = accuracy_score(labels, preds_ensemble)
    print(f"\nTest Accuracy for Ensemble: {accuracy_ensemble * 100:.2f}%")

    # Generate classification report for ensemble
    generate_classification_report(labels, preds_ensemble, class_names, "Ensemble")

    # Plot confusion matrix for ensemble
    plot_confusion_matrix(labels, preds_ensemble, class_names, "Ensemble")

    # Create confidence level distribution plot
    plt.figure(figsize=(12, 6))

    # Plot distribution of confidence levels
    sns.histplot(confidences, bins=20, kde=True)
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ensemble Confidence Scores")
    plt.savefig("ensemble_confidence_distribution.png", bbox_inches="tight")
    plt.close()

    # Create status distribution pie chart
    status_counts = pd.Series(statuses).value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90,
            colors=['green', 'orange', 'red'])
    plt.axis('equal')
    plt.title("Distribution of Prediction Confidence Status")
    plt.savefig("ensemble_status_distribution.png", bbox_inches="tight")
    plt.close()

    # Compare models with agreement matrix
    agreement_matrix = np.zeros((len(class_names), len(class_names)))
    for i in range(len(predsC)):
        agreement_matrix[predsC[i]][predsD[i]] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement_matrix, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Model D Predictions")
    plt.ylabel("Model C Predictions")
    plt.title("Model Agreement Matrix")
    plt.savefig("model_agreement_matrix.png", bbox_inches="tight")
    plt.close()

    # Save detailed results to CSV
    results_df = pd.DataFrame({
        "True Label": labels,
        "ModelC Prediction": predsC,
        "ModelC Confidence": np.max(probsC, axis=1),
        "ModelD Prediction": predsD,
        "ModelD Confidence": np.max(probsD, axis=1),
        "Ensemble Prediction": preds_ensemble,
        "Ensemble Confidence": confidences,
        "Status": statuses
    })

    # Convert numeric labels to class names
    results_df["True Label"] = results_df["True Label"].apply(lambda x: class_names[x])
    results_df["ModelC Prediction"] = results_df["ModelC Prediction"].apply(lambda x: class_names[x])
    results_df["ModelD Prediction"] = results_df["ModelD Prediction"].apply(lambda x: class_names[x])
    results_df["Ensemble Prediction"] = results_df["Ensemble Prediction"].apply(lambda x: class_names[x])

    # Save to CSV
    results_df.to_csv("dual_model_evaluation_results.csv", index=False)

    print("\nEvaluation complete! Results saved to files.")


if __name__ == "__main__":
    main()
import json
import torch
from torchvision.transforms import ToPILImage
import cv2
from dataload import transform
from modelA import MobileNetLSTMSTAM
from modelB import ESNetLSTM


def load_models(model_paths, num_classes):
    models = {}
    modelA = MobileNetLSTMSTAM(num_classes)
    checkpoint = torch.load(model_paths['ModelA'], map_location='cpu')
    model_state_dict = modelA.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                       if k in model_state_dict and v.shape == model_state_dict[k].shape}

    if 'fc.weight' in checkpoint['model_state_dict']:
        pretrained_dict['fc.weight'] = checkpoint['model_state_dict']['fc.weight'][:num_classes, :]
    if 'fc.bias' in checkpoint['model_state_dict']:
        pretrained_dict['fc.bias'] = checkpoint['model_state_dict']['fc.bias'][:num_classes]

    model_state_dict.update(pretrained_dict)
    modelA.load_state_dict(model_state_dict, strict=False)
    modelA.eval()
    models['ModelA'] = modelA

    modelB = ESNetLSTM(num_classes)
    checkpoint = torch.load(model_paths['ModelB'], map_location='cpu')

    model_state_dict = modelB.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                       if k in model_state_dict and v.shape == model_state_dict[k].shape}

    if 'fc.weight' in checkpoint['model_state_dict']:
        pretrained_dict['fc.weight'] = checkpoint['model_state_dict']['fc.weight'][:num_classes, :]
    if 'fc.bias' in checkpoint['model_state_dict']:
        pretrained_dict['fc.bias'] = checkpoint['model_state_dict']['fc.bias'][:num_classes]

    model_state_dict.update(pretrained_dict)
    modelB.load_state_dict(model_state_dict, strict=False)
    modelB.eval()
    models['ModelB'] = modelB

    return models


def load_thresholds(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_classes_from_file(file_path):
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(classes)} classes from {file_path}")
    return classes


def detect_sign_language():
    classes = load_classes_from_file('words.txt')
    num_classes = len(classes)

    model_paths = {
        'ModelA': "model/modelA.pth",
        'ModelB': "model/modelB.pth"
    }
    models = load_models(model_paths, num_classes)
    thresholds = load_thresholds("bestThresholds/bestThresholds.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models.values():
        model.to(device)

    cap = cv2.VideoCapture(0)
    to_pil = ToPILImage()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
        roi = frame[100:300, 100:300]
        roi_pil = to_pil(roi)
        roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs_A = models['ModelA'](roi_tensor)
            outputs_B = models['ModelB'](roi_tensor)

            prob_A = torch.softmax(outputs_A, dim=1)
            prob_B = torch.softmax(outputs_B, dim=1)

            max_prob_A, pred_A = torch.max(prob_A, 1)
            max_prob_B, pred_B = torch.max(prob_B, 1)

            pred_A = pred_A.item()
            pred_B = pred_B.item()
            max_prob_A = max_prob_A.item()
            max_prob_B = max_prob_B.item()

            pred_A = min(pred_A, len(thresholds['Best thresholds a (ModelA)']) - 1)
            pred_B = min(pred_B, len(thresholds['Best thresholds b (ModelB)']) - 1)

            threshold_A = thresholds['Best thresholds a (ModelA)'][pred_A]
            threshold_B = thresholds['Best thresholds b (ModelB)'][pred_B]

            threshold_A = float(threshold_A) if threshold_A != "inf" else 0.0
            threshold_B = float(threshold_B) if threshold_B != "inf" else 0.0

            condition_r = (max_prob_A > threshold_A) and (max_prob_B > threshold_B)

            if condition_r and pred_A == pred_B:
                prediction = classes[pred_A]
                confidence = (max_prob_A + max_prob_B) / 2
                color = (0, 255, 0)
            else:
                prediction = "Uncertain"
                confidence = min(max_prob_A, max_prob_B)
                color = (0, 0, 255)

        cv2.putText(frame, f"Detected: {prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"ModelA: {classes[pred_A]} ({max_prob_A:.2f})", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"ModelB: {classes[pred_B]} ({max_prob_B:.2f})", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('ASL Detection (Dual Model Verification)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_sign_language()
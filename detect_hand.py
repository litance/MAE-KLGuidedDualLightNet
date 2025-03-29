import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
import torchvision.models as models
import cv2
import mediapipe as mp
from dataload import transform

class MobileNetLSTMSTAM(nn.Module):
    def __init__(self, num_classes=36):
        super(MobileNetLSTMSTAM, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.mobilenet.features[-1] = nn.Identity()
        self.lstm = nn.LSTM(160, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.mobilenet.features(x)
        x = x.mean([2, 3])
        x = x.view(batch_size, 1, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


def load_model(model_path, num_classes):
    model = MobileNetLSTMSTAM(num_classes)
    original_state_dict = torch.load(model_path)
    model_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in original_state_dict.items()
                       if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


def load_classes_from_file(file_path):
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(classes)} classes from {file_path}")
    return classes


def detect_sign_language():
    classes = load_classes_from_file('words.txt')
    model_path = "model/modelB.pth"
    model = load_model(model_path, num_classes=len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cap = cv2.VideoCapture(0)
    to_pil = ToPILImage()

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])

                width, height = frame.shape[1], frame.shape[0]
                x_min = int(x_min * width)
                x_max = int(x_max * width)
                y_min = int(y_min * height)
                y_max = int(y_max * height)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue

                roi_pil = to_pil(roi)
                roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(roi_tensor)
                    _, predicted = torch.max(outputs, 1)
                    prediction = classes[predicted.item()]

                cv2.putText(frame, f"Detected: {prediction}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('ASL Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_sign_language()

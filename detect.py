import torch
import torch.nn as nn
#import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torchvision.models as models
import cv2
from dataload import transform

class MobileNetLSTMSTAM(nn.Module):
    def __init__(self, num_classes=36):
        super(MobileNetLSTMSTAM, self).__init__()
        self.mobilenet = models.mobilenet_v3(pretrained=True)
        self.mobilenet.features[18] = nn.Identity()
        self.lstm = nn.LSTM(320, 512, batch_first=True)
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

    model_path = "model/modelA.pth"
    model = load_model(model_path, num_classes=len(classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            outputs = model(roi_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = classes[predicted.item()]

        cv2.putText(frame, f"Detected: {prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ASL Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_sign_language()
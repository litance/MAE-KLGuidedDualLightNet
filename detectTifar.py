import json
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from modelC import MobileNetLSTMSTAM
from modelD import LightCNN_LSTM_STAM


def load_models(model_paths, num_classes):
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model C...")
    try:
        # Load ModelC
        modelC = MobileNetLSTMSTAM(num_classes)
        checkpoint = torch.load(model_paths['ModelC'], map_location=device)
        modelC.load_state_dict(checkpoint['model_state_dict'], strict=False)
        modelC.to(device)
        modelC.eval()
        models['ModelC'] = modelC
        print(f"Model C loaded successfully. Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error loading Model C: {str(e)}")
        raise e

    print("Loading Model D...")
    try:
        # Load ModelD - FIXED: was loading Model D weights into Model C
        modelD = LightCNN_LSTM_STAM(num_classes)
        checkpoint = torch.load(model_paths['ModelD'], map_location=device)
        modelD.load_state_dict(checkpoint['model_state_dict'], strict=False)  # FIXED
        modelD.to(device)
        modelD.eval()
        models['ModelD'] = modelD
        print(f"Model D loaded successfully. Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error loading Model D: {str(e)}")
        raise e

    return models


def load_thresholds(json_path):
    try:
        if not os.path.exists(json_path):
            print(f"Warning: Threshold file {json_path} not found. Using default thresholds.")
            # Default threshold values if file not found
            return {'ModelC': {str(i): 0.6 for i in range(10)},
                    'ModelD': {str(i): 0.6 for i in range(10)}}

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Validate thresholds format
        for model_name in ['ModelC', 'ModelD']:
            if model_name not in data:
                print(f"Warning: {model_name} thresholds not found in {json_path}. Using defaults.")
                data[model_name] = {str(i): 0.6 for i in range(10)}

        return data
    except Exception as e:
        print(f"Error loading thresholds: {str(e)}. Using default thresholds.")
        return {'ModelC': {str(i): 0.6 for i in range(10)},
                'ModelD': {str(i): 0.6 for i in range(10)}}


def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


def predict_image(image_path, models, thresholds, transform, classes, debug=False):
    image = Image.open(image_path).convert('RGB')  # Force convert to RGB to handle various image formats
    image_tensor = transform(image).unsqueeze(0)

    device = next(models['ModelC'].parameters()).device  # Get the device from model parameters
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # ModelC prediction
        outputC = models['ModelC'](image_tensor)
        probC = torch.softmax(outputC, dim=1)

        # Get top-3 predictions for ModelC for better analysis
        values_C, indices_C = torch.topk(probC, 3, dim=1)
        max_probC, predC = values_C[0][0].item(), indices_C[0][0].item()

        # ModelD prediction
        outputD = models['ModelD'](image_tensor)
        probD = torch.softmax(outputD, dim=1)

        # Get top-3 predictions for ModelD
        values_D, indices_D = torch.topk(probD, 3, dim=1)
        max_probD, predD = values_D[0][0].item(), indices_D[0][0].item()

        # Get thresholds with fallback values
        thresholdC = float(thresholds['ModelC'].get(str(predC), 0.5))
        thresholdD = float(thresholds['ModelD'].get(str(predD), 0.5))

        # Enhanced decision logic
        agree_on_class = (predC == predD)
        both_confident = (max_probC > thresholdC) and (max_probD > thresholdD)

        # High confidence from one model can compensate for the other
        high_confidence = (max_probC > 0.9 or max_probD > 0.9)
        moderate_agreement = (max_probC > 0.7 and max_probD > 0.4) or (max_probD > 0.7 and max_probC > 0.4)

        condition = (agree_on_class and both_confident) or (agree_on_class and high_confidence)

        if condition:
            prediction = classes[predC]
            confidence = (max_probC + max_probD) / 2
            status = "Confident"
            color = "green"
        elif agree_on_class and moderate_agreement:
            prediction = classes[predC] + " (Moderate)"
            confidence = (max_probC + max_probD) / 2
            status = "Moderately Confident"
            color = "orange"
        else:
            prediction = "Uncertain"
            confidence = min(max_probC, max_probD)
            status = "Uncertain"
            color = "red"

        # Debug information
        if debug:
            print(f"\nDebug information:")
            print(
                f"ModelC top predictions: {[(classes[indices_C[0][i].item()], values_C[0][i].item()) for i in range(3)]}")
            print(
                f"ModelD top predictions: {[(classes[indices_D[0][i].item()], values_D[0][i].item()) for i in range(3)]}")
            print(f"Thresholds - C: {thresholdC}, D: {thresholdD}")
            print(f"Agree on class: {agree_on_class}, Both confident: {both_confident}")
            print(f"High confidence: {high_confidence}, Moderate agreement: {moderate_agreement}")
            print(f"Final decision: {status} - {prediction} ({confidence:.2%})")

    return {
        'prediction': prediction,
        'confidence': confidence,
        'status': status,
        'color': color,
        'modelC_pred': classes[predC],
        'modelC_prob': max_probC,
        'modelD_pred': classes[predD],
        'modelD_prob': max_probD,
        # Additional debug data
        'modelC_top3': [(classes[indices_C[0][i].item()], values_C[0][i].item()) for i in range(3)],
        'modelD_top3': [(classes[indices_D[0][i].item()], values_D[0][i].item()) for i in range(3)]
    }


class CIFAR10ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CIFAR-10 Classifier")
        self.root.geometry("600x700")  # Set default window size

        # Create debug mode checkbox variable
        self.debug_mode = tk.BooleanVar()
        self.debug_mode.set(False)

        # Create directory for saved models if it doesn't exist
        os.makedirs("model", exist_ok=True)
        os.makedirs("bestThresholds", exist_ok=True)

        # Load classes
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        self.num_classes = len(self.classes)

        # Loading status label
        self.status_label = ttk.Label(self.root, text="Loading models...", font=('Helvetica', 12))
        self.status_label.pack(pady=20)
        self.root.update()

        # Set model paths
        self.model_paths = {
            'ModelC': "model/modelC.pth",
            'ModelD': "model/modelD.pth"
        }

        try:
            # Load models and thresholds
            self.models = load_models(self.model_paths, self.num_classes)
            self.thresholds = load_thresholds("bestThresholds/auc_scores_cifar10.json")
            self.transform = get_transform()
            self.status_label.destroy()  # Remove loading label
            self.setup_ui()  # Setup UI after loading models
        except Exception as e:
            self.status_label.config(text=f"Error loading models: {str(e)}", foreground="red")
            self.root.update()
            # Add a retry button
            retry_button = ttk.Button(self.root, text="Retry", command=self.retry_loading)
            retry_button.pack(pady=10)

    def retry_loading(self):
        """Retry loading models if initial load fails"""
        for widget in self.root.winfo_children():
            widget.destroy()

        self.__init__(self.root)

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top frame for image display
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Preview")
        self.image_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(padx=5, pady=5)

        # Frame for buttons
        self.button_frame = ttk.Frame(main_frame)
        self.button_frame.pack(padx=10, pady=5, fill=tk.X)

        self.load_button = ttk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Debug mode checkbox
        self.debug_check = ttk.Checkbutton(self.button_frame, text="Debug Mode",
                                           variable=self.debug_mode)
        self.debug_check.pack(side=tk.RIGHT, padx=5)

        # Frame for results
        self.result_frame = ttk.LabelFrame(main_frame, text="Classification Results")
        self.result_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Prediction label
        self.prediction_label = ttk.Label(self.result_frame, text="Prediction: ", font=('Helvetica', 12))
        self.prediction_text = tk.StringVar()
        self.prediction_display = ttk.Label(self.result_frame, textvariable=self.prediction_text,
                                            font=('Helvetica', 12, 'bold'))
        self.prediction_label.pack(anchor=tk.W, pady=(5, 0))
        self.prediction_display.pack(anchor=tk.W)

        # Confidence label
        self.confidence_label = ttk.Label(self.result_frame, text="Confidence: ", font=('Helvetica', 12))
        self.confidence_text = tk.StringVar()
        self.confidence_display = ttk.Label(self.result_frame, textvariable=self.confidence_text,
                                            font=('Helvetica', 12))
        self.confidence_label.pack(anchor=tk.W, pady=(5, 0))
        self.confidence_display.pack(anchor=tk.W)

        # Status label
        self.status_label = ttk.Label(self.result_frame, text="Status: ", font=('Helvetica', 12))
        self.status_text = tk.StringVar()
        self.status_display = ttk.Label(self.result_frame, textvariable=self.status_text, font=('Helvetica', 12))
        self.status_label.pack(anchor=tk.W, pady=(5, 0))
        self.status_display.pack(anchor=tk.W)

        # Model details frame
        self.model_frame = ttk.LabelFrame(self.result_frame, text="Model Details")
        self.model_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Model C results
        self.modelC_label = ttk.Label(self.model_frame, text="Model C: ")
        self.modelC_text = tk.StringVar()
        self.modelC_display = ttk.Label(self.model_frame, textvariable=self.modelC_text)
        self.modelC_label.pack(anchor=tk.W, pady=(5, 0))
        self.modelC_display.pack(anchor=tk.W)

        # Model C top-3 predictions
        self.modelC_top3_label = ttk.Label(self.model_frame, text="Model C Top-3: ")
        self.modelC_top3_text = tk.StringVar()
        self.modelC_top3_display = ttk.Label(self.model_frame, textvariable=self.modelC_top3_text)
        self.modelC_top3_label.pack(anchor=tk.W, pady=(5, 0))
        self.modelC_top3_display.pack(anchor=tk.W)

        # Model D results
        self.modelD_label = ttk.Label(self.model_frame, text="Model D: ")
        self.modelD_text = tk.StringVar()
        self.modelD_display = ttk.Label(self.model_frame, textvariable=self.modelD_text)
        self.modelD_label.pack(anchor=tk.W, pady=(10, 0))
        self.modelD_display.pack(anchor=tk.W)

        # Model D top-3 predictions
        self.modelD_top3_label = ttk.Label(self.model_frame, text="Model D Top-3: ")
        self.modelD_top3_text = tk.StringVar()
        self.modelD_top3_display = ttk.Label(self.model_frame, textvariable=self.modelD_top3_text)
        self.modelD_top3_label.pack(anchor=tk.W, pady=(5, 0))
        self.modelD_top3_display.pack(anchor=tk.W)

        # Status bar
        self.statusbar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            try:
                self.statusbar.config(text=f"Processing image: {os.path.basename(file_path)}...")
                self.root.update()

                # Display the image
                image = Image.open(file_path)
                image.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo

                # Get prediction with debug flag
                result = predict_image(file_path, self.models, self.thresholds,
                                       self.transform, self.classes, debug=self.debug_mode.get())

                # Update UI with results
                self.prediction_text.set(result['prediction'])
                self.confidence_text.set(f"{result['confidence']:.2%}")
                self.status_text.set(result['status'])

                self.modelC_text.set(
                    f"{result['modelC_pred']} with probability {result['modelC_prob']:.2%}")
                self.modelD_text.set(
                    f"{result['modelD_pred']} with probability {result['modelD_prob']:.2%}")

                # Display top-3 predictions
                top3_c_text = ", ".join([f"{cls} ({prob:.2%})" for cls, prob in result['modelC_top3']])
                top3_d_text = ", ".join([f"{cls} ({prob:.2%})" for cls, prob in result['modelD_top3']])

                self.modelC_top3_text.set(top3_c_text)
                self.modelD_top3_text.set(top3_d_text)

                # Change color based on confidence
                self.prediction_display.config(foreground=result['color'])
                self.status_display.config(foreground=result['color'])

                self.statusbar.config(text="Ready")

            except Exception as e:
                self.statusbar.config(text="Error")
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CIFAR10ClassifierApp(root)
    root.mainloop()
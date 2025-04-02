import json
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from modelA import MobileNetLSTMSTAM
from modelB import LightCNN_LSTM_STAM
from dataset_integration import DatasetManager
from model_trainer import ModelTrainer
import platform


def load_models(model_paths, num_classes):
    # Original load_models function remains unchanged
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model C...")
    try:
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
        modelD = LightCNN_LSTM_STAM(num_classes)
        checkpoint = torch.load(model_paths['ModelD'], map_location=device)
        modelD.load_state_dict(checkpoint['model_state_dict'], strict=False)
        modelD.to(device)
        modelD.eval()
        models['ModelD'] = modelD
        print(f"Model D loaded successfully. Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error loading Model D: {str(e)}")
        raise e

    return models


def load_auc_scores(json_path):
    # Original load_auc_scores function remains unchanged
    try:
        if not os.path.exists(json_path):
            print(f"Warning: AUC scores file {json_path} not found. Using default AUC scores.")
            return {'ModelC': {str(i): 0.95 for i in range(10)},
                    'ModelD': {str(i): 0.95 for i in range(10)}}

        with open(json_path, 'r') as f:
            data = json.load(f)

        for model_name in ['ModelC', 'ModelD']:
            if model_name not in data:
                print(f"Warning: {model_name} AUC scores not found in {json_path}. Using defaults.")
                data[model_name] = {str(i): 0.95 for i in range(10)}

        return data
    except Exception as e:
        print(f"Error loading AUC scores: {str(e)}. Using default AUC scores.")
        return {'ModelC': {str(i): 0.95 for i in range(10)},
                'ModelD': {str(i): 0.95 for i in range(10)}}


def get_transform():
    # Original get_transform function remains unchanged
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


def predict_image(image_path, models, auc_scores, transform, classes, debug=False):
    # Original predict_image function remains unchanged
    image = Image.open(image_path).convert('RGB')  # Force convert to RGB to handle various image formats
    image_tensor = transform(image).unsqueeze(0)

    device = next(models['ModelC'].parameters()).device  # Get the device from model parameters
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputC = models['ModelC'](image_tensor)
        probC = torch.softmax(outputC, dim=1)

        values_C, indices_C = torch.topk(probC, 3, dim=1)
        max_probC, predC = values_C[0][0].item(), indices_C[0][0].item()

        outputD = models['ModelD'](image_tensor)
        probD = torch.softmax(outputD, dim=1)

        values_D, indices_D = torch.topk(probD, 3, dim=1)
        max_probD, predD = values_D[0][0].item(), indices_D[0][0].item()

        aucC = float(auc_scores['ModelC'].get(str(predC), 0.95))
        aucD = float(auc_scores['ModelD'].get(str(predD), 0.95))

        confidenceC = max_probC * aucC
        confidenceD = max_probD * aucD

        agree_on_class = (predC == predD)

        thresholdC = 0.5 / aucC if aucC > 0.5 else 0.5
        thresholdD = 0.5 / aucD if aucD > 0.5 else 0.5

        both_confident = (max_probC > thresholdC) and (max_probD > thresholdD)

        weighted_probC = max_probC * aucC
        weighted_probD = max_probD * aucD

        high_confidence = (weighted_probC > 0.85) or (weighted_probD > 0.85)
        moderate_agreement = (weighted_probC > 0.65 and weighted_probD > 0.35) or (
                weighted_probD > 0.65 and weighted_probC > 0.35)

        combined_confidence = (confidenceC + confidenceD) / 2

        condition = (agree_on_class and both_confident) or (agree_on_class and high_confidence)

        if condition:
            prediction = classes[predC]
            confidence = combined_confidence
            status = "Confident"
            color = "green"
        elif agree_on_class and moderate_agreement:
            prediction = classes[predC] + " (Moderate)"
            confidence = combined_confidence
            status = "Moderately Confident"
            color = "orange"
        else:
            prediction = "Uncertain"
            confidence = min(confidenceC, confidenceD)
            status = "Uncertain"
            color = "red"

        # Debug information
        if debug:
            print(f"\nDebug information:")
            print(
                f"ModelC top predictions: {[(classes[indices_C[0][i].item()], values_C[0][i].item()) for i in range(3)]}")
            print(
                f"ModelD top predictions: {[(classes[indices_D[0][i].item()], values_D[0][i].item()) for i in range(3)]}")
            print(f"AUC scores - C: {aucC}, D: {aucD}")
            print(f"Weighted probabilities - C: {weighted_probC:.4f}, D: {weighted_probD:.4f}")
            print(f"Thresholds - C: {thresholdC:.4f}, D: {thresholdD:.4f}")
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
        'modelC_auc': aucC,
        'modelC_weighted': weighted_probC,
        'modelD_pred': classes[predD],
        'modelD_prob': max_probD,
        'modelD_auc': aucD,
        'modelD_weighted': weighted_probD,
        # Additional debug data
        'modelC_top3': [(classes[indices_C[0][i].item()], values_C[0][i].item()) for i in range(3)],
        'modelD_top3': [(classes[indices_D[0][i].item()], values_D[0][i].item()) for i in range(3)]
    }


class CIFAR10ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CIFAR-10 Classifier")
        self.root.geometry("700x600")  # Adjust initial window size

        # Detect OS for platform-specific behaviors
        self.system = platform.system()

        # Create debug mode checkbox variable
        self.debug_mode = tk.BooleanVar()
        self.debug_mode.set(False)

        # Create directory for saved models if it doesn't exist
        os.makedirs("model", exist_ok=True)
        os.makedirs("bestThresholds", exist_ok=True)
        os.makedirs("datasets/main", exist_ok=True)
        os.makedirs("datasets/temp", exist_ok=True)

        # Default dataset paths
        self.main_dataset_path = "../datasets/main/cifar-10-batches-py"
        self.temp_dataset_path = "datasets/temp"

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
            # Load models and AUC scores
            self.models = load_models(self.model_paths, self.num_classes)
            self.auc_scores = load_auc_scores("bestThresholds/auc_scores_cifar10.json")
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
        # Create a canvas with scrollbar
        self.main_canvas = tk.Canvas(self.root)

        # Add vertical scrollbar
        self.v_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas
        self.main_canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a main frame inside the canvas
        self.scrollable_frame = ttk.Frame(self.main_canvas)

        # Add the frame to the canvas
        self.canvas_frame = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Configure canvas to adjust scroll region when the frame changes size
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )

        # Bind mousewheel to scroll based on platform
        self._bind_mousewheel()

        # Configure canvas to adjust the width of the frame when canvas changes size
        self.main_canvas.bind("<Configure>", self._resize_frame)

        # Main content frame
        main_frame = ttk.Frame(self.scrollable_frame)
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
        self.debug_check = ttk.Checkbutton(self.button_frame, text="Debug Mode", variable=self.debug_mode)
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

        # Model C AUC and weighted probability
        self.modelC_auc_label = ttk.Label(self.model_frame, text="Model C AUC & Weighted Prob: ")
        self.modelC_auc_text = tk.StringVar()
        self.modelC_auc_display = ttk.Label(self.model_frame, textvariable=self.modelC_auc_text)
        self.modelC_auc_label.pack(anchor=tk.W, pady=(2, 0))
        self.modelC_auc_display.pack(anchor=tk.W)

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

        # Model D AUC and weighted probability
        self.modelD_auc_label = ttk.Label(self.model_frame, text="Model D AUC & Weighted Prob: ")
        self.modelD_auc_text = tk.StringVar()
        self.modelD_auc_display = ttk.Label(self.model_frame, textvariable=self.modelD_auc_text)
        self.modelD_auc_label.pack(anchor=tk.W, pady=(2, 0))
        self.modelD_auc_display.pack(anchor=tk.W)

        # Model D top-3 predictions
        self.modelD_top3_label = ttk.Label(self.model_frame, text="Model D Top-3: ")
        self.modelD_top3_text = tk.StringVar()
        self.modelD_top3_display = ttk.Label(self.model_frame, textvariable=self.modelD_top3_text)
        self.modelD_top3_label.pack(anchor=tk.W, pady=(5, 0))
        self.modelD_top3_display.pack(anchor=tk.W)

        # Add dataset management and fine-tuning section
        self.fine_tune_frame = ttk.LabelFrame(main_frame, text="Dataset Integration & Fine-tuning")
        self.fine_tune_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Dataset paths frame
        self.dataset_paths_frame = ttk.Frame(self.fine_tune_frame)
        self.dataset_paths_frame.pack(padx=5, pady=5, fill=tk.X)

        # Main dataset path
        ttk.Label(self.dataset_paths_frame, text="Main Dataset:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.main_dataset_var = tk.StringVar(value=self.main_dataset_path)
        ttk.Entry(self.dataset_paths_frame, textvariable=self.main_dataset_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(self.dataset_paths_frame, text="Browse",
                   command=lambda: self.browse_folder(self.main_dataset_var)).grid(row=0, column=2, padx=5)

        # Temp dataset path
        ttk.Label(self.dataset_paths_frame, text="Temp Dataset:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.temp_dataset_var = tk.StringVar(value=self.temp_dataset_path)
        ttk.Entry(self.dataset_paths_frame, textvariable=self.temp_dataset_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(self.dataset_paths_frame, text="Browse",
                   command=lambda: self.browse_folder(self.temp_dataset_var)).grid(row=1, column=2, padx=5)

        # Fine-tuning settings
        self.fine_tune_settings_frame = ttk.Frame(self.fine_tune_frame)
        self.fine_tune_settings_frame.pack(padx=5, pady=5, fill=tk.X)

        # Learning rate
        ttk.Label(self.fine_tune_settings_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.learning_rate_var = tk.StringVar(value="0.001")
        ttk.Entry(self.fine_tune_settings_frame, textvariable=self.learning_rate_var, width=10).grid(row=0, column=1,
                                                                                                     padx=5,
                                                                                                     sticky=tk.W)

        # Epochs
        ttk.Label(self.fine_tune_settings_frame, text="Epochs:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.epochs_var = tk.IntVar(value=5)
        ttk.Spinbox(self.fine_tune_settings_frame, from_=1, to=20, textvariable=self.epochs_var, width=5).grid(row=0,
                                                                                                               column=3,
                                                                                                               padx=5,
                                                                                                               sticky=tk.W)

        # KL divergence display
        ttk.Label(self.fine_tune_settings_frame, text="KL Divergence:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.kl_div_var = tk.StringVar(value="N/A")
        ttk.Label(self.fine_tune_settings_frame, textvariable=self.kl_div_var).grid(row=1, column=1, sticky=tk.W)

        # Integration proportion display
        ttk.Label(self.fine_tune_settings_frame, text="Integration %:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.integration_prop_var = tk.StringVar(value="N/A")
        ttk.Label(self.fine_tune_settings_frame, textvariable=self.integration_prop_var).grid(row=1, column=3,
                                                                                              sticky=tk.W)

        # Fine-tuning buttons
        self.fine_tune_buttons_frame = ttk.Frame(self.fine_tune_frame)
        self.fine_tune_buttons_frame.pack(padx=5, pady=10, fill=tk.X)

        # Calculate KL button
        self.calc_kl_button = ttk.Button(self.fine_tune_buttons_frame, text="Calculate KL & Proportion",
                                         command=self.calculate_kl_and_proportion)
        self.calc_kl_button.pack(side=tk.LEFT, padx=5)

        # Fine-tune button
        self.fine_tune_button = ttk.Button(self.fine_tune_buttons_frame, text="Fine-tune Models",
                                           command=self.fine_tune_models)
        self.fine_tune_button.pack(side=tk.LEFT, padx=5)

        # Progress bar for fine-tuning
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.fine_tune_frame, orient="horizontal",
                                        length=300, mode="determinate", variable=self.progress_var)
        self.progress.pack(padx=5, pady=5, fill=tk.X)
        self.progress_label = ttk.Label(self.fine_tune_frame, text="")
        self.progress_label.pack(padx=5, anchor=tk.W)

        # Add some padding at the bottom
        ttk.Frame(main_frame).pack(pady=20)

        # Status bar
        self.statusbar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Update UI to show scrollbar if needed
        self.scrollable_frame.update_idletasks()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

        # Initial focus for key bindings to work
        self.root.focus_set()

    def _bind_mousewheel(self):
        """Bind mousewheel events based on platform"""
        if self.system == "Windows":
            # Windows uses MouseWheel event
            self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows)
        elif self.system == "Darwin":
            # macOS uses ScrollWheel event
            self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel_macos)
            self.main_canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
            self.main_canvas.bind_all("<Button-5>", self._on_mousewheel_linux)
        else:
            # Linux uses Button-4 and Button-5
            self.main_canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
            self.main_canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

        # Add key bindings for scrolling
        self.root.bind("<Up>", lambda event: self.main_canvas.yview_scroll(-1, "units"))
        self.root.bind("<Down>", lambda event: self.main_canvas.yview_scroll(1, "units"))
        self.root.bind("<Prior>", lambda event: self.main_canvas.yview_scroll(-1, "pages"))  # Page Up
        self.root.bind("<Next>", lambda event: self.main_canvas.yview_scroll(1, "pages"))  # Page Down
        self.root.bind("<Home>", lambda event: self.main_canvas.yview_moveto(0))
        self.root.bind("<End>", lambda event: self.main_canvas.yview_moveto(1))

    def _on_mousewheel_windows(self, event):
        """Handle mousewheel scrolling for Windows"""
        self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_macos(self, event):
        """Handle mousewheel scrolling for macOS"""
        self.main_canvas.yview_scroll(int(-1 * event.delta), "units")

    def _on_mousewheel_linux(self, event):
        """Handle mousewheel scrolling for Linux"""
        if event.num == 4:
            self.main_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.main_canvas.yview_scroll(1, "units")

    def _resize_frame(self, event):
        """Resize the frame width when the canvas changes size"""
        canvas_width = event.width
        self.main_canvas.itemconfig(self.canvas_frame, width=canvas_width)

    def browse_folder(self, stringvar):
        """Browse for a folder and update the StringVar"""
        folder_path = filedialog.askdirectory(title="Select Dataset Directory")
        if folder_path:
            stringvar.set(folder_path)

    def calculate_kl_and_proportion(self):
        """Calculate KL divergence and integration proportion"""
        self.statusbar.config(text="Calculating KL divergence and integration proportion...")
        self.root.update()

        try:
            # Get dataset paths from UI
            main_path = self.main_dataset_var.get()
            temp_path = self.temp_dataset_var.get()

            # Create dataset manager
            dataset_manager = DatasetManager(main_path, temp_path)

            # Calculate KL divergence
            kl_div = dataset_manager.calculate_kl_divergence()
            self.kl_div_var.set(f"{kl_div:.4f}")

            # Calculate integration proportion
            prop = dataset_manager.calculate_integration_proportion()
            self.integration_prop_var.set(f"{prop:.2%}")

            self.statusbar.config(text="KL divergence and integration proportion calculated")
        except Exception as e:
            self.statusbar.config(text="Error calculating KL divergence")
            messagebox.showerror("Error", f"Failed to calculate KL divergence: {str(e)}")

    def fine_tune_models(self):
        """Fine-tune models with integrated dataset"""
        self.statusbar.config(text="Preparing for fine-tuning...")
        self.root.update()

        try:
            # Get dataset paths from UI
            main_path = self.main_dataset_var.get()
            temp_path = self.temp_dataset_var.get()

            # Get fine-tuning settings
            learning_rate = float(self.learning_rate_var.get())
            epochs = self.epochs_var.get()

            # Create dataset manager and get dataloaders
            dataset_manager = DatasetManager(main_path, temp_path, batch_size=64)
            dataloaders = dataset_manager.get_dataloaders()

            # Calculate and display KL divergence and integration proportion
            kl_div = dataset_manager.calculate_kl_divergence()
            self.kl_div_var.set(f"{kl_div:.4f}")

            prop = dataset_manager.calculate_integration_proportion()
            self.integration_prop_var.set(f"{prop:.2%}")

            # Reset progress bar
            self.progress_var.set(0)
            self.progress_label.config(text="Preparing models for fine-tuning...")

            # Create model trainer
            device = next(self.models['ModelC'].parameters()).device
            trainer = ModelTrainer(self.models, dataloaders, device, learning_rate, epochs)

            # Start fine-tuning in a separate thread
            import threading
            thread = threading.Thread(target=self._fine_tune_thread, args=(trainer,))
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.statusbar.config(text="Error starting fine-tuning")
            messagebox.showerror("Error", f"Failed to start fine-tuning: {str(e)}")

    def _fine_tune_thread(self, trainer):
        """Run fine-tuning in a separate thread"""
        try:
            # Define a callback function to update progress
            def progress_callback(model_name, epoch, total_epochs, accuracy):
                progress = ((epoch / total_epochs) * 100) / len(self.models)
                if model_name == 'ModelC':
                    base_progress = 0
                else:
                    base_progress = 50
                progress = base_progress + progress
                self.progress_var.set(progress)
                self.progress_label.config(
                    text=f"Fine-tuning {model_name}: Epoch {epoch}/{total_epochs} - Accuracy: {accuracy:.2%}")
                self.root.update_idletasks()

            # Fine-tune models
            results = trainer.fine_tune(callback=progress_callback)

            # Update UI on the main thread
            self.root.after(0, self._fine_tuning_complete, results)

        except Exception as e:
            # Handle exceptions and update UI on the main thread
            self.root.after(0, self._fine_tuning_error, str(e))

    def _fine_tuning_complete(self, results):
        """Called when fine-tuning completes successfully"""
        self.progress_var.set(100)
        self.progress_label.config(text="Fine-tuning completed successfully")

        # Update model paths to use fine-tuned models
        self.model_paths = {
            'ModelC': "model/ModelC_fine_tuned.pth",
            'ModelD': "model/ModelD_fine_tuned.pth"
        }

        # Reload models
        try:
            self.models = load_models(self.model_paths, self.num_classes)
            self.statusbar.config(text="Fine-tuning complete. Models updated successfully.")

            # Show results
            result_msg = "Fine-tuning Results:\n"
            for model_name, accuracy in results.items():
                result_msg += f"{model_name}: {accuracy:.4f} validation accuracy\n"

            messagebox.showinfo("Fine-tuning Complete", result_msg)

        except Exception as e:
            messagebox.showerror("Error", f"Fine-tuning completed but failed to load updated models: {str(e)}")
            self.statusbar.config(text="Error loading fine-tuned models")

    def _fine_tuning_error(self, error_msg):
        """Called when fine-tuning encounters an error"""
        self.progress_label.config(text=f"Error during fine-tuning: {error_msg}")
        self.statusbar.config(text="Fine-tuning failed")
        messagebox.showerror("Fine-tuning Error", f"An error occurred during fine-tuning: {error_msg}")

    def load_image(self):
        # Original load_image function remains unchanged
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
                result = predict_image(file_path, self.models, self.auc_scores,
                                       self.transform, self.classes, debug=self.debug_mode.get())

                # Update UI with results
                self.prediction_text.set(result['prediction'])
                self.confidence_text.set(f"{result['confidence']:.2%}")
                self.status_text.set(result['status'])

                self.modelC_text.set(
                    f"{result['modelC_pred']} with probability {result['modelC_prob']:.2%}")
                self.modelD_text.set(
                    f"{result['modelD_pred']} with probability {result['modelD_prob']:.2%}")

                # Display AUC scores and weighted probabilities
                self.modelC_auc_text.set(
                    f"AUC: {result['modelC_auc']:.4f}, Weighted Prob: {result['modelC_weighted']:.4f}")
                self.modelD_auc_text.set(
                    f"AUC: {result['modelD_auc']:.4f}, Weighted Prob: {result['modelD_weighted']:.4f}")

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
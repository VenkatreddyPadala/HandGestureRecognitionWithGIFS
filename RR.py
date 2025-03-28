import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from googletrans import Translator

# Update path to your GIFS folder
GESTURE_FOLDER = "GIFS"

def get_all_gifs(base_folder):
    """
    Get all GIF files from both the base folder and its subfolders.
    For files in the root folder, use filename as category.
    For files in subfolders, use subfolder name as category.
    Returns a list of tuples (file_path, category).
    """
    gif_files = []
    
    # First, get files from the root folder
    for file in os.listdir(base_folder):
        if file.endswith('.gif'):
            full_path = os.path.join(base_folder, file)
            # If file is directly in root folder, use filename without extension as category
            if os.path.dirname(full_path) == base_folder:
                category = os.path.splitext(file)[0]  # Use filename without .gif
                gif_files.append((full_path, category))
    
    # Then get files from subfolders
    for root, dirs, files in os.walk(base_folder):
        if root == base_folder:  # Skip root folder as we already processed it
            continue
        category = os.path.basename(root)
        for file in files:
            if file.endswith('.gif'):
                full_path = os.path.join(root, file)
                gif_files.append((full_path, category))
    
    return gif_files

class SignLanguageRecognizer:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_features(self, frame):
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.holistic.process(frame_rgb)
        
        features = []
        
        # Extract hand landmarks if detected
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0] * 63)  # Padding for left hand
            
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0] * 63)  # Padding for right hand
            
        # Extract face landmarks for context (reduced set)
        if results.face_landmarks:
            important_face_landmarks = [0, 4, 8, 12, 14, 17, 57, 130, 287, 359]  # Key facial points
            for idx in important_face_landmarks:
                landmark = results.face_landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0] * 30)  # Padding for face
            
        return np.array(features) if len(features) > 0 else None

    def __del__(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.root.geometry("1000x800")
        
        self.recognizer = SignLanguageRecognizer()
        self.classifier = None
        self.label_mapping = None
        self.categories = None
        
        # Try to load existing model
        self.load_model()
        self.create_ui()

    def load_model(self):
        """Attempt to load an existing trained model."""
        model_path = 'sign_language_model.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.classifier = data['classifier']
                    self.label_mapping = data['label_mapping']
                    self.categories = data['categories']
                return True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
        return False
        
    def create_ui(self):
        # Configure styles
        style = ttk.Style()
        style.configure('Bold.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
        
        # Main container
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Sign Language Recognition System",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 20))
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            status_frame, 
            textvariable=self.status_var,
            style='Bold.TLabel'
        ).pack()
        
        # Training section
        train_frame = ttk.LabelFrame(main_frame, text="Model Training", padding=10)
        train_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add model status indicator
        self.model_status_var = tk.StringVar(
            value="Model Status: " + 
            ("Loaded" if self.classifier is not None else "Not Loaded")
        )
        ttk.Label(
            train_frame,
            textvariable=self.model_status_var,
            style='Info.TLabel'
        ).pack(pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            train_frame,
            variable=self.progress_var,
            maximum=100,
            length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(train_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Train button
        train_button = ttk.Button(
            button_frame,
            text="Train New Model",
            command=self.train_model,
            style='Bold.TButton'
        )
        train_button.pack(side=tk.LEFT, padx=5)
        
        # Load model button
        load_button = ttk.Button(
            button_frame,
            text="Load Existing Model",
            command=self.load_existing_model,
            style='Bold.TButton'
        )
        load_button.pack(side=tk.LEFT, padx=5)
        
        # Recognition section
        recog_frame = ttk.LabelFrame(main_frame, text="Sign Recognition", padding=10)
        recog_frame.pack(fill=tk.BOTH, expand=True)
        
        # Upload button
        upload_button = ttk.Button(
            recog_frame,
            text="Upload Sign GIF",
            command=self.upload_gif,
            style='Bold.TButton'
        )
        upload_button.pack(pady=5)
        
        # Display frame
        self.display_frame = ttk.Frame(recog_frame)
        self.display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Image display
        self.image_label = ttk.Label(self.display_frame)
        self.image_label.pack(pady=10)
        
        # Results frame
        results_frame = ttk.Frame(self.display_frame)
        results_frame.pack(fill=tk.X, pady=10)
        
        # Results variables
        self.gesture_var = tk.StringVar()
        self.confidence_var = tk.StringVar()
        self.telugu_var = tk.StringVar()
        
        # Results labels
        ttk.Label(
            results_frame,
            textvariable=self.gesture_var,
            style='Bold.TLabel'
        ).pack(pady=2)
        
        ttk.Label(
            results_frame,
            textvariable=self.confidence_var,
            style='Info.TLabel'
        ).pack(pady=2)
        
        ttk.Label(
            results_frame,
            textvariable=self.telugu_var,
            style='Bold.TLabel'
        ).pack(pady=2)

    def update_progress(self, value, message=""):
        self.progress_var.set(value)
        if message:
            self.status_var.set(message)
        self.root.update()

    def load_existing_model(self):
        """Load an existing model from a file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Pickle files", "*.pkl")]
        )
        
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.classifier = data['classifier']
                    self.label_mapping = data['label_mapping']
                    self.categories = data['categories']
                
                self.model_status_var.set("Model Status: Loaded")
                messagebox.showinfo(
                    "Success",
                    f"Model loaded successfully!\nCategories: {len(self.categories)}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def train_model(self):
        if self.classifier is not None:
            response = messagebox.askyesno(
                "Confirm Training",
                "A model is already loaded. Do you want to train a new one?"
            )
            if not response:
                return

        try:
            self.update_progress(0, "Loading training data...")
            
            # Get all GIF files recursively
            gif_files = get_all_gifs(GESTURE_FOLDER)
            total_files = len(gif_files)
            
            if total_files == 0:
                raise Exception("No GIF files found in the specified folder")
            
            # Prepare training data
            X = []  # Features
            y = []  # Labels
            categories = set()  # Keep track of unique categories
            
            for idx, (gif_path, category) in enumerate(gif_files):
                categories.add(category)
                
                # Extract frames from GIF
                gif = Image.open(gif_path)
                frames = []
                try:
                    for frame_idx in range(gif.n_frames):
                        gif.seek(frame_idx)
                        frame = cv2.cvtColor(np.array(gif.convert('RGB')), cv2.COLOR_RGB2BGR)
                        frames.append(frame)
                except Exception as e:
                    print(f"Error processing frames from {gif_path}: {str(e)}")
                    continue
                
                # Process each frame
                for frame in frames:
                    features = self.recognizer.extract_features(frame)
                    if features is not None:
                        X.append(features)
                        y.append(category)
                
                progress = (idx + 1) / total_files * 100
                self.update_progress(
                    progress,
                    f"Processing {os.path.basename(gif_path)}..."
                )
            
            if len(X) == 0:
                raise Exception("No valid features extracted from the training data")
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Create label mapping
            unique_labels = sorted(list(categories))
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            y_encoded = np.array([self.label_mapping[label] for label in y])
            
            # Train the classifier
            self.update_progress(90, "Training classifier...")
            self.classifier = RandomForestClassifier(n_estimators=100, max_depth=20)
            self.classifier.fit(X, y_encoded)
            self.categories = list(categories)
            
            # Save the model and category information
            save_path = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl")],
                initialfile="sign_language_model.pkl"
            )
            
            if save_path:
                self.update_progress(95, "Saving model...")
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'classifier': self.classifier,
                        'label_mapping': self.label_mapping,
                        'categories': list(categories)
                    }, f)
            
            self.update_progress(100, "Training completed successfully!")
            self.model_status_var.set("Model Status: Loaded")
            
            messagebox.showinfo(
                "Success",
                f"Model trained on {len(categories)} different categories!"
            )
            
        except Exception as e:
            self.update_progress(0, "Training failed!")
            self.model_status_var.set("Model Status: Not Loaded")
            messagebox.showerror("Error", str(e))

    def predict_sign(self, gif_path, actual_category=None):
        try:
            # Process GIF frames
            gif = Image.open(gif_path)
            predictions = []
            
            for frame_idx in range(gif.n_frames):
                gif.seek(frame_idx)
                frame = cv2.cvtColor(np.array(gif.convert('RGB')), cv2.COLOR_RGB2BGR)
                features = self.recognizer.extract_features(frame)
                
                if features is not None:
                    pred = self.classifier.predict_proba([features])[0]
                    predictions.append(pred)
            
            if not predictions:
                raise Exception("No valid signs detected in the GIF")
            
            # Average predictions across frames
            avg_pred = np.mean(predictions, axis=0)
            predicted_idx = np.argmax(avg_pred)
            confidence = avg_pred[predicted_idx] * 100
            
            # Get predicted label
            reverse_mapping = {v: k for k, v in self.label_mapping.items()}
            predicted_category = reverse_mapping[predicted_idx]
            
            # Update display with only the prediction
            self.gesture_var.set(f"Predicted Sign: {predicted_category}")
            self.confidence_var.set(f"Accuracy: {confidence:.1f}%")
            
            # Translate to Telugu
            try:
                translator = Translator()
                telugu_text = translator.translate(
                    predicted_category.replace('_', ' ').title(),
                    dest='te'
                ).text
                self.telugu_var.set(f"Telugu: {telugu_text}")
            except Exception as e:
                self.telugu_var.set("Telugu translation failed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")

    def upload_gif(self):
        if self.classifier is None:
            messagebox.showwarning(
                "Warning",
                "Please load or train a model first!"
            )
            return
        
        file_path = filedialog.askopenfilename(
            initialdir=GESTURE_FOLDER,
            filetypes=[("GIF files", "*.gif")]
        )
        
        if file_path:
            try:
                # Display GIF
                gif = Image.open(file_path)
                gif.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(gif)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                
                # If file is in root folder, use filename as category, otherwise use folder name
                if os.path.dirname(file_path) == GESTURE_FOLDER:
                    category = os.path.splitext(os.path.basename(file_path))[0]
                else:
                    category = os.path.basename(os.path.dirname(file_path))
                self.predict_sign(file_path, category)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading GIF: {str(e)}")

    def __del__(self):
        """Cleanup resources when the application is closed."""
        if hasattr(self, 'recognizer'):
            del self.recognizer

def main():
    # Create the root window
    root = tk.Tk()
    
    try:
        # Set window icon if available
        root.iconbitmap('icon.ico')
    except:
        pass  # Icon file not found, continue without it
    
    # Create and run the application
    app = SignLanguageApp(root)
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()
    
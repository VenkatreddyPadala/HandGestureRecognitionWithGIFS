import numpy as np
import cv2
import os
from tqdm import tqdm

# Initialize the lists to hold image data and labels
data = []
labels = []

# Define the dataset path and class labels
dataset_path = 'extracted_frames'  # Replace with the actual path to your extracted frames directory
class_labels = os.listdir(dataset_path)  # Assuming each sub-folder is a class

# Loop through each class folder
for label in class_labels:
    class_folder = os.path.join(dataset_path, label)
    
    # Loop through all subfolders and extract frames
    for subfolder in os.listdir(class_folder):
        subfolder_path = os.path.join(class_folder, subfolder)
        
        # Loop through all image files in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.png'):  # Check if the file is an image
                img_path = os.path.join(subfolder_path, filename)
                
                # Read the image
                img = cv2.imread(img_path)
                
                # Check if the image was successfully read
                if img is not None:
                    img_resized = cv2.resize(img, (64, 64))  # Resize image to (64, 64)
                    data.append(img_resized)  # Append the image data
                    labels.append(label)  # Append the class label

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Now you can save the data and labels
np.save('images_data.npy', data)  # Saving image data
np.save('labels_data.npy', labels)  # Saving corresponding labels

print("Data and labels saved successfully!")

# Optionally, you can print the shapes of the data and labels
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

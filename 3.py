# Load the saved data
import numpy as np
data = np.load('images_data.npy')
labels = np.load('labels_data.npy')

# Print the shapes of the loaded data
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Normalize the images
data = data.astype('float32') / 255.0

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Print the shape of the train and test sets
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

import tensorflow as tf
to_categorical = tf.keras.utils.to_categorical
from sklearn.preprocessing import LabelEncoder

# Encode the labels as integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

# Print the shape of the one-hot encoded labels
print("One-hot encoded labels shape (train):", y_train_one_hot.shape)
print("One-hot encoded labels shape (test):", y_test_one_hot.shape)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(27, activation='softmax')  # Assuming 27 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(X_test, y_test_one_hot))

# Save the trained model
# Save the trained model in the native Keras format
model.save('trained_model.keras')


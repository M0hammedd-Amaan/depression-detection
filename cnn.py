import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define paths
data_path = r"D:\iiit\ds"

# Function to extract features from audio files
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.mean(zcr)
    return np.hstack([mfcc_mean, chroma_mean, mel_mean, zcr_mean])

# Data augmentation function
def augment_audio(y, sr):
    # Pitch shift
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
    # Time stretch
    y_stretched = librosa.effects.time_stretch(y=y, rate=0.8)
    # Add noise
    noise = np.random.randn(len(y))
    y_noisy = y + 0.005 * noise
    # Change speed
    y_speed = librosa.effects.time_stretch(y=y, rate=1.2)
    # Change volume
    y_louder = y * 1.5
    return y_shifted, y_stretched, y_noisy, y_speed, y_louder

# Load and process dataset
data = []
labels = []
for file_name in os.listdir(data_path):
    if file_name.endswith('.wav'):
        file_path = os.path.join(data_path, file_name)
        category = int(file_name.split('_')[-1].replace('.wav', '').replace('category', ''))
        
        # Original audio
        y, sr = librosa.load(file_path, sr=None)
        features = extract_features(y, sr)
        data.append(features)
        labels.append(category)

        # Augmented audios
        y_shifted, y_stretched, y_noisy, y_speed, y_louder = augment_audio(y, sr)
        
        # Fix length for consistency
        y_shifted = librosa.util.fix_length(y_shifted, size=len(y))
        y_stretched = librosa.util.fix_length(y_stretched, size=len(y))
        y_noisy = librosa.util.fix_length(y_noisy, size=len(y))
        y_speed = librosa.util.fix_length(y_speed, size=len(y))
        y_louder = librosa.util.fix_length(y_louder, size=len(y))
        
        # Extract features
        shifted_features = extract_features(y_shifted, sr)
        stretched_features = extract_features(y_stretched, sr)
        noisy_features = extract_features(y_noisy, sr)
        speed_features = extract_features(y_speed, sr)
        louder_features = extract_features(y_louder, sr)
        
        # Append features
        data.append(shifted_features)
        labels.append(category)
        data.append(stretched_features)
        labels.append(category)
        data.append(noisy_features)
        labels.append(category)
        data.append(speed_features)
        labels.append(category)
        data.append(louder_features)
        labels.append(category)

data = np.array(data)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize data
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0)

# Reshape data for Conv1D input
X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Build the model
input_shape = (X_train.shape[1], 1)
model = Sequential([
    Input(shape=input_shape),
    Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(256, 3, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(512, 3, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(5, activation='softmax')  # 5 classes for categories 0-4
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)
]

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks)

# Save the model
model.save('depression_detection_model.keras')

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Print final accuracy
final_accuracy = history.history['val_accuracy'][-1]
print(f"Final validation accuracy: {final_accuracy:.4f}")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
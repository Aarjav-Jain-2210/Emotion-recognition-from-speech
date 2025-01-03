import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
# Define the data folder path
data_folder = r"C:\Users\Administrator\Downloads"
# Define the load_data function
def load_data(data_folder):
    audio_data = []
    labels = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Extract emotion label from the filename
                emotion = file.split("-")[2]
                labels.append(emotion)
                audio_data.append(file_path)
    return audio_data, labels
# Load the audio data and labels
audio_data, labels = load_data(data_folder)
# Load the audio data and labels
audio_data, labels = load_data(data_folder)
# Check the length of the audio_data and labels lists
print(f'Number of audio files: {len(audio_data)}')
print(f'Number of labels: {len(labels)}')
# Check the first few audio files and labels
print(f'First 5 audio files: {audio_data[:5]}')
print(f'First 5 labels: {labels[:5]}')
X_train, X_test, y_train, y_test = train_test_split(audio_data, labels, test_size=0.2, random_state=42)
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)
X_train = np.array([extract_features(x) for x in X_train])
X_test = np.array([extract_features(x) for x in X_test])
y_train_encoded = to_categorical(LabelEncoder().fit_transform(y_train))
y_test_encoded = to_categorical(LabelEncoder().fit_transform(y_test))
model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(Dense(units=y_train_encoded.shape[1], activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, validation_split=0.2)
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}') 
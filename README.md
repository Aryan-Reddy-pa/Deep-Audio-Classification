Deep Audio Classification

This project is a simple end-to-end audio classification pipeline built using Python, Librosa, and TensorFlow. It loads audio files, extracts spectrogram-based features, trains a CNN model, and predicts the genre or class of an input audio sample.
The notebook walks through the entire workflow so anyone can understand how audio classification works from scratch.

Project Workflow

1. Audio Loading
The project uses Librosa to load audio files such as WAV or MP3.
Each file is resampled consistently (44100 Hz) to ensure the model receives uniform input.
y, sr = librosa.load("blues.00000.wav", sr=44100)
2. Feature Extraction
Each audio signal is converted to a Mel Spectrogram.
The spectrogram is resized so it can be fed into a CNN like an image.
Steps:
Compute Mel spectrogram
Convert to log scale
Resize to a fixed resolution (128×128)
This transforms raw audio into a structured 2D format that the model can understand.

4. Dataset Preparation
The notebook builds a dataset using folders of audio files.
Each folder represents a class.
Example structure:
dataset/
 ├── blues/
 ├── classical/
 ├── rock/
 └── jazz/
The code loops through each folder, extracts features, and stores:
X: spectrogram images
y: class labels
Labels are one-hot encoded using to_categorical.
5. Train–Test Split
The dataset is split into training and testing sets using:

train_test_split(data, labels, test_size=0.2, random_state=42)

6. Model Architecture
   
A Convolutional Neural Network (CNN) is used for classification.
Layers include:
Conv2D + MaxPool2D blocks
Flatten
Dense
Dropout
Output layer with softmax
Optimizer: Adam
Loss: Categorical Crossentropy
Metrics: Accuracy

8. Training
The model is trained for multiple epochs while tracking accuracy and loss.
The project includes:
Training history
Visualizations (accuracy and loss curves)
training_hist.json is saved so results can be loaded later.
9. Prediction
   
After training, the model can classify any new audio file.
The audio undergoes the same preprocessing steps before being passed to the model.
Dependencies
Install required packages:
pip install librosa tensorflow matplotlib seaborn numpy scikit-learn


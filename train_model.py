"""Train the Camp Doodle CNN on the prepared QuickDraw dataset.

Loads data/features.npy and data/labels.npy, splits them into train,
validation and test sets, trains a small CNN with early stopping and
saves the best model to models/Camp.keras.
"""
import os

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
from keras.models import Sequential
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from config import categories

SEED = 50
MODEL_PATH = os.path.join("models", "Camp.keras")


def load_data():
    """Load the dataset and return (features, labels) ready for the CNN."""
    features = np.load(os.path.join("data", "features.npy"))
    labels = np.load(os.path.join("data", "labels.npy"))
    # Scale pixels from 0-255 to 0-1; float32 uses half the memory of float64
    features = features.astype("float32") / 255.0
    # Keras expects (samples, height, width, channels); 1 channel = grayscale
    features = features.reshape(-1,28,28,1)
    return features, labels


def build_model():
    """Create the CNN: two convolution blocks followed by a dense classifier."""
    return Sequential([ Input(shape=(28, 28, 1)),
    # 32 filters of 3x3 slide over the image and look for small patterns;
    # ReLU sets negative values to 0 and keeps positive ones
    Conv2D(32,(3, 3), activation="relu", name="eye1"),
    # Keep only the strongest value of every 2x2 block (shrinks the image)
    MaxPool2D(pool_size=(2, 2), name="filter1"),
    # More filters on the second block to find more complex shapes
    Conv2D(64, (3, 3), activation="relu", name="eye2"),
    MaxPool2D(pool_size=(2, 2), name="filter2"),
    # Turn the feature maps into one long vector for the dense layers
    Flatten(name="flat"),
    Dense(128, activation="relu", name="brain"),
    # Randomly switch off 50% of the neurons during training to
    # reduce overfitting
    Dropout(0.5, name="no_cheat"),
    # One output per category; softmax turns the scores into probabilities
    Dense(len(categories), activation="softmax", name="output")])


def main():
    """Split the data, train the model and evaluate it on the test set."""
    features, labels = load_data()
    # Two-step split: 15% test, then 20% of the rest for validation.
    # stratify keeps the categories equally represented in every set, and the
    # fixed random_state gives the same split on every run.
    features_temp, features_test, labels_temp, labels_test = train_test_split(
        features, labels, test_size=0.15, random_state=SEED, stratify=labels)
    features_train, features_val, labels_train, labels_val = train_test_split(
        features_temp, labels_temp, test_size=0.20, random_state=SEED, stratify=labels_temp)
    print("Data is ready")
    # Fixed seed so that the weight initialization is reproducible
    keras.utils.set_random_seed(SEED)
    model = build_model()
    model.summary()
    # "sparse" because the labels are integers (0-19), not one-hot vectors;
    # the loss punishes confident wrong predictions the most
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Stop when val_loss has not improved for 3 epochs and go back to the
    # best weights instead of keeping the last ones
    stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(features_train, labels_train, epochs=30,
            batch_size=64, validation_data=(features_val, labels_val), callbacks=[stop])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    # The test set is used only once, here, so the result is an honest estimate
    print("\n Final Exam Results")
    _, accuracy = model.evaluate(features_test, labels_test)
    print(f"Real Accuracy: {accuracy*100:.2f}%\n")

    predictions = np.argmax(model.predict(features_test), axis=1)
    print("the model gets confused:")
    # Rows are the true categories, columns are the predicted ones
    print(confusion_matrix(labels_test, predictions))


if __name__ == "__main__":
    main()

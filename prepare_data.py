"""Build the training dataset from the downloaded QuickDraw files.

Takes the first SIZE drawings of every category in config.py and saves
them to data/features.npy (pixels) and data/labels.npy (category index).
"""
import os
import sys
import numpy as np

from config import categories

# Number of drawings taken from each category
SIZE = 10000
FEATURES_PATH = os.path.join("data", "features.npy")
LABELS_PATH = os.path.join("data", "labels.npy")
# Skip if the dataset already exists. Delete both files to rebuild it,
# for example after changing SIZE or the order of the categories.
if os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH):
    print("data already exist!")
    sys.exit()
else:
    print("selecting samples...")
    drawings = []
    labels = []
    for index, category in enumerate(categories):
        filepath = os.path.join("data", f"{category}.npy")
        data = np.load(filepath)
        # Fail early instead of silently building an unbalanced dataset
        if len(data) < SIZE:
            raise ValueError(f"{category} has only {len(data)} samples")
        sample = data[:SIZE]
        drawings.append(sample)
        # The label of a drawing is the index of its category in config.py
        array = np.full(len(sample), index)
        labels.append(array)
        print(f"added {len(sample)} samples for {category}")
    features = np.concatenate(drawings)
    labels = np.concatenate(labels)
    # Pixels stay as 0-255 integers to save space; Train_Model.py normalizes them
    np.save(FEATURES_PATH, features)
    np.save(LABELS_PATH, labels)
    print(f"dataset has been created fully with total samples: {len(features)}")

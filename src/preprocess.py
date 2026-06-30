"""
preprocess.py

Purpose:
--------
1. Read all Healthy vibration files
2. Read all Broken vibration files
3. Extract features
4. Create Machine Learning Dataset
5. Save dataset as CSV
"""

import os
import pandas as pd

from src.config import (
    HEALTHY_FOLDER,
    BROKEN_FOLDER,
    CSV_PATH
)

from src.feature_extraction import extract_features


def process_healthy_data():

    healthy_features = []

    for file in os.listdir(HEALTHY_FOLDER):

        if file.lower().endswith((".txt", ".csv")):

            file_path = os.path.join(HEALTHY_FOLDER, file)

            features = extract_features(file_path)

            features["File"] = file
            features["Label"] = "Healthy"

            healthy_features.append(features)

    healthy_df = pd.DataFrame(healthy_features)

    return healthy_df


def process_broken_data():

    broken_features = []

    for file in os.listdir(BROKEN_FOLDER):

        if file.lower().endswith((".txt", ".csv")):

            file_path = os.path.join(BROKEN_FOLDER, file)

            features = extract_features(file_path)

            features["File"] = file
            features["Label"] = "Broken"

            broken_features.append(features)

    broken_df = pd.DataFrame(broken_features)

    return broken_df


def create_dataset():

    print("=" * 60)
    print("Creating Gearbox Dataset...")
    print("=" * 60)

    healthy_df = process_healthy_data()

    broken_df = process_broken_data()

    final_df = pd.concat(
        [healthy_df, broken_df],
        ignore_index=True
    )

    final_df.to_csv(CSV_PATH, index=False)

    print("\nDataset Created Successfully\n")

    print(final_df.head())

    print("\nDataset Shape :", final_df.shape)

    print("\nClass Distribution\n")

    print(final_df["Label"].value_counts())

    return final_df


if __name__ == "__main__":

    create_dataset()
"""
feature_extraction.py
---------------------
Extract statistical features from gearbox vibration signal.
"""
import os
import pandas as pd
import numpy as np


def extract_features(file_path):
    """
    Read vibration signal (.txt) and extract features.

    Parameters:
        file_path (str): Path of vibration text file

    Returns:
        dict: Extracted features
    """

    # Load vibration signal
    extension = os.path.splitext(file_path)[1].lower()

    # ---------------- TXT ----------------
    if extension == ".txt":

        signal = np.loadtxt(file_path)

    # ---------------- CSV ----------------
    elif extension == ".csv":

        df = pd.read_csv(file_path)

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.empty:
            raise ValueError(
                "CSV file must contain at least one numeric column."
            )

        # Use first numeric column as vibration signal
        signal = numeric_df.iloc[:, 0].to_numpy()

    # ---------------- Unsupported ----------------
    else:

        raise ValueError(
            "Only .txt and .csv files are supported."
        )

    # Statistical Features
    mean = np.mean(signal)
    std = np.std(signal)
    minimum = np.min(signal)
    maximum = np.max(signal)
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    crest_factor = peak / rms if rms != 0 else 0

    features = {
        "Mean": mean,
        "Std": std,
        "Minimum": minimum,
        "Maximum": maximum,
        "Peak": peak,
        "RMS": rms,
        "CrestFactor": crest_factor
    }

    return features


# ----------------------------
# Test Feature Extraction
# ----------------------------
if __name__ == "__main__":

    file_path = "../Dataset/Healthy Data/h30hz10.txt"

    try:
        features = extract_features(file_path)

        print("Extracted Features:\n")

        for key, value in features.items():
            print(f"{key:15}: {value:.5f}")

    except Exception as e:
        print("File:", file_path)
        print("Error:", e)
        
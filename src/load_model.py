"""
load_model.py

Purpose:
--------
1. Load Saved Machine Learning Model
2. Return the trained model
"""

import os
import joblib

from src.config import MODEL_PATH


def load_model():
    """
    Load the trained model from disk.

    Returns
    -------
    model : Trained Machine Learning Model
    """

    # Check whether model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found!\n{MODEL_PATH}"
        )

    # Load Model
    model = joblib.load(MODEL_PATH)

    #print("=" * 60)
    #print("MODEL LOADED SUCCESSFULLY")
    #print("=" * 60)
    #print(f"Model Path : {MODEL_PATH}")

    return model


if __name__ == "__main__":

    model = load_model()

    print("\nLoaded Model:")
    print(model)
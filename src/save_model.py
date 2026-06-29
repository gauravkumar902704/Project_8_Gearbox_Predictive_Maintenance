"""
save_model.py

Purpose:
--------
1. Save trained Machine Learning model
2. Create Models folder automatically (if not exists)
3. Store model as .pkl file
"""

import os
import joblib

from src.config import MODEL_PATH


def save_model(model):
    """
    Save trained model to disk.

    Parameters
    ----------
    model : Trained Machine Learning Model
    """

    # Create Models folder if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save Model
    joblib.dump(model, MODEL_PATH)

    print("=" * 60)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model Location : {MODEL_PATH}")


if __name__ == "__main__":

    print("This file is used for saving trained models.")
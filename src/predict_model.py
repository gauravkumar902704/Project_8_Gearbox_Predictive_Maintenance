"""
predict_model.py

Purpose:
--------
1. Load trained model
2. Read new vibration signal (.txt)
3. Extract features
4. Predict Gearbox Health
5. Display confidence score
"""

import os
import pandas as pd

from src.feature_extraction import extract_features
from src.load_model import load_model


def predict(file_path):
    """
    Predict Gearbox Health
    """

    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nFile Not Found!\n{file_path}"
        )

    # Load trained model
    model = load_model()

    # Extract Features
    features = extract_features(file_path)

    input_df = pd.DataFrame([features])

    # Prediction
    prediction = model.predict(input_df)

    # Confidence Score
    probability = model.predict_proba(input_df)

    confidence = max(probability[0]) * 100

    print("\n" + "="*60)
    print("GEARBOX HEALTH PREDICTION")
    print("="*60)

    print(f"\nInput File : {os.path.basename(file_path)}")

    if prediction[0] == 0:
        print("\nPrediction : Broken Tooth")
    else:
        print("\nPrediction : Healthy")

    print(f"\nConfidence : {confidence:.2f}%")

    print("\nExtracted Features\n")
    print(input_df)

    return prediction[0], confidence


if __name__ == "__main__":

    test_file = input("Enter vibration file path : ")

    predict(test_file)
"""
main.py

Author  : Gaurav Kumar
Project : Predictive Maintenance of Gearbox using Vibration Sensors

Description:
------------
Main entry point of the project.

Pipeline:
1. Create Feature Dataset
2. Train Machine Learning Models
3. Evaluate Models
4. Save Best Model
5. Predict Gearbox Health
"""

from src.preprocess import create_dataset
from src.train_model import train_pipeline
from src.evaluate_model import evaluation_pipeline
from src.save_model import save_model
from src.predict_model import predict


def main():

    print("=" * 70)
    print("GEARBOX PREDICTIVE MAINTENANCE USING MACHINE LEARNING")
    print("=" * 70)

    # ==========================================================
    # Step 1 : Create Dataset
    # ==========================================================

    print("\nSTEP 1 : Creating Dataset...\n")

    create_dataset()

    # ==========================================================
    # Step 2 : Train Models
    # ==========================================================

    print("\nSTEP 2 : Training Models...\n")

    (
        best_model,
        best_model_name,
        models,
        results,
        X_test,
        y_test,
        label_encoder
    ) = train_pipeline()

    # ==========================================================
    # Step 3 : Evaluate Models
    # ==========================================================

    print("\nSTEP 3 : Evaluating Models...\n")

    evaluation_pipeline(
        models,
        X_test,
        y_test
    )

    # ==========================================================
    # Step 4 : Save Best Model
    # ==========================================================

    print("\nSTEP 4 : Saving Best Model...\n")

    save_model(best_model)

    print(f"\nBest Model Saved : {best_model_name}")

    # ==========================================================
    # Step 5 : Prediction
    # ==========================================================

    print("\nSTEP 5 : Prediction System\n")

    file_path = input(
        "Enter vibration (.txt) file path : "
    )

    predict(file_path)

    print("\nProject Completed Successfully")


# ==========================================================
# Main Function
# ==========================================================

if __name__ == "__main__":
    main()
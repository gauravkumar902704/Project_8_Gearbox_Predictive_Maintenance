"""
evaluate_model.py

Purpose:
--------
1. Evaluate all trained models
2. Generate Accuracy
3. Generate Confusion Matrix
4. Generate Classification Report
5. Compare all models
6. Plot Accuracy Graph
"""

import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models
    """

    results = []

    for name, model in models.items():

        prediction = model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)

        print("\n" + "="*60)
        print(name)
        print("="*60)

        print(f"\nAccuracy : {accuracy*100:.2f}%")

        print("\nConfusion Matrix\n")
        print(confusion_matrix(y_test, prediction))
        cm = confusion_matrix(y_test, prediction)

        image_folder = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Images"
        )

        os.makedirs(image_folder, exist_ok=True)

        plt.figure(figsize=(5,4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Broken", "Healthy"],
            yticklabels=["Broken", "Healthy"]
        )

        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        save_path = os.path.join(
            image_folder,
            f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        

        print("\nClassification Report\n")
        print(classification_report(y_test, prediction))

        results.append({
            "Model": name,
            "Accuracy": accuracy
        })

    result_df = pd.DataFrame(results)

    return result_df


def show_best_model(result_df):
    """
    Display Best Model
    """

    best_model = result_df.loc[
        result_df["Accuracy"].idxmax()
    ]

    print("\n" + "="*60)
    print("BEST MODEL")
    print("="*60)

    print(best_model)

    return best_model


def plot_accuracy(result_df):
    """
    Plot and save model accuracy graph.
    """

    import os

    image_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "Images"
    )

    os.makedirs(image_folder, exist_ok=True)

    plt.figure(figsize=(7,3.5))

    plt.bar(
        result_df["Model"],
        result_df["Accuracy"]
    )

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.grid(axis="y")

    save_path = os.path.join(
        image_folder,
        "accuracy_comparison.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nAccuracy graph saved at:\n{save_path}")


def evaluation_pipeline(models, X_test, y_test):
    """
    Complete Evaluation Pipeline
    """

    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)

    result_df = evaluate_models(
        models,
        X_test,
        y_test
    )

    print("\nAccuracy Comparison Table\n")

    print(result_df)

    best_model = show_best_model(result_df)

    plot_accuracy(result_df)
    feature_names = X_test.columns

    plot_feature_importance(
        best_model,
        feature_names
    )

    return result_df, best_model


def plot_feature_importance(best_model, feature_names):
    """
    Plot Feature Importance of Decision Tree or Random Forest
    """

    import os

    if not hasattr(best_model, "feature_importances_"):
        print("\nFeature Importance not available for this model.")
        return

    image_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "Images"
    )

    os.makedirs(image_folder, exist_ok=True)

    importance = best_model.feature_importances_

    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10,5))

    plt.bar(
        range(len(feature_names)),
        importance[indices]
    )

    plt.xticks(
        range(len(feature_names)),
        np.array(feature_names)[indices],
        rotation=30
    )

    plt.title("Feature Importance")

    plt.xlabel("Features")

    plt.ylabel("Importance")

    plt.grid(axis="y")

    save_path = os.path.join(
        image_folder,
        "feature_importance.png"
    )

    plt.savefig(save_path,
                dpi=300,
                bbox_inches="tight")

    plt.show()

    print(f"\nFeature Importance saved at:\n{save_path}")
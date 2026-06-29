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

import pandas as pd
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
    Plot Model Accuracy
    """

    plt.figure(figsize=(10,5))

    plt.bar(
        result_df["Model"],
        result_df["Accuracy"]
    )

    plt.title("Model Comparison")

    plt.xlabel("Machine Learning Models")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=20)

    plt.grid(axis="y")

    plt.show()


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

    return result_df, best_model
"""
train_model.py

Purpose:
--------
1. Load processed dataset
2. Encode labels
3. Split dataset
4. Train Machine Learning models
5. Return trained models
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from src.config import CSV_PATH


def load_dataset():
    """
    Load processed CSV dataset
    """
    df = pd.read_csv(CSV_PATH)
    return df


def prepare_data(df):
    """
    Encode labels and split features/target
    """

    le = LabelEncoder()

    df["Label"] = le.fit_transform(df["Label"])

    X = df.drop(columns=["Label", "File"])
    y = df["Label"]

    return X, y, le


def split_data(X, y):
    """
    Train-Test Split
    """

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def train_models(X_train, y_train):
    """
    Train all Machine Learning Models
    """

    models = {

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier(n_neighbors=3),

        "Logistic Regression":
            LogisticRegression(random_state=42),

        "Support Vector Machine":
            SVC(kernel="linear",
                random_state=42),

        "Gaussian Naive Bayes":
            GaussianNB()

    }

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} Trained Successfully")

    return trained_models


def evaluate_accuracy(models, X_test, y_test):
    """
    Calculate Accuracy of Every Model
    """

    result = {}

    for name, model in models.items():

        prediction = model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)

        result[name] = accuracy

    return result


def train_pipeline():
    """
    Complete Training Pipeline
    """

    print("=" * 60)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 60)

    df = load_dataset()

    X, y, le = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = train_models(X_train, y_train)

    results = evaluate_accuracy(
        models,
        X_test,
        y_test
    )

    print("\nModel Accuracy\n")

    for name, score in results.items():

        print(f"{name:<30} : {score*100:.2f}%")

    best_model_name = max(
        results,
        key=results.get
    )

    best_model = models[best_model_name]

    print("\nBest Model :", best_model_name)

    return (
        best_model,
        best_model_name,
        models,
        results,
        X_test,
        y_test,
        le
    )


if __name__ == "__main__":

    train_pipeline()
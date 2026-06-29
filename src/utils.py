"""
utils.py

Purpose:
--------
Utility functions used throughout the project.
"""

import matplotlib.pyplot as plt


# ==========================================================
# Print Section Heading
# ==========================================================

def print_heading(title):
    """
    Print a formatted heading.
    """

    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


# ==========================================================
# Display Dataset Information
# ==========================================================

def dataset_info(df):
    """
    Display dataset information.
    """

    print_heading("Dataset Information")

    print(f"Shape : {df.shape}")

    print("\nColumns")
    print(df.columns)

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nUnique Values")
    print(df.nunique())


# ==========================================================
# Plot Vibration Signal
# ==========================================================

def plot_signal(signal, title):
    """
    Plot vibration signal.
    """

    plt.figure(figsize=(15,5))

    plt.plot(signal)

    plt.title(title)

    plt.xlabel("Sample Number")

    plt.ylabel("Amplitude")

    plt.grid()

    plt.show()


# ==========================================================
# Histogram
# ==========================================================

def plot_histogram(signal, title):
    """
    Plot Histogram.
    """

    plt.figure(figsize=(8,5))

    plt.hist(signal, bins=50)

    plt.title(title)

    plt.xlabel("Amplitude")

    plt.ylabel("Frequency")

    plt.grid()

    plt.show()


# ==========================================================
# Box Plot
# ==========================================================

def plot_box(signal, title):
    """
    Plot Box Plot.
    """

    plt.figure(figsize=(6,6))

    plt.boxplot(signal)

    plt.title(title)

    plt.grid()

    plt.show()
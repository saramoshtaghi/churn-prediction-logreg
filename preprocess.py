import os
import kaggle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Install and configure Kaggle API
def setup_kaggle():
    os.system("pip install kaggle")
    os.makedirs("~/.kaggle", exist_ok=True)
    os.system("cp /content/kaggle.json ~/.kaggle/")
    os.system("chmod 600 ~/.kaggle/kaggle.json")

# Download dataset from Kaggle
def download_dataset():
    os.system("kaggle datasets download -d royjafari/customer-churn -p /content/sample_data --unzip")

# Load dataset
def load_dataset():
    csv_file = "/content/sample_data/Customer Churn.csv"  # Adjust if filename differs
    df = pd.read_csv(csv_file)
    print("\nSample Data:")
    print(df.head())
    return df

# Remove ZIP file to save space
def cleanup_zip():
    zip_path = "/content/sample_data/customer-churn.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("ZIP file deleted to save space.")

# Display dataset structure
def display_dataset_info(df):
    print("\nDataset Information:")
    print(df.info())
    print("\nFirst Few Rows:")
    print(df.head())

# Visualize class distribution
def plot_class_distribution(df):
    class_counts = df["Churn"].value_counts()
    print("\nClass Distribution:")
    print(class_counts)
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="coolwarm")
    plt.xticks([0, 1], ["Not Churn (0)", "Churn (1)"])
    plt.xlabel("Churn Label")
    plt.ylabel("Count")
    plt.title("Churn Class Distribution")
    plt.show()

# Generate correlation heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# Check missing values
def check_missing_values(df):
    print("\nMissing Values per Column:")
    print(df.isnull().sum())

# Main function to execute preprocessing steps
def main():
    setup_kaggle()
    download_dataset()
    df = load_dataset()
    cleanup_zip()
    display_dataset_info(df)
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    check_missing_values(df)

if __name__ == "__main__":
    main()

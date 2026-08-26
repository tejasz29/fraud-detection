import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "data"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}")
        print("Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Place creditcard.csv in the data/ folder")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()

    print("Class distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print(f"Fraud rate: {class_counts[1] / len(df) * 100:.4f}%")
    print()

    print("Basic statistics:")
    print(df.describe())
    print()

    print("Missing values:")
    print(df.isnull().sum().sum())
    print()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Class', palette=['#2ecc71', '#e74c3c'])
    plt.title('Class Distribution (0=Normal, 1=Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=150)
    plt.close()
    print("Saved: data/class_distribution.png")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    normal_amounts = df[df['Class'] == 0]['Amount']
    fraud_amounts = df[df['Class'] == 1]['Amount']
    plt.hist(normal_amounts, bins=50, alpha=0.6, label='Normal', color='#2ecc71', density=True)
    plt.hist(fraud_amounts, bins=50, alpha=0.6, label='Fraud', color='#e74c3c', density=True)
    plt.title('Amount Distribution (Density)')
    plt.xlabel('Amount')
    plt.ylabel('Density')
    plt.legend()
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.boxplot([normal_amounts, fraud_amounts], labels=['Normal', 'Fraud'])
    plt.title('Amount Distribution (Boxplot)')
    plt.ylabel('Amount')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'amount_distribution.png'), dpi=150)
    plt.close()
    print("Saved: data/amount_distribution.png")

    plt.figure(figsize=(14, 12))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, square=True, linewidths=0.1, cbar_kws={"shrink": 0.5})
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print("Saved: data/correlation_heatmap.png")

    print("\nFraud amount statistics:")
    print(fraud_amounts.describe())
    print("\nNormal amount statistics:")
    print(normal_amounts.describe())

    print("\nTime feature statistics:")
    print(df['Time'].describe())
    print(f"Time range: {df['Time'].min()} to {df['Time'].max()} seconds")
    print(f"Time span: {(df['Time'].max() - df['Time'].min()) / 3600:.1f} hours")

if __name__ == "__main__":
    main()
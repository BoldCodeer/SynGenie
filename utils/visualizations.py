import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    """
    Generate summary plots for a dataset.

    Args:
        df (pd.DataFrame): DataFrame to visualize.
    """
    if df.empty:
        print("No data to visualize.")
        return

    print("Generating data visualizations...")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Histograms for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols].hist(bins=20, figsize=(15, 10))
    plt.suptitle("Distribution of Numeric Columns")
    plt.tight_layout()
    plt.show()

    # Pair plot (limited to first 4 numeric columns)
    if len(numeric_cols) > 1:
        sns.pairplot(df[numeric_cols[:4]])
        plt.suptitle("Pairplot of First 4 Numeric Features", y=1.02)
        plt.show()

    print("Visualization complete.")

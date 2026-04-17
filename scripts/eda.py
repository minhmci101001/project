import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set visual style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="pastel")
# Set font
try:
    plt.rcParams['font.family'] = 'Segoe UI'
except:
    pass

def load_data(filepath):
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def feature_engineering_eda(df):
    """Calculate additional features for visualization."""
    df_plot = df.copy()

    # Title length
    df_plot['title_length'] = df_plot['title'].astype(str).apply(len)

    # Calculate Engagement Rate
    # Add 1 to the denominator to avoid division by zero
    df_plot['engagement_rate'] = (df_plot['like_count'] + df_plot['comment_count']) / (df_plot['view_count'] + 1)

    # Label text
    df_plot['Trending Status'] = df_plot['is_trending'].map({1: 'Trending', 0: 'Normal'})

    # Parse publish time and convert to Vietnam timezone (UTC+7)
    if 'published_at' in df_plot.columns:
        df_plot['published_at'] = pd.to_datetime(df_plot['published_at'])
        df_plot['publish_hour'] = (df_plot['published_at'].dt.hour + 7) % 24
    else:
        df_plot['publish_hour'] = 12  # Fallback default

    return df_plot


def run_eda(df_plot, output_dir):
    print("\n Generating exploratory data analysis charts...")
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Class Distribution
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df_plot, x='Trending Status', palette=['#3498db', '#e74c3c'])
    plt.title("Dataset Class Distribution (Trending vs Normal)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Number of Videos", fontsize=12)
    plt.xlabel("")
    for i in ax.containers:
        ax.bar_label(i, padding=3, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_distribution.png"), dpi=300)
    plt.close()

    # Chart 2: View Count Distribution (Log Scale)
    plt.figure(figsize=(10, 6))
    df_plot['log_views'] = np.log1p(df_plot['view_count'])
    sns.kdeplot(data=df_plot, x='log_views', hue='Trending Status', fill=True,
                palette=['#3498db', '#e74c3c'], common_norm=False, alpha=0.5)
    plt.title("View Count Distribution (Log Scale) by Trending Status", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Log1p(View Count)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_view_distribution.png"), dpi=300)
    plt.close()

    # Chart 3: Correlation Matrix of Engagement Metrics
    plt.figure(figsize=(10, 8))
    numeric_cols = ['is_trending', 'view_count', 'like_count', 'comment_count',
                    'duration_seconds', 'title_length', 'engagement_rate']
    corr = df_plot[numeric_cols].corr()
    # Mask upper triangle to reduce clutter
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=1, cbar_kws={"shrink": .8})
    plt.title("Correlation Heatmap of Engagement Metrics", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_correlation_matrix.png"), dpi=300)
    plt.close()

    # Chart 4: Does Title Length Affect Trending?
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_plot, x='Trending Status', y='title_length',
                palette=['#3498db', '#e74c3c'], showfliers=False)
    plt.title("Title Length vs Trending Likelihood", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Title Length (characters)", fontsize=12)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_title_length_boxplot.png"), dpi=300)
    plt.close()

    # Chart 5: Engagement Rate Comparison
    plt.figure(figsize=(8, 6))
    # Exclude extreme outliers for cleaner visualization
    sns.boxplot(data=df_plot, x='Trending Status', y='engagement_rate',
                palette=['#3498db', '#e74c3c'], showfliers=False)
    plt.title("Engagement Rate (Likes+Comments / Views) by Group", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Engagement Rate", fontsize=12)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_engagement_rate_boxplot.png"), dpi=300)
    plt.close()

    # Chart 6: Content Category Distribution
    plt.figure(figsize=(12, 6))
    category_map = {
        10: 'Music', 20: 'Gaming', 22: 'People/Blogs', 23: 'Comedy',
        24: 'Entertainment', 25: 'News', 27: 'Education', 28: 'Tech'
    }
    df_plot['Category Name'] = df_plot['category_id'].map(category_map).fillna('Other')
    ax2 = sns.countplot(data=df_plot, x='Category Name', hue='Trending Status',
                        palette=['#3498db', '#e74c3c'])
    plt.title("Content Category Distribution by Trending Status", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Number of Videos", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_category_distribution.png"), dpi=300)
    plt.close()

    # Chart 7: Publish Hour Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_plot, x='publish_hour', hue='Trending Status', multiple="dodge",
                 bins=24, palette=['#3498db', '#e74c3c'], shrink=0.8)
    plt.title("Impact of Publish Hour on Trending Likelihood", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Number of Videos", fontsize=12)
    plt.xlabel("Hour of Day (0–23, Vietnam Time UTC+7)", fontsize=12)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_publish_hour_distribution.png"), dpi=300)
    plt.close()

    print(f"Successfully generated 7 EDA charts. Saved to: {output_dir}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'youtube_data.csv')
    output_dir = os.path.join(os.path.dirname(current_dir), 'plots')

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at: {data_path}")
        print("Please run crawl_data.py first!")
    else:
        df = load_data(data_path)
        df_engineered = feature_engineering_eda(df)
        run_eda(df_engineered, output_dir)

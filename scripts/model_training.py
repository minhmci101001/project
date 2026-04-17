import os
import pandas as pd
import numpy as np
import emoji
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import joblib

# ── Visual style for all model evaluation plots ──────────────────────────────
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="pastel")
try:
    plt.rcParams['font.family'] = 'Segoe UI'
except:
    pass

PALETTE = {'RF': '#e74c3c', 'Baseline': '#3498db'}


# ── Helper text-feature functions ────────────────────────────────────────────

def contains_emoji(text):
    return 1 if emoji.emoji_count(str(text)) > 0 else 0

def is_question(text):
    return 1 if '?' in str(text) else 0

def is_exclamation(text):
    return 1 if '!' in str(text) else 0

def has_number_in_text(text):
    import re
    return 1 if bool(re.search(r'\d', str(text))) else 0


# ── Feature Engineering ──────────────────────────────────────────────────────

def feature_engineering(df):
    """Process raw data and create model-ready features."""
    df_model = df.copy()

    # NOTE: Engagement rate intentionally excluded — it causes data leakage
    # (likes/comments/views are only available after publication).

    # 1. Title text features
    df_model['title_length']   = df_model['title'].astype(str).str.len()
    df_model['has_emoji']      = df_model['title'].apply(contains_emoji)
    df_model['is_question']    = df_model['title'].apply(is_question)
    df_model['is_exclamation'] = df_model['title'].apply(is_exclamation)
    df_model['has_number']     = df_model['title'].apply(has_number_in_text)

    # 2. Tags count
    df_model['tags_count'] = df_model['tags'].apply(
        lambda x: len(str(x).split('|')) if pd.notnull(x) else 0
    )

    # 3. Time features
    df_model['published_at'] = pd.to_datetime(df_model['published_at'])
    df_model['publish_hour'] = df_model['published_at'].dt.hour
    df_model['day_of_week']  = df_model['published_at'].dt.weekday  # 0=Mon, 6=Sun

    # 4. Channel & category features
    df_model['category_id'] = (
        pd.to_numeric(df_model['category_id'], errors='coerce').fillna(24).astype(int)
    )
    if 'subscriber_count' not in df_model.columns:
        df_model['subscriber_count'] = 0
    df_model['subscriber_count'] = (
        pd.to_numeric(df_model['subscriber_count'], errors='coerce').fillna(0).astype(int)
    )
    # Log-transform to reduce skew from large channels
    df_model['log_subscriber_count'] = np.log1p(df_model['subscriber_count'])

    # Only use pre-publication (deterministic) features
    features = [
        'duration_seconds',
        'title_length', 'has_emoji', 'is_question', 'is_exclamation', 'has_number',
        'tags_count', 'category_id', 'log_subscriber_count',
        'publish_hour', 'day_of_week'
    ]

    X = df_model[features].fillna(0)
    y = df_model['is_trending']

    return X, y, features


# ── Model Training & Evaluation ──────────────────────────────────────────────

def train_and_evaluate(X, y, plots_dir):
    """Train both models, print metrics, save all evaluation plots, return RF model."""
    os.makedirs(plots_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Baseline: Logistic Regression ────────────────────────────────────────
    print("\n" + "="*55)
    print("  BASELINE MODEL — LOGISTIC REGRESSION")
    print("="*55)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    baseline = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    baseline.fit(X_train_scaled, y_train)

    y_pred_base = baseline.predict(X_test_scaled)
    y_prob_base = baseline.predict_proba(X_test_scaled)[:, 1]

    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base  = f1_score(y_test, y_pred_base)
    auc_base = roc_auc_score(y_test, y_prob_base)

    print(f"  Accuracy : {acc_base:.4f}")
    print(f"  F1-Score : {f1_base:.4f}")
    print(f"  ROC-AUC  : {auc_base:.4f}")
    print("\nClassification Report (Baseline):\n",
          classification_report(y_test, y_pred_base))

    # ── Proposed: Random Forest ───────────────────────────────────────────────
    print("\n" + "="*55)
    print("  PROPOSED MODEL — RANDOM FOREST")
    print("="*55)

    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=12,
        random_state=42, class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf  = rf_model.predict_proba(X_test)[:, 1]

    acc_rf  = accuracy_score(y_test, y_pred_rf)
    f1_rf   = f1_score(y_test, y_pred_rf)
    auc_rf  = roc_auc_score(y_test, y_prob_rf)

    print(f"  Accuracy : {acc_rf:.4f}")
    print(f"  F1-Score : {f1_rf:.4f}")
    print(f"  ROC-AUC  : {auc_rf:.4f}")
    print("\nClassification Report (Random Forest):\n",
          classification_report(y_test, y_pred_rf))

    # Feature importance table
    feat_imp = pd.DataFrame({
        'Feature':    X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importances (Random Forest):")
    print(feat_imp.to_string(index=False))

    # ── Generate & save evaluation plots ─────────────────────────────────────
    print("\nGenerating model evaluation plots...")

    _plot_confusion_matrix(y_test, y_pred_rf, plots_dir)
    _plot_roc_curve(y_test, y_prob_base, y_prob_rf, auc_base, auc_rf, plots_dir)
    _plot_feature_importance(feat_imp, plots_dir)
    _plot_model_comparison(acc_base, f1_base, auc_base, acc_rf, f1_rf, auc_rf, plots_dir)

    print(f"  All evaluation plots saved to: {plots_dir}")

    return rf_model, scaler


# ── Individual plot functions ─────────────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred_rf, output_dir):
    """Chart 08 — Confusion Matrix heatmap for Random Forest."""
    cm = confusion_matrix(y_test, y_pred_rf)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Reds',
        xticklabels=['Normal', 'Trending'],
        yticklabels=['Normal', 'Trending'],
        linewidths=1, ax=ax
    )
    ax.set_title("Confusion Matrix — Random Forest", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    # Annotate TN / FP / FN / TP
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, labels[i][j],
                    ha='center', va='center', fontsize=9, color='white', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_confusion_matrix.png"), dpi=300)
    plt.close()
    print("  [1/4] Confusion matrix saved.")


def _plot_roc_curve(y_test, y_prob_base, y_prob_rf, auc_base, auc_rf, output_dir):
    """Chart 09 — ROC Curve comparison between Baseline and Random Forest."""
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_rf,   tpr_rf,   _ = roc_curve(y_test, y_prob_rf)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_base, tpr_base, color=PALETTE['Baseline'], lw=2,
            label=f"Logistic Regression (AUC = {auc_base:.3f})")
    ax.plot(fpr_rf,   tpr_rf,   color=PALETTE['RF'],       lw=2,
            label=f"Random Forest      (AUC = {auc_rf:.3f})")
    ax.plot([0, 1], [0, 1], 'w--', lw=1, label="Random Classifier (AUC = 0.500)")

    ax.set_title("ROC Curve — Model Comparison", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_roc_curve.png"), dpi=300)
    plt.close()
    print("  [2/4] ROC curve saved.")


def _plot_feature_importance(feat_imp, output_dir):
    """Chart 10 — Horizontal bar chart of Random Forest feature importances."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = sns.color_palette("Reds_r", len(feat_imp))
    bars = ax.barh(feat_imp['Feature'][::-1], feat_imp['Importance'][::-1],
                   color=colors, edgecolor='none')

    # Value labels
    for bar, val in zip(bars, feat_imp['Importance'][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=10)

    ax.set_title("Feature Importances — Random Forest", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Importance Score (Mean Decrease in Impurity)", fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, feat_imp['Importance'].max() * 1.18)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "10_feature_importance.png"), dpi=300)
    plt.close()
    print("  [3/4] Feature importance chart saved.")


def _plot_model_comparison(acc_base, f1_base, auc_base, acc_rf, f1_rf, auc_rf, output_dir):
    """Chart 11 — Grouped bar chart comparing key metrics across both models."""
    metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
    base_vals = [acc_base, f1_base, auc_base]
    rf_vals   = [acc_rf,   f1_rf,   auc_rf]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, base_vals, width, label='Logistic Regression (Baseline)',
                   color=PALETTE['Baseline'], edgecolor='none')
    bars2 = ax.bar(x + width / 2, rf_vals,   width, label='Random Forest (Proposed)',
                   color=PALETTE['RF'],       edgecolor='none')

    # Value labels above each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=11)

    ax.set_title("Model Performance Comparison", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.axhline(y=0.5, color='white', linestyle='--', linewidth=0.8, alpha=0.4, label='Chance level')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "11_model_comparison.png"), dpi=300)
    plt.close()
    print("  [4/4] Model comparison chart saved.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir  = os.path.dirname(current_dir)
    data_path    = os.path.join(project_dir, 'data',   'youtube_data.csv')
    model_dir    = os.path.join(project_dir, 'models')
    plots_dir    = os.path.join(project_dir, 'plots')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    model_path  = os.path.join(model_dir, 'trending_rf_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')

    df = pd.read_csv(data_path)
    X, y, features = feature_engineering(df)

    model, scaler = train_and_evaluate(X, y, plots_dir)

    # Persist both model artifacts
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel  saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

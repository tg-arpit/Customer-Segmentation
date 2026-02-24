import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os, io, base64

DATA_PATH = os.path.join(os.path.dirname(__file__), 'Mall_Customers.csv')

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = ['CustomerID', 'Genre', 'Age', 'Annual_Income', 'Spending_Score']
    return df

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

def get_data_overview(df):
    info = {
        'shape': df.shape,
        'head': df.head().to_html(classes='table table-sm table-bordered', index=False),
        'describe': df.describe().round(2).to_html(classes='table table-sm table-bordered'),
        'nulls': df.isnull().sum().to_dict(),
        'genre_counts': df['Genre'].value_counts().to_dict()
    }
    return info

def plot_eda(df):
    charts = {}

    # Distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for ax, col, color in zip(axes, ['Age', 'Annual_Income', 'Spending_Score'], colors):
        ax.hist(df[col], bins=20, color=color, edgecolor='white', alpha=0.85)
        ax.set_title(col.replace('_', ' '))
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
    plt.tight_layout()
    charts['distributions'] = fig_to_base64(fig)

    # Gender breakdown
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df['Genre'].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
           colors=['#4C72B0', '#DD8452'], startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title('Gender Distribution', fontweight='bold')
    charts['gender_pie'] = fig_to_base64(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[['Age', 'Annual_Income', 'Spending_Score']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                linewidths=0.5, square=True)
    ax.set_title('Correlation Heatmap', fontweight='bold')
    charts['heatmap'] = fig_to_base64(fig)

    # Scatter: Income vs Spending
    fig, ax = plt.subplots(figsize=(7, 5))
    for genre, color in [('Male', '#4C72B0'), ('Female', '#DD8452')]:
        sub = df[df['Genre'] == genre]
        ax.scatter(sub['Annual_Income'], sub['Spending_Score'],
                   label=genre, color=color, alpha=0.7, s=60)
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_title('Annual Income vs Spending Score', fontweight='bold')
    ax.legend()
    charts['income_vs_spending'] = fig_to_base64(fig)

    return charts

def elbow_and_silhouette(df):
    X = df[['Annual_Income', 'Spending_Score']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias, sil_scores = [], []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(list(k_range), inertias, 'o-', color='#4C72B0', linewidth=2, markersize=8)
    axes[0].axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Optimal k=5')
    axes[0].set_title('Elbow Method', fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].legend()

    axes[1].plot(list(k_range), sil_scores, 's-', color='#55A868', linewidth=2, markersize=8)
    axes[1].axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Optimal k=5')
    axes[1].set_title('Silhouette Score', fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()
    plt.tight_layout()

    best_k = list(k_range)[sil_scores.index(max(sil_scores))]
    return fig_to_base64(fig), best_k, round(max(sil_scores), 4)

def run_kmeans(df, k=5):
    X = df[['Annual_Income', 'Spending_Score']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    df = df.copy()
    df['Cluster'] = labels

    # Cluster labels based on income/spending behavior
    cluster_names = {
        0: '🔵 Careful Spenders',
        1: '🟠 Standard Customers',
        2: '🟢 High Value Targets',
        3: '🔴 Impulsive Buyers',
        4: '🟣 Conservative Savers'
    }

    # Map clusters by centroid characteristics
    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    cluster_order = sorted(range(k), key=lambda i: (centers_orig[i][0], centers_orig[i][1]))
    name_keys = list(cluster_names.values())
    label_map = {orig: name_keys[rank] for rank, orig in enumerate(cluster_order)}
    df['Cluster_Name'] = df['Cluster'].map(label_map)

    colors = ['#4C72B0', '#DD8452', '#55A868', '#c44e52', '#9467bd']

    # 2D Cluster plot (Income vs Spending)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(k):
        sub = df[df['Cluster'] == i]
        ax.scatter(sub['Annual_Income'], sub['Spending_Score'],
                   s=80, color=colors[i], label=f'Cluster {i}', alpha=0.8)
    centers_inv = scaler.inverse_transform(km.cluster_centers_)
    ax.scatter(centers_inv[:, 0], centers_inv[:, 1],
               s=250, c='black', marker='X', zorder=5, label='Centroids')
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Spending Score (1-100)', fontsize=12)
    ax.set_title(f'K-Means Clustering (k={k}): Income vs Spending Score', fontweight='bold', fontsize=13)
    ax.legend()
    chart_2d = fig_to_base64(fig)

    # 3D-style scatter (Age vs Income, colored by cluster)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(k):
        sub = df[df['Cluster'] == i]
        ax.scatter(sub['Age'], sub['Annual_Income'],
                   s=80, color=colors[i], label=f'Cluster {i}', alpha=0.8)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Annual Income (k$)', fontsize=12)
    ax.set_title('Cluster View: Age vs Annual Income', fontweight='bold', fontsize=13)
    ax.legend()
    chart_age = fig_to_base64(fig)

    # Cluster profile bar charts
    profile = df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean().round(1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Cluster Profiles (Mean Values)', fontweight='bold', fontsize=13)
    for ax, col in zip(axes, ['Age', 'Annual_Income', 'Spending_Score']):
        bars = ax.bar(profile.index, profile[col], color=colors, edgecolor='white', linewidth=1.5)
        ax.set_title(col.replace('_', ' '))
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Mean Value')
        for bar, val in zip(bars, profile[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    chart_profile = fig_to_base64(fig)

    # Cluster size pie
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = df['Cluster'].value_counts().sort_index()
    ax.pie(sizes, labels=[f'Cluster {i}' for i in sizes.index],
           autopct='%1.1f%%', colors=colors, startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title('Customer Distribution by Cluster', fontweight='bold')
    chart_size = fig_to_base64(fig)

    # Summary table
    summary = df.groupby('Cluster').agg(
        Count=('CustomerID', 'count'),
        Avg_Age=('Age', 'mean'),
        Avg_Income=('Annual_Income', 'mean'),
        Avg_Spending=('Spending_Score', 'mean')
    ).round(1)
    summary.index = [label_map.get(i, f'Cluster {i}') for i in summary.index]
    summary_html = summary.to_html(classes='table table-bordered table-hover')

    sil = round(silhouette_score(X_scaled, labels), 4)

    return {
        'chart_2d': chart_2d,
        'chart_age': chart_age,
        'chart_profile': chart_profile,
        'chart_size': chart_size,
        'summary_html': summary_html,
        'silhouette': sil,
        'n_customers': len(df)
    }

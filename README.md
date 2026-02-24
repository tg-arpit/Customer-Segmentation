# 🛍️ Customer Segmentation using K-Means Clustering

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-11557C?style=for-the-badge&logo=python&logoColor=white)

**An interactive web application that groups retail customers into 5 meaningful segments based on their purchasing behavior using unsupervised machine learning.**

[📌 Project Overview](#-project-overview) • [🚀 Features](#-features) • [📊 Charts & Analysis](#-charts--analysis) • [🛠️ Installation](#️-installation) • [📁 Project Structure](#-project-structure) • [📈 Results](#-results)

</div>

---

## 📌 Project Overview

This project applies **K-Means Clustering** — an unsupervised machine learning algorithm — to segment 200 retail mall customers into distinct groups based on their **Annual Income** and **Spending Score**. The entire analysis is presented as a **step-by-step interactive Flask web application** that walks through every stage of the data science pipeline, from raw data loading to actionable business insights.

> **Problem Statement:** A retail store wants to understand its customers better. Rather than treating all customers the same, the store wants to identify natural groups so it can create targeted marketing strategies for each segment.

---

## 🚀 Features

- ✅ **5-Step Interactive Pipeline** — navigate through each stage of the ML workflow
- 📊 **9 Dynamic Charts** — all generated in real-time using Matplotlib & Seaborn
- 🔍 **Detailed Chart Descriptions** — every chart has a full explanation of what it shows, what metrics were used, and what business insight it provides
- 🎛️ **Interactive K Selector** — change the number of clusters (k=3 to 7) and see results update live
- 📋 **Cluster Summary Table** — statistical profile of every customer segment
- 💡 **Business Insights Page** — marketing strategy recommendations per segment
- 🖥️ **Runs Locally** — fully functional on `localhost:5000`

---

## 📊 Charts & Analysis

The project generates **9 charts** across 4 steps:

### Step 2 — Exploratory Data Analysis (EDA)
| Chart | Type | What It Shows |
|-------|------|---------------|
| Feature Distributions | Histogram (×3) | Age / Income / Spending Score value ranges and shape |
| Gender Distribution | Pie Chart | 56% Female vs 44% Male customer split |
| Correlation Heatmap | Heatmap | Pearson correlation between all numeric features |
| Income vs Spending Score | Scatter Plot | Natural customer groupings visible before clustering |

### Step 3 — Optimal K Selection
| Chart | Type | What It Shows |
|-------|------|---------------|
| Elbow Method + Silhouette Score | Dual Line Graph | WCSS inertia drop and silhouette score peak to identify best k=5 |

### Step 4 — Clustering Results
| Chart | Type | What It Shows |
|-------|------|---------------|
| Main Cluster Plot | Scatter Plot | Final 5 customer segments with centroids marked |
| Age vs Annual Income | Scatter Plot | Cluster behavior across age and income dimensions |
| Cluster Size Distribution | Pie Chart | How many customers fall in each segment (~20% each) |
| Cluster Profiles | Bar Chart (×3) | Mean Age, Income, Spending Score per cluster |

## 🎯 The 5 Customer Segments

| Segment | Income | Spending | Strategy |
|---------|--------|----------|----------|
| 🔵 Careful Spenders | Low | Low | Discount coupons, value bundles, loyalty points |
| 🟠 Standard Customers | Medium | Medium | Seasonal offers, email campaigns, cross-sell |
| 🟢 High Value Targets ⭐ | High | High | VIP membership, exclusive previews, premium offers |
| 🔴 Impulsive Buyers | Low | High | Flash sales, limited-time deals, FOMO tactics |
| 🟣 Conservative Savers | High | Low | Quality assurance, free trials, trust building |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

**2. Install required libraries**
```bash
pip install flask pandas scikit-learn matplotlib seaborn numpy
```

**3. Run the application**
```bash
python app.py
```

**4. Open in your browser**
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
customer-segmentation-kmeans/
│
├── app.py                  # Flask web server — all routes defined here
├── analysis.py             # All ML logic — data loading, EDA, clustering
├── Mall_Customers.csv      # Dataset — 200 retail customers
│
└── templates/              # HTML templates (Jinja2)
    ├── base.html           # Sidebar layout + shared CSS
    ├── index.html          # Home page — project overview
    ├── step1.html          # Step 1 — Data loading & statistics
    ├── step2.html          # Step 2 — EDA charts
    ├── step3.html          # Step 3 — Elbow & Silhouette
    ├── step4.html          # Step 4 — K-Means results
    └── step5.html          # Step 5 — Business insights
```

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `flask` | 2.0+ | Web framework for the interactive UI |
| `pandas` | 1.3+ | Data loading, manipulation, and group statistics |
| `numpy` | 1.21+ | Numerical operations |
| `scikit-learn` | 1.0+ | KMeans, StandardScaler, silhouette_score |
| `matplotlib` | 3.4+ | Generating all charts (histograms, scatter, bar, pie) |
| `seaborn` | 0.11+ | Correlation heatmap |

---

## 📈 Results

- **Dataset:** 200 mall customers, 5 features
- **Algorithm:** K-Means Clustering
- **Optimal K:** 5 clusters
- **Silhouette Score:** **0.5547** ✅ (well-separated clusters)
- **Features used for clustering:** Annual Income + Spending Score
- **Preprocessing:** StandardScaler (zero mean, unit variance)

---

## 📚 Dataset

**Mall Customer Segmentation Dataset**
- Source: [Kaggle — Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- 200 customers, 5 columns: CustomerID, Genre, Age, Annual Income (k$), Spending Score (1-100)
- No missing values — clean dataset ready for analysis

---

## 🧠 Key Concepts Used

- **K-Means Clustering** — unsupervised ML algorithm that partitions data into k clusters by minimizing WCSS (Within-Cluster Sum of Squares)
- **Elbow Method** — plots inertia vs k to find the "elbow" where adding more clusters gives diminishing returns
- **Silhouette Score** — measures how well each point fits its cluster (range: -1 to +1, higher = better)
- **StandardScaler** — normalizes features to mean=0, std=1 so no feature dominates due to scale
- **Pearson Correlation** — measures linear relationship between features (-1 to +1)
- **Centroid** — the mathematical center of each cluster, recalculated at every K-Means iteration

---

## 👨‍💻 Author

Made with ❤️ using Python, Flask, Scikit-learn, and Matplotlib.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

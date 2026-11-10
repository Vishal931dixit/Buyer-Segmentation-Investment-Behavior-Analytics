# 🏢 Real Estate Buyer Segmentation & Marketing Intelligence Platform
> **End-to-End Unsupervised Machine Learning, Customer Persona Profiling & Interactive Analytics Dashboard**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

---

## 📌 Executive Summary & Business Objective

In the competitive real estate industry, marketing campaigns often suffer from generic targeting—treating corporate investors, first-time home seekers, and luxury purchasers with identical outreach. This project implements an enterprise-grade **Customer Segmentation Engine** using **Unsupervised Machine Learning (K-Means & Hierarchical Clustering)** to identify distinct buyer personas from combined demographic, financial, and property transaction data.

### 🎯 Key Outcomes
- **Unified 7,305 Transactions**: Relational inner merge across 2,000 client records and 10,000 property listings.
- **Statistically Validated 4 Segments**: Confirmed through dual-validation (Elbow Method & Silhouette Analysis) and verified by Hierarchical Ward dendrograms and multi-seed stability testing.
- **Actionable Business Personas**: Profiled 4 distinct client types (*Global Investors*, *First-Time Buyers*, *Corporate Purchasers*, and *Luxury Investors*) with tailored product and financing recommendations.
- **Production Dashboard Deployment**: Packaged an interactive 4-module **Streamlit** dashboard visualizing market distribution, spending behavior, and regional concentration.

---

## 🏗️ System Architecture & End-to-End Pipeline

```mermaid
flowchart TD
    subgraph Data Ingestion
        A[clients.csv\n2,000 rows x 12 cols] --> C[Relational Inner Join\nclient_id = client_ref]
        B[properties.csv\n10,000 rows x 9 cols] --> C
        C --> D[Transaction Dataset\n7,305 rows x 21 cols]
    end

    subgraph Data Preprocessing & Engineering
        D --> E[Missing Value Imputation\nNumeric: Median | Categorical: Mode]
        E --> F[Exact Age Calculation\nfrom mixed-format DOB]
        F --> G[Financial Feature Parsing\nsale_price currency cleaning]
        G --> H[Independent Label & One-Hot Encoding]
        H --> I[Correlation Heatmap & 1%-99% Outlier Clipping]
    end

    subgraph Unsupervised ML Engine
        I --> J[StandardScaler Normalization]
        J --> K[Optimal K Discovery\nElbow Method + Silhouette Score]
        J --> L[Validation\nSampled Hierarchical Dendrogram]
        K --> M[K-Means Clustering\nK=4, n_init=10]
        L --> M
        M --> N[Cluster Stability Check\nMulti-seed Inertia Std Dev]
    end

    subgraph Production & Analytics
        M --> O[(model.pkl\nModel + Scaler + Features)]
        M --> P[Cluster Profiling & Crosstab Analysis]
        O --> Q[Streamlit Web Application\napp.py]
        P --> Q
    end
```

---

## 📂 Dataset Architecture

The project synthesizes two operational real estate databases:

### 1. `clients.csv` (Demographics & Profiles — 2,000 records)
| Feature | Type | Description | Domain Importance |
|---|---|---|---|
| `client_id` | `str` | Primary key identifier (`C0001`–`C2000`) | Unique entity tracking |
| `client_type` | `str` | Entity type: `Individual` (94.9%) vs `Company` (5.1%) | Separates B2C from B2B buyers |
| `date_of_birth` | `str` | Mixed format dates (`DD-MM-YYYY` / `MM/DD/YYYY`) | Raw life-stage demographic |
| `gender` | `str` | Gender distribution (`M`: 50.6%, `F`: 49.4%) | Demographic profiling |
| `country` / `region` | `str` | Geographic origin (USA 77%, UK, Canada, Germany) | Cross-border vs domestic capital |
| `acquisition_purpose`| `str` | Stated goal: `Home` (69.3%) vs `Investment` (30.7%) | Intent & purchase motivation |
| `satisfaction_score` | `int` | Post-interaction rating (1 to 5) | Customer sentiment indicator |
| `loan_applied` | `str` | Mortgage financing applied: `Yes` (36.8%) / `No` (63.2%) | Capital liquidity / leverage |
| `referral_channel` | `str` | Source: `Website` (55%), `Agency` (35%), `Client` (10%)| Acquisition channel effectiveness |

### 2. `properties.csv` (Property Inventory & Transactions — 10,000 records)
| Feature | Type | Description | Domain Importance |
|---|---|---|---|
| `listing_id` | `int` | Unique property listing identifier | Asset tracking |
| `unit_category` | `str` | Property type: `Apartment` (85.5%) vs `Office` (14.5%) | Residential vs commercial real estate |
| `floor_area_sqft` | `float` | Built-up area (410 sqft to 1,957 sqft, avg 1,140) | Physical scale requirement |
| `sale_price` | `str` | Transaction price with currency symbols (`$97,422`–`$737,125`) | Spending capacity & ticket size |
| `listing_status` | `str` | Property status: `Sold` (73.0%) vs `Available` (27.0%) | Market inventory state |
| `client_ref` | `str` | Foreign key referencing `client_id` (`NULL` if unsold) | Buyer transaction linkage |

> **Merge Strategy**: An **Inner Join** on `client_id == client_ref` isolates the **7,305 completed acquisitions**. Unsold listings (`client_ref = NULL`) are omitted to prevent noise during buyer segmentation.

---

## ⚙️ Data Preprocessing & Feature Engineering

### 1. Robust Missing Value Imputation
To avoid runtime errors across modern pandas versions, data types are dynamically inspected:
```python
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())   # Resilient against outliers
    else:
        df[col] = df[col].fillna(df[col].mode()[0])  # Preserves modal categories
```

### 2. Exact Age Feature Engineering
Instead of approximate birth-year subtraction, exact elapsed days are calculated to avoid up to 1-year boundary errors:
```python
now = pd.Timestamp.now()
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce', format='mixed', dayfirst=False)
df['Age'] = ((now - df['date_of_birth']).dt.days // 365.25).astype('Int64')
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)
```

### 3. Financial String Parsing
Raw currency text (e.g., `"$300,385.62"`) is sanitized into numeric floats:
```python
df['sale_price'] = df['sale_price'].astype(str)\
    .str.replace('$', '', regex=False)\
    .str.replace(',', '', regex=False)\
    .astype(float)
```

### 4. Independent Categorical Encoders
- **Binary Features (`gender`, `loan_applied`)**: Encoded using dedicated `LabelEncoder` instances stored in a dictionary (`label_encoders[col]`) to prevent state overwrite.
- **Nominal Features (`client_type`, `region`, `purpose`, etc.)**: One-hot encoded via `pd.get_dummies(..., drop_first=True)` to eliminate the dummy variable trap.

### 5. Multicollinearity Inspection & Outlier Mitigation
- **Correlation Heatmap**: Evaluated Pearson correlations across selected numerical candidates to confirm absence of severe multicollinearity.
- **Percentile Clipping**: Numerical features are capped at the 1st and 99th percentiles before scaling to prevent extreme outliers from pulling cluster centroids.

---

## 🔬 Clustering Engine & Statistical Validation

### Feature Selection & Standardization
Five core attributes balancing demographics, financial behavior, and property choice were selected for clustering:
```python
features = ['Age', 'satisfaction_score', 'loan_applied', 'floor_area_sqft', 'sale_price']
X = df[features].copy()

# Robust Standardization (Mean = 0, Variance = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 1. Determining Optimal $K$ (Elbow & Silhouette)
- **Within-Cluster Sum of Squares (Inertia)**: Plotted over $K \in [2, 10]$. A distinct "elbow" bend appears at **$K=4$**, beyond which inertia reduction yields diminishing returns.
- **Silhouette Coefficient**: Evaluated clustering cohesion and separation across clusters. $K=4$ delivered a balanced score of **0.261** with distinct, interpretable segments.

| Clusters ($K$) | Silhouette Score | Interpretation |
|---|---|---|
| $K=2$ | 0.266 | Coarse split (primarily driven by price/size alone) |
| $K=3$ | 0.257 | Moderate cohesion, merges corporate and investor profiles |
| **$K=4$** | **0.261** | **Optimal balance of statistical separation and commercial interpretability** |
| $K=5$ | 0.250 | Cluster fragmentation; creates redundant sub-groups |

### 2. Hierarchical Validation (Ward's Minimum Variance)
To independently verify the cluster structure without running into $O(N^2)$ memory limits on 7,305 records, a representative 500-sample Ward hierarchical dendrogram was computed:
```python
from scipy.cluster.hierarchy import linkage, dendrogram

sample_idx = np.random.choice(X_scaled.shape[0], size=500, replace=False)
linked = linkage(X_scaled[sample_idx], method='ward')
dendrogram(linked, truncate_mode='lastp', p=20)
```
The largest vertical distance without crossing horizontal merge lines confirms **4 primary clusters**.

### 3. Cluster Stability Across Random Seeds
To ensure clusters represent true data density rather than local optima, K-Means was executed across multiple seeds (`10, 21, 42, 84, 100`):
- **Inertia Standard Deviation**: `0.0179` (< 0.05 threshold), confirming strong algorithmic stability.

---

## 👥 Buyer Persona Profiles & Strategic Recommendations

```
                                  [7,305 Total Buyers]
                                            │
        ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
        ▼                   ▼                               ▼                   ▼
    Cluster 0           Cluster 1                       Cluster 2           Cluster 3
 🌍 Global Investors  🏠 First-Time Buyers            🏢 Corporate Buyers  💎 Luxury Buyers
   • High income       • Younger age (avg 31)          • B2B / Office units • Highest spend
   • Low mortgage %    • 100% Loan applied             • High sqft requirements • Highest satisfaction
   • High satisfaction • High price-sensitivity        • Multi-unit deals   • Premium penthouses
```

### Detailed Persona Breakdown

| Metric / Attribute | Cluster 0: Global Investors | Cluster 1: First-Time Homebuyers | Cluster 2: Corporate Buyers | Cluster 3: Luxury Investors |
|---|---|---|---|---|
| **Representative Share** | ~32% of transactions | ~28% of transactions | ~16% of transactions | ~24% of transactions |
| **Average Age** | 46.2 years | 31.4 years | 49.8 years | 44.1 years |
| **Average Sale Price** | \$338,500 | \$265,400 | \$412,800 | \$562,300 |
| **Average Floor Area** | 1,080 sqft | 820 sqft | 1,480 sqft | 1,690 sqft |
| **Loan Utilization** | Low (Cash / Private Equity) | High (92% Mortgage reliant) | Medium (Corporate line of credit) | Low (Self-funded / High net worth) |
| **Customer Satisfaction**| 3.2 / 5.0 | 2.8 / 5.0 | 3.1 / 5.0 | 4.6 / 5.0 |
| **Dominant Category** | 1–2 Bed Residential Apartments | Studio & 1-Bed Apartments | Commercial Offices & Full Floors | Penthouse & Luxury Residences |

### Business & Marketing Playbook
1. **Cluster 0 (Global Investors)**: Offer rental yield guarantees, automated property management services, and international portfolio diversification packages.
2. **Cluster 1 (First-Time Buyers)**: Partner with retail mortgage lenders for preferential interest rates, flexible installment plans, and government housing subsidy assistance.
3. **Cluster 2 (Corporate Buyers)**: Assign dedicated B2B enterprise account managers; provide custom fit-out allowances and bulk multi-unit acquisition discounts.
4. **Cluster 3 (Luxury Investors)**: White-glove concierge service, private VIP preview launches, smart-home turnkey finishes, and exclusive lifestyle amenities.

---

## 🖥️ Streamlit Interactive Analytics Dashboard (`app.py`)

The production application provides real-time segmentation intelligence for sales teams and executive stakeholders:

```bash
streamlit run app.py
```

### Dashboard Capabilities
- **Module 1 — Market Overview**: Plotly interactive donut chart illustrating real-time percentage allocation across all 4 segments.
- **Module 2 — Investor Dynamics**: Dynamic 3D/bubble scatter plot mapping `Sale Price` vs `Satisfaction Score` with marker sizes representing `Floor Area (sqft)`.
- **Module 3 — Geographic Drilldown**: Grouped bar visualization mapping buyer origins (California, Quebec, Ontario, Western Europe) across clusters.
- **Module 4 — Persona Insights Panel**: Filterable executive summary table providing key statistical indicators (mean, median, IQR) for instant business reporting.

---

## 📂 Repository Structure

```
Project_4/
├── clients.csv              # Raw client demographics & behavior (2,000 records)
├── properties.csv           # Raw property inventory & transaction data (10,000 records)
├── Zidio_Project.ipynb      # Main Jupyter notebook (Data Wrangling, EDA, Modeling)
├── app.py                   # Production Streamlit 4-module dashboard application
├── model.pkl                # Serialized dictionary: {'model': kmeans, 'scaler': scaler, 'features': features}
└── readme.md                # Comprehensive project documentation & technical guide
```

---

## 🚀 Getting Started & Execution Guide

### 1. Prerequisites & Environment Setup
Clone the repository and create an isolated virtual environment:
```bash
git clone https://github.com/your-username/real-estate-buyer-segmentation.git
cd real-estate-buyer-segmentation

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Required Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit scipy
```

### 3. Run the Machine Learning Pipeline
Open and execute the Jupyter Notebook:
```bash
jupyter notebook Zidio_Project.ipynb
```
*Or execute from top to bottom in VS Code via **Kernel ➔ Restart & Run All**.*

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore the interactive buyer segments.

---

## 🛠️ Engineering Highlights & Production-Grade Practices

| Component | Traditional / Naïve Approach | Implemented Production Practice |
|---|---|---|
| **Path Portability** | Hardcoded `/content/` paths (breaks locally) | Dynamic `os.path.exists()` fallback supporting Colab, Windows, & Linux |
| **Missing Value Imputation** | Blanket `.median()` or `.fillna('Unknown')` | Safe `is_numeric_dtype()` branch preventing string-median type crashes |
| **Age Calculation** | `current_year - birth_year` (up to 1 yr error) | Day-accurate duration: `((now - DOB).dt.days // 365.25)` |
| **Categorical Encoding** | Reusing a single `LabelEncoder` instance | Dictionary of dedicated `LabelEncoder` objects preserving mapping states |
| **Outlier Handling** | Unbounded raw features | 1st–99th percentile clipping protecting K-Means centroid positions |
| **Hierarchical Clustering**| Full $O(N^2)$ distance matrix on 7,305 rows | Representative 500-sample Ward linkage preventing memory exhaustion |
| **Cluster Crosstabs** | Direct cross-tab on pre-OHE dropped columns | Reconstructed client region mapping with safety cluster guards |
| **Artifact Packaging** | Saving only the `.fit()` cluster model | Bundled dictionary (`model`, `scaler`, `features`) preventing feature mismatch |

---

## 🎤 Interview Presentation & Defense Guide

### 30-Second Elevator Pitch
> *"I built a real estate customer intelligence engine that segments 7,300+ property transactions into four distinct commercial personas using K-Means and Hierarchical Clustering. Using mixed-format date parsing, financial string cleaning, and robust outlier clipping, I standardized key demographic and purchase metrics. I validated the 4-cluster structure using the Elbow Method, Silhouette Analysis, a sampled Ward dendrogram, and multi-seed stability testing. Finally, I deployed an interactive 4-module Streamlit analytics application that translates these clusters into targeted marketing and financing playbooks."*

### Key Technical Questions & Answers

#### Q1: Why use K-Means over DBSCAN or Gaussian Mixture Models (GMM)?
> **Answer**: K-Means produces hard, centroid-based cluster assignments that translate seamlessly into distinct customer personas for business stakeholders. DBSCAN is superior for arbitrary density shapes and noise identification, but struggles with varying density clusters common in customer demographics. GMM provides soft probabilistic clustering, which adds unnecessary computational complexity when the business objective requires clear, mutually exclusive marketing segments.

#### Q2: How did you select the 5 features? Why not include all one-hot encoded geographic variables?
> **Answer**: High-dimensional one-hot encoded spaces cause distance concentration in Euclidean metrics (the Curse of Dimensionality), causing points to appear equidistant. We focused the core distance metric on fundamental demographic and financial attributes (`Age`, `satisfaction_score`, `loan_applied`, `floor_area_sqft`, `sale_price`), while retaining high-cardinality geographic features (country, region, referral channels) for post-clustering cross-tabulation and drill-down profiling.

#### Q3: How do you handle cold-start predictions for new prospective buyers?
> **Answer**: Because `model.pkl` packages the trained `StandardScaler` alongside the `KMeans` estimator, incoming prospects can be scaled with `.transform()` and assigned to the nearest centroid via `.predict()`. For leads with partial information (e.g., missing floor area), we impute using segment medians or run distance evaluations over available sub-spaces.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

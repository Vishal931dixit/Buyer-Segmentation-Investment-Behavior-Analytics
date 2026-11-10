# 🏠 Project 4 — Real Estate Buyer Segmentation
### Complete Step-by-Step Explanation + Data Scientist Evaluation

---

## 📌 Project Overview

| Attribute | Detail |
|-----------|--------|
| **Project Name** | Buyer Segmentation using Clustering |
| **Domain** | Real Estate / PropTech |
| **Type** | Unsupervised Machine Learning |
| **Algorithms** | K-Means Clustering, Hierarchical (Agglomerative) Clustering |
| **Deployment** | Streamlit Web Dashboard |
| **Files** | `Zidio_Project.ipynb`, `zidio_project.py`, `app.py`, `clients.csv`, `properties.csv`, `model.pkl` |

### 🎯 Business Objective
Identify distinct **buyer personas** within a real estate company's customer base by clustering clients based on their demographics, financial behaviour, and property purchase patterns — enabling **targeted marketing** and **personalized services**.

---

## 📂 Dataset Description

### File 1: `clients.csv` — 2,000 rows × 12 columns

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------|
| `client_id` | str | Unique client identifier | C0001, C0002 |
| `client_type` | str | Individual or Company | Individual: 94.9%, Company: 5.1% |
| `first_name`, `last_name` | str | Client name | — |
| `date_of_birth` | str | Mixed format dates | 05-11-1968, 11/26/1962 |
| `gender` | str | M / F | M: 50.6%, F: 49.4% |
| `country` | str | Country of origin | USA (77%), UK, Canada, Germany |
| `region` | str | Geographic region | California, Quebec, Oregon |
| `acquisition_purpose` | str | Why buying? | Home (69.3%), Investment (30.7%) |
| `satisfaction_score` | int | 1–5 satisfaction rating | Evenly distributed (avg ~3) |
| `loan_applied` | str | Yes / No | No (63.2%), Yes (36.8%) |
| `referral_channel` | str | How they found us | Website (55%), Agency (35%), Client (10%) |

### File 2: `properties.csv` — 10,000 rows × 9 columns

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------|
| `listing_id` | int | Unique property listing ID | 1012 |
| `tower_number` | int | Building tower | 1, 2, 3 |
| `transaction_date` | str | Date of transaction | 01-01-2024 |
| `unit_category` | str | Type of property | Apartment (85.5%), Office (14.5%) |
| `unit_number` | int | Unit identifier | 12, 15, 21 |
| `floor_area_sqft` | float | Property size | 410–1957 sqft (avg 1,140) |
| `sale_price` | str | Price with $ and commas | $300,385.62 (avg $344,375) |
| `listing_status` | str | Sold / Available | Sold: 73%, Available: 27% |
| `client_ref` | str | References client_id | C0027 (NULL for unsold listings) |

> **Note:** `client_ref` has 2,695 null values (the "Available" listings with no buyer yet).
> The inner join on sold properties produces **7,305 merged records**.

---

## 🔄 Step-by-Step Pipeline

### Step 1 — Import Libraries

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
```

Loads all required data manipulation, visualisation, and ML libraries.

---

### Step 2 — Load & Merge Datasets

```python
clients    = pd.read_csv("clients.csv")       # 2,000 × 12
properties = pd.read_csv("properties.csv")    # 10,000 × 9

df = pd.merge(
    clients, properties,
    left_on='client_id', right_on='client_ref',
    how='inner'   # keeps only sold properties
)
# Result: 7,305 × 21 (transaction-level records)
```

**Why inner join?** Only "Sold" properties have a `client_ref`, so an inner join naturally filters to completed transactions — the only records meaningful for buyer analysis.

---

### Step 3 — Handle Missing Values

```python
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)   # mode for categorical
    else:
        df[col].fillna(df[col].median(), inplace=True)    # median for numeric
```

**Strategy:**
- **Categorical columns** → filled with **mode** (most frequent value) — avoids creating spurious new categories
- **Numeric columns** → filled with **median** — robust to outliers unlike mean

---

### Step 4 — Remove Duplicates

```python
df.drop_duplicates(inplace=True)
```

Ensures each client-property transaction is counted only once.

---

### Step 5 — Normalize Text Labels

```python
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip().str.lower()
```

Prevents encoding errors from inconsistent casing (e.g., `"Yes"` vs `"yes"` being treated as different categories during LabelEncoding).

---

### Step 6 — Feature Engineering: Age

```python
df['date_of_birth'] = pd.to_datetime(
    df['date_of_birth'], errors='coerce', format='mixed', dayfirst=False
)
df['Age'] = pd.Timestamp.now().year - df['date_of_birth'].dt.year
df['Age'].fillna(df['Age'].median(), inplace=True)   # handle unparseable dates
```

**Why create Age?**
- Raw `date_of_birth` is a string — unusable by ML algorithms
- Age is a **continuous, meaningful** demographic feature
- `format='mixed'` handles both `DD-MM-YYYY` and `MM/DD/YYYY` formats in the dataset
- Unparseable dates become `NaN`, filled with the median age

---

### Step 7 — Label Encoding (Binary Columns)

```python
le = LabelEncoder()
for col in ['gender', 'loan_applied']:
    df[col] = le.fit_transform(df[col])
```

| Original | Encoded |
|----------|---------|
| gender: f / m | 0 / 1 |
| loan_applied: no / yes | 0 / 1 |

**Why Label Encoding here?** These are **binary** columns with only 2 categories — no ordinal bias is introduced.

---

### Step 8 — One-Hot Encoding (Multi-Category Columns)

```python
categorical_cols = [
    'client_type', 'country', 'region', 'acquisition_purpose',
    'referral_channel', 'unit_category', 'listing_status'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

**Why OHE?** These columns have 3+ unordered categories. Label Encoding would impose a false ordinal relationship (e.g., Website=0, Agency=1, Client=2 implies Agency is "between" the other two). `drop_first=True` avoids the **dummy variable trap** (multicollinearity).

---

### Step 9 — Clean `sale_price` Column

```python
df['sale_price'] = df['sale_price'].astype(str)\
    .str.replace('$', '', regex=False)\
    .str.replace(',', '', regex=False)\
    .astype(float)
```

The price was stored as a **formatted string** (e.g., `$300,385.62`). Removes the currency symbol and thousands separator to convert to a usable float.

---

### Step 10 — Feature Selection

```python
features = ['Age', 'satisfaction_score', 'loan_applied', 'floor_area_sqft', 'sale_price']
X = df[features]
```

**Why these 5 features?**

| Feature | What It Captures |
|---------|---------|
| `Age` | Demographic life-stage of the buyer |
| `satisfaction_score` | Post-purchase experience / loyalty signal |
| `loan_applied` | Financial dependency / creditworthiness |
| `floor_area_sqft` | Property size preference / budget tier |
| `sale_price` | Actual spending power |

These 5 form a compact, balanced representation of **who the buyer is** and **what they buy**.

---

### Step 11 — Feature Scaling

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why scaling is CRITICAL for K-Means:**
K-Means computes Euclidean distances. Without scaling, `sale_price` (~$344K average) would completely dominate `loan_applied` (0 or 1). `StandardScaler` transforms each feature to mean=0, std=1, giving equal weight to all features.

---

### Step 12 — Optimal K: Elbow Method

```python
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)   # Within-Cluster Sum of Squares

plt.plot(range(2, 11), wcss, marker='o')
```

**What is WCSS (Inertia)?**
The sum of squared distances of all points to their assigned cluster centroid. As K increases, WCSS always decreases — the "elbow" is where the rate of decrease sharply flattens, indicating diminishing returns from adding more clusters. The elbow was identified at **K=4**.

---

### Step 13 — Silhouette Score Validation

```python
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Clusters: {i}  Score: {round(score, 3)}")
```

**What is Silhouette Score?**
- Range: **-1 to +1**
- For each point: measures how similar it is to its own cluster vs. neighbouring clusters
- Score closer to **+1** = well-separated, dense clusters
- Used to confirm K=4 as the statistically optimal cluster count

**Decision:** Both Elbow Method and Silhouette Score confirm **K=4** as optimal.

---

### Step 14 — Final K-Means Model (K=4)

```python
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

Assigns each of the 7,305 buyer-transaction records to one of 4 clusters.

---

### Step 15 — Hierarchical Clustering (Validation)

```python
linked = linkage(X_scaled, method='ward')
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
```

**Why use Hierarchical Clustering?**
Used as a **cross-validation method** to independently confirm that 4 clusters is the natural grouping. Ward linkage minimizes total within-cluster variance.

**How to read the Dendrogram:** The height of each merge (y-axis) indicates how dissimilar clusters are when joined. The largest vertical gap in merge heights reveals the natural number of clusters.

---

### Step 16 — Cluster Analysis & Profiling

```python
cluster_analysis = df.groupby('Cluster')[['Age', 'satisfaction_score', 'sale_price']].mean()
pd.crosstab(df['Cluster'], clients['region'])    # Geographic distribution
pd.crosstab(df['Cluster'], df['loan_applied'])   # Loan behavior
```

Profiling each cluster across key dimensions to give them meaningful business names.

---

### Step 17 — Cluster Visualization

```python
sns.scatterplot(x=df['Age'], y=df['sale_price'], hue=df['Cluster'], palette='Set2')
plt.title("Customer Segments")
```

Plots Age vs Sale Price coloured by cluster — allows visual inspection of segment separation.

---

### Step 18 — Identified Buyer Segments

| Cluster | Buyer Type | Key Characteristics |
|---------|-----------|---------------------|
| **C1** | 🌍 Global Investors | High income, investment-purpose purchases, low loan dependency |
| **C2** | 🏠 First-Time Buyers | Younger age, high loan dependency, home acquisition purpose |
| **C3** | 🏢 Corporate Buyers | Company client_type, multiple unit purchases, office category |
| **C4** | 💎 Luxury Investors | Highest sale prices, high satisfaction scores, large floor areas |

---

### Step 19 — Model Serialization

```python
model_objects = {"model": kmeans, "scaler": scaler, "features": features}
with open("model.pkl", "wb") as f:
    pickle.dump(model_objects, f)
```

Saves the trained KMeans model, the fitted scaler, and the feature list together — so the app can load and apply them consistently without retraining. All three must be bundled to avoid feature mismatch errors.

---

### Step 20 — Streamlit Dashboard (`app.py`)

```
streamlit run app.py
```

**4-Module Interactive Dashboard:**

#### Module 1 — Buyer Segmentation Overview
- **Plotly Pie Chart** showing the % distribution of buyers across the 4 clusters

#### Module 2 — Investor Behavior Dashboard
- **Plotly Bubble Scatter Plot**: x=sale_price, y=satisfaction_score, size=floor_area_sqft, color=Cluster
- Reveals relationship between investment size and satisfaction

#### Module 3 — Geographic Buyer Analysis
- **Grouped Bar Chart**: Region × Cluster cross-tabulation
- Shows which regions contribute most to each buyer type

#### Module 4 — Segment Insights Panel
- **Summary Table**: Mean Age, Sale Price, Satisfaction Score per cluster
- Directly interpretable stats for business stakeholders

---

## 🔁 End-to-End Data Flow

```
clients.csv (2,000×12)  ──┐
                           ├─► Inner Join (client_id = client_ref)
properties.csv (10,000×9)─┘
              │
              ▼ Merged df (7,305 × 21)
              │
     ┌────────▼────────┐
     │  Data Cleaning   │  Missing values, duplicates, text normalization
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │Feature Engineering│  Age from date_of_birth
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │   Encoding       │  Label Encoding (binary) + OHE (multi-class)
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │Feature Selection │  5 features: Age, satisfaction, loan, sqft, price
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │  StandardScaler  │  Normalize to mean=0, std=1
     └───┬─────────────┘
         │
   ┌─────▼──────┐  ┌─────────────────────┐
   │Elbow Method│  │Hierarchical Dendogram│
   │ Silhouette │  │   (Ward linkage)     │
   └─────┬──────┘  └──────────┬──────────┘
         └────────┬───────────┘
                  │ Both confirm K=4
         ┌────────▼────────┐
         │  KMeans (K=4)   │──► model.pkl
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │Cluster Profiling │  Analysis + Visualization
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  app.py          │  Streamlit 4-Module Dashboard
         └─────────────────┘
```

---

## 🐛 Code Issues Identified

> **CAUTION — These bugs would cause runtime errors if run as-is:**

| # | File | Line | Issue | Correct Fix |
|---|------|------|-------|-------------|
| 1 | `zidio_project.py` | 402 | `df["Cluster"]=clusters` — `clusters` is **undefined** at notebook level | Use `df["Cluster"]=kmeans.labels_` |
| 2 | `app.py` | 254 | `summary=(=` — **syntax error** (double equals) | Change to `summary = (` |
| 3 | `zidio_project.py` | 408–545 | Streamlit code mixed into notebook script | Isolate only in `app.py` |
| 4 | `zidio_project.py` | 167+189 | `sale_price` cleaning done **twice** redundantly | Remove line 189 |

---

## 📈 Data Scientist Evaluation

### ✅ Strengths

| Skill Area | What Was Done Well |
|---|---|
| **Problem Framing** | Clear unsupervised learning objective with real business value (buyer segmentation = actionable marketing) |
| **Data Merging** | Correctly identified and used the right join key (`client_id` ↔ `client_ref`) with appropriate `inner` join |
| **Mixed Date Parsing** | Used `format='mixed'` to handle inconsistent date formats — a practical real-world skill |
| **Missing Value Strategy** | Appropriate use of mode (categorical) and median (numeric) — not just blanket mean imputation |
| **Encoding Choice** | Correctly differentiated between binary (Label Encoding) and multi-class (OHE) with `drop_first=True` |
| **Currency Parsing** | Cleaned `$` and `,` from price strings — handling messy real-world financial data |
| **Scaling Awareness** | Used `StandardScaler` and explicitly understood WHY it's needed for K-Means |
| **Dual Validation** | Used both Elbow Method AND Silhouette Score to choose K — shows methodological rigour |
| **Cross-Validation** | Used Hierarchical Clustering as independent verification of K=4 |
| **Business Translation** | Named clusters meaningfully (Global Investors, First-Time Buyers, etc.) |
| **Model Persistence** | Saved model + scaler + features together in one `.pkl` — production-aware design |
| **Full Deployment** | Built a 4-module interactive Streamlit dashboard — goes beyond just a notebook |

### ⚠️ Areas for Improvement

| Issue | Why It Matters | How to Fix |
|---|---|---|
| **Only 5 features selected** | Drops rich OHE columns (country, region, purpose) that could sharpen clusters | Use PCA to reduce dimensionality of all features or add more features thoughtfully |
| **No feature correlation analysis** | May include correlated features (floor_area_sqft ↔ sale_price) | Add correlation heatmap and VIF check before feature selection |
| **LabelEncoder reuse** | Using same `le` instance for both binary columns overwrites state | Use separate `le` per column or use `OrdinalEncoder` |
| **Approximate Age calculation** | Uses only birth year (off by up to 1 year) | Use `relativedelta` or `.to_period('Y')` for exact age |
| **No outlier treatment** | Sale prices range $97K–$737K (7.5× range) — outliers skew centroids | Apply IQR or Z-score filtering before clustering |
| **No cluster stability test** | Single `random_state=42` may not represent global optimum | Run K-Means multiple times with different seeds + check inertia variance |
| **Hierarchical on full dataset** | Running `linkage()` on 7,305 rows is O(n²) in memory — slow | Use a sample (~500 rows) for dendrogram visualization |
| **Code bugs** | Syntax errors in `app.py`, undefined `clusters` variable | Fix before demo (see issues table above) |

### 🎓 Skill Demonstration Score

| Category | Score | Notes |
|----------|-------|-------|
| Data Wrangling | 4/5 | Good merging & cleaning; minor price cleaning redundancy |
| Feature Engineering | 3/5 | Age feature is good; misses correlation analysis & PCA |
| ML Methodology | 4/5 | Dual validation (Elbow + Silhouette) + Hierarchical is solid |
| Code Quality | 3/5 | Readable but has bugs; Streamlit code mixed in notebook |
| Business Insight | 4/5 | Good cluster naming and actionable recommendations |
| Deployment | 4/5 | Multi-module Streamlit dashboard is a strong differentiator |
| **Overall** | **3.7/5** | **Solid intermediate-level Data Science project** |

---

## 💬 How to Explain This in an Interview

### 30-Second Elevator Pitch
> *"I built a customer segmentation system for a real estate company using K-Means clustering on 2,000 buyer profiles merged with 10,000 property transactions. After preprocessing and feature engineering including age extraction from mixed-format dates and currency parsing, I validated the optimal 4 clusters using both the Elbow Method and Silhouette Scores, confirmed by a Hierarchical dendrogram. The model identified four distinct buyer types — Global Investors, First-Time Buyers, Corporate Buyers, and Luxury Investors — and I deployed the results as an interactive Streamlit dashboard with geographic and behavioral drill-downs."*

### Key Technical Points to Highlight
1. **Why K-Means?** Fast, scalable, interpretable centroids — good for profiling use case
2. **Why K=4?** Elbow curve flattened + Silhouette peaked at K=4 + dendrogram confirmed
3. **Why Standard Scaler?** K-Means uses Euclidean distance — unscaled features with large ranges dominate
4. **Why inner join?** Only sold properties have a buyer link — available listings are irrelevant for segmentation
5. **Model pkl design** — bundled scaler + features + model so the app never needs to retrain or risk feature mismatch

### Potential Follow-up Questions & Answers

**Q: Why not use DBSCAN or GMM instead of K-Means?**
> K-Means is chosen for interpretability — business stakeholders understand centroid-based segments easily. DBSCAN would work if we expected non-spherical clusters, and GMM if we needed probabilistic cluster membership. K-Means was the right pragmatic choice here.

**Q: How would you improve this model?**
> Add all OHE features and apply PCA to reduce to 10–15 dimensions before clustering. Add outlier detection using IQR. Use `KMeans++` initialization (default in sklearn). Validate cluster stability across multiple random seeds.

**Q: What is your biggest concern with this approach?**
> The age calculation is approximate (year-only). More critically, running Hierarchical Clustering on the full 7,305 rows is memory-intensive — a sampled dendrogram would be computationally safer.

---

## 📁 Project Files

| File | Role |
|------|------|
| `Zidio_Project.ipynb` | Main Jupyter notebook (EDA + ML pipeline) |
| `zidio_project.py` | Auto-exported Python script from notebook |
| `app.py` | Streamlit deployment dashboard |
| `clients.csv` | Client demographics dataset (2,000 rows × 12 cols) |
| `properties.csv` | Property transaction dataset (10,000 rows × 9 cols) |
| `model.pkl` | Serialized KMeans + StandardScaler + feature list (30 KB) |

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(
    page_title="Buyer Segmentation Dashboard",
    layout="wide"
)

st.title("🏠 Real Estate Customer (Buyer) Segmentation Dashboard")
st.markdown("*Multi-Pillar Segmentation: Demographics, Investment Purpose, Financing Behavior & Geographic Patterns*")

######################################################
# Load Data
######################################################

clients = pd.read_csv("clients.csv")
properties = pd.read_csv("properties.csv")

# Clean properties sale_price
properties_clean = properties.dropna(subset=['client_ref']).copy()
properties_clean['sale_price'] = properties_clean['sale_price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

# Aggregate properties per client for true Customer (Buyer) Segmentation
buyer_props = properties_clean.groupby('client_ref').agg(
    sale_price=('sale_price', 'mean'),             # Average property ticket size
    floor_area_sqft=('floor_area_sqft', 'mean'),   # Average floor area acquired
    total_spend=('sale_price', 'sum'),             # Total portfolio investment
    properties_bought=('listing_id', 'count')      # Portfolio unit count
).reset_index()

df = pd.merge(clients, buyer_props, left_on='client_id', right_on='client_ref', how='inner')

# Retain unencoded display columns for drill-down analysis
df['region_display'] = df['region'].copy()
df['country_display'] = df['country'].copy()

######################################################
# Load Model & Pipeline Artifacts
######################################################

with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
scaler = model_data["scaler"]
pca = model_data["pca"]
features = model_data["features"]
clipping_bounds = model_data.get("clipping_bounds", {})

######################################################
# Feature Engineering (Aligned with 4 Pillars)
######################################################

# Missing values imputation
for col in df.columns:
    if col not in ['region_display', 'country_display']:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Exact age calculation (accounting for day/month/year)
now = pd.Timestamp.now()
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce', format='mixed', dayfirst=False)
df['Age'] = ((now - df['date_of_birth']).dt.days // 365.25).astype('Int64')
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)

# 4 Pillars Encoding
# Pillar 1 (Demographics): Age + is_company
df['is_company'] = (df['client_type'].astype(str).str.strip().str.lower() == 'company').astype(int)

# Pillar 2 (Investment Purpose): is_investment
df['is_investment'] = (df['acquisition_purpose'].astype(str).str.strip().str.lower() == 'investment').astype(int)

# Pillar 3 (Financing Behavior): loan_applied
df['loan_applied'] = (df['loan_applied'].astype(str).str.strip().str.lower() == 'yes').astype(int)

# Pillar 4 (Geographic Patterns): is_international (Non-USA)
df['is_international'] = (df['country'].astype(str).str.strip().str.lower() != 'usa').astype(int)

# Feature extraction
X = df[features].copy()

# Outlier clipping (matching training pipeline bounds)
for col, bounds in clipping_bounds.items():
    if col in X.columns:
        X[col] = X[col].clip(lower=bounds[0], upper=bounds[1])

# Scale and project to PCA orthogonal space
scaled_data = scaler.transform(X)
pca_data = pca.transform(scaled_data)

# Predict clusters
df["Cluster"] = model.predict(pca_data)

# Map clusters to descriptive, data-backed persona names
persona_map = {
    0: "Cluster 0: Corporate & Commercial Real Estate Buyers",
    1: "Cluster 1: Primary Residential Homebuyers",
    2: "Cluster 2: Luxury & High-Spend Property Acquirers",
    3: "Cluster 3: Private Real Estate Investors"
}
df["Persona"] = df["Cluster"].map(persona_map)

######################################################
# SIDEBAR
######################################################

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Buyer Segmentation Overview",
        "Investor Behavior Dashboard",
        "Geographic Buyer Analysis",
        "Segment Insights Panel"
    ]
)

######################################################
# MODULE 1: Overview
######################################################

if page == "Buyer Segmentation Overview":
    st.header("Customer Persona Distribution (4-Pillar Alignment)")
    cluster_count = df['Persona'].value_counts()
    fig = px.pie(
        values=cluster_count.values,
        names=cluster_count.index,
        title="Buyer Segmentation Breakdown (N=2,000 Unique Customers)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)

######################################################
# MODULE 2: Behavior
######################################################

elif page == "Investor Behavior Dashboard":
    st.header("Customer Investment & Property Dynamics")
    fig = px.scatter(
        df,
        x='sale_price',
        y='total_spend',
        color='Persona',
        size='floor_area_sqft',
        hover_data=['Age', 'satisfaction_score', 'properties_bought', 'country_display'],
        labels={
            'sale_price': 'Average Property Price ($)',
            'total_spend': 'Total Portfolio Spend ($)',
            'floor_area_sqft': 'Avg Floor Area (sqft)'
        },
        title="Average Property Price vs. Total Spend by Customer Segment",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)

######################################################
# MODULE 3: Geographic Analysis
######################################################

elif page == "Geographic Buyer Analysis":
    st.header("Regional & International Customer Distribution")
    geo_choice = st.radio("Group By:", ["Domestic Regions (USA States)", "Country of Origin"])
    if geo_choice == "Domestic Regions (USA States)":
        us_df = df[df['country_display'].str.lower() == 'usa']
        top_regions = us_df['region_display'].value_counts().head(10).index
        filtered = us_df[us_df['region_display'].isin(top_regions)]
        ct = pd.crosstab(filtered['region_display'], filtered['Persona'])
        fig = px.bar(
            ct,
            barmode='group',
            labels={'value': 'Customer Count', 'region_display': 'US State / Region'},
            title="Top 10 US Regions across Customer Personas",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
    else:
        top_countries = df['country_display'].value_counts().head(8).index
        filtered = df[df['country_display'].isin(top_countries)]
        ct = pd.crosstab(filtered['country_display'], filtered['Persona'])
        fig = px.bar(
            ct,
            barmode='group',
            labels={'value': 'Customer Count', 'country_display': 'Country of Origin'},
            title="Top Countries across Customer Personas",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
    st.plotly_chart(fig, use_container_width=True)

######################################################
# MODULE 4: Insights Panel
######################################################

else:
    st.header("Customer Segment Statistical Summary")
    summary = (
        df.groupby("Persona")[
            ['Age', 'sale_price', 'total_spend', 'floor_area_sqft', 'is_company', 'is_investment', 'loan_applied', 'is_international', 'properties_bought']
        ]
        .mean()
        .reset_index()
    )
    st.dataframe(summary.style.format({
        'Age': '{:.1f}',
        'sale_price': '${:,.2f}',
        'total_spend': '${:,.2f}',
        'floor_area_sqft': '{:,.1f} sqft',
        'is_company': '{:.1%}',
        'is_investment': '{:.1%}',
        'loan_applied': '{:.1%}',
        'is_international': '{:.1%}',
        'properties_bought': '{:.2f}'
    }))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Buyer Segmentation Dashboard",
    layout="wide"
)

st.title(
"🏠 Buyer Segmentation Dashboard"
)

######################################################
# Load Data
######################################################

clients=pd.read_csv(
"clients.csv"
)

properties=pd.read_csv(
"properties.csv"
)

df=pd.merge(

clients,
properties,

left_on='client_id',
right_on='client_ref'

)

######################################################
# Load Model
######################################################

with open(
"model.pkl",
"rb"
) as file:

    model_data=pickle.load(
    file
)

model=model_data["model"]

scaler=model_data["scaler"]

features=model_data["features"]

######################################################
# Basic Cleaning (Replicate notebook preprocessing)
######################################################

# Handling missing values (object columns with mode, numeric with median)
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Remove duplicate entries
df.drop_duplicates(inplace=True)

# Normalize labels (strip and lowercase object columns)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Robustly convert 'date_of_birth' to datetime, handling mixed formats
df['date_of_birth']=pd.to_datetime(
df['date_of_birth'], errors='coerce', format='mixed', dayfirst=False
)

df['Age']=(
pd.Timestamp.now().year
-
df['date_of_birth'].dt.year
)

# Fill NaN values in 'Age' column using its median (for unparseable dates)
if df['Age'].isnull().any():
    df['Age'] = df['Age'].fillna(df['Age'].median())

# Label Encoding for binary columns
le = LabelEncoder()
binary_cols=[
    'gender',
    'loan_applied'
]

for col in binary_cols:
    df[col]=le.fit_transform(df[col])

# One-Hot Encoding for other categorical columns
categorical_cols=[
'client_type',
'country',
'region',
'acquisition_purpose',
'referral_channel',
'unit_category',
'listing_status'
]

df=pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

# Clean 'sale_price' column by removing '$' and ',' and converting to float
df['sale_price'] = df['sale_price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

X=df[features]

scaled_data=scaler.transform(
X
)

clusters=model.predict(
scaled_data
)

df["Cluster"]=clusters

######################################################
# SIDEBAR
######################################################

st.sidebar.title(
"Navigation"
)

page=st.sidebar.radio(

"Select Module",

[
"Buyer Segmentation Overview",
"Investor Behavior Dashboard",
"Geographic Buyer Analysis",
"Segment Insights Panel"
]

)

######################################################
# MODULE 1
######################################################

if page=="Buyer Segmentation Overview":

    st.header(
    "Cluster Distribution"
    )

    cluster_count=(
    df['Cluster']
    .value_counts()
    )

    fig=px.pie(

    values=cluster_count.values,
    names=cluster_count.index,

    title="Cluster Distribution"

    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )

######################################################
# MODULE 2
######################################################

elif page=="Investor Behavior Dashboard":

    st.header(
    "Investment Patterns"
    )

    fig=px.scatter(

    df,

    x='sale_price',

    y='satisfaction_score',

    color='Cluster',

    size='floor_area_sqft',

    hover_data=['Age']

    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )

######################################################
# MODULE 3
######################################################

elif page=="Geographic Buyer Analysis":

    st.header(
    "Regional Buyer Distribution"
    )

    region_cluster=pd.crosstab(

    df['region'],
    df['Cluster']

    )

    fig=px.bar(

    region_cluster,

    barmode='group'

    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )

######################################################
# MODULE 4
######################################################

else:

    st.header(
    "Cluster Statistics"
    )

    summary=(

    df.groupby(
    "Cluster"
    )

    [

    ['Age',
    'sale_price',
    'satisfaction_score']

    ]

    .mean()

    )

    st.dataframe(
    summary
    )

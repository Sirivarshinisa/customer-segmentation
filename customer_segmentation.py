import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation using RFM & Clustering")

# ---- DATA LOADING ----
@st.cache_data
def load_data():
    # Unzip and load data
    if not os.path.exists("data.csv"):
        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall()
    df = pd.read_csv("data.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

df = load_data()

# ---- RFM CALCULATION ----
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# ---- SCALING ----
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# ---- CLUSTERING ----
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# ---- PCA FOR VISUALIZATION ----
pca = PCA(n_components=2)
components = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = components[:, 0]
rfm['PCA2'] = components[:, 1]

# ---- VISUALIZATIONS ----
st.subheader("📍 PCA Cluster Visualization")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', ax=ax1)
st.pyplot(fig1)

st.subheader("📊 Cluster-wise RFM Summary")
st.write(rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2))

st.subheader("🔍 Explore Individual Clusters")
cluster_num = st.selectbox("Select Cluster", sorted(rfm['Cluster'].unique()))
filtered = rfm[rfm['Cluster'] == cluster_num]

fig2 = sns.pairplot(filtered[['Recency', 'Frequency', 'Monetary']])
st.pyplot(fig2)

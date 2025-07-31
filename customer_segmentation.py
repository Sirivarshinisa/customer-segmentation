import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Sidebar
st.sidebar.title("📊 Customer Segmentation App")
st.sidebar.markdown("""
This app segments customers based on:
- **Recency** (Days since last purchase)  
- **Frequency** (Number of purchases)  
- **Monetary** (Total spent)

Upload your data or use sample below.  
Built using K-Means Clustering.
""")

# Upload
uploaded_file = st.sidebar.file_uploader("Upload your data.csv", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data instead", value=True)

# Load Data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

if uploaded_file and not use_sample:
    df = load_data(uploaded_file)
else:
    df = load_data("data.csv")

# RFM
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Normalize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Optional: Label clusters
cluster_names = {
    0: "At Risk",
    1: "Loyal Customers",
    2: "New Customers",
    3: "Big Spenders"
}
rfm['Segment'] = rfm['Cluster'].map(cluster_names)

# Summary
st.title("🎯 Customer Segmentation using K-Means")
st.subheader("📈 Cluster Summary Statistics")
st.write(rfm.groupby(['Cluster', 'Segment'])[['Recency', 'Frequency', 'Monetary']].mean().round(1))

# Insight
top_cluster = rfm.groupby('Cluster')['Monetary'].mean().idxmax()
st.success(f"💡 Cluster {top_cluster} (**{cluster_names[top_cluster]}**) has the highest average monetary value!")

# Visualization
st.subheader("📉 Cluster Visualization (Recency vs Monetary)")
fig, ax = plt.subplots()
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='Set1', ax=ax)
st.pyplot(fig)

# Download segmented data
st.subheader("⬇️ Download Segmented Customer Data")
csv = rfm.to_csv(index=False)
st.download_button("Download CSV", data=csv, file_name="segmented_customers.csv", mime="text/csv")

# Pairplot filter
st.subheader("🔍 Pairplot of Selected Segment")
segment_filter = st.selectbox("Choose a Segment", rfm['Segment'].unique())
filtered = rfm[rfm['Segment'] == segment_filter]
st.write(filtered.describe())
st.pyplot(sns.pairplot(filtered[['Recency', 'Frequency', 'Monetary']]))

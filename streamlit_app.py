import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Borneo Express Analytics", layout="wide")

# --- Fungsi Generator Data ---
@st.cache_data
def generate_borneo_data():
    np.random.seed(42)
    n = 10000
    data = {
        'Hari_dalam_Minggu': np.random.randint(1, 8, n),
        'Jenis_Layanan': np.random.choice([0, 1, 2], n), # 0:Reguler, 1:Kilat, 2:SameDay
        'Berat_Paket_KG': np.random.uniform(0.5, 20.0, n),
        'Jarak_KM': np.random.uniform(5, 50, n),
        'Volume_Paket': np.random.randint(50, 500, n)
    }
    return pd.DataFrame(data)

# --- Load Data & Model ---
df = generate_borneo_data()
X = df.drop('Volume_Paket', axis=1)
y = df['Volume_Paket']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
dt = DecisionTreeRegressor().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

# --- Sidebar ---
st.sidebar.title("🛠️ Kontrol Parameter")
st.sidebar.markdown("### Pengaturan Model")
n_estimators = st.sidebar.slider("Jumlah Pohon (Random Forest)", 50, 200, 100)

# --- Tampilan Utama ---
st.title("📦 Prediksi Volume Pengiriman Paket")
st.markdown("### Borneo Express Pontianak")

# Metrik Utama
col1, col2, col3 = st.columns(3)
col1.metric("Total Data", "10.000 Transaksi")
col2.metric("Rata-rata Volume", f"{df['Volume_Paket'].mean():.1f} Paket/Hari")
col3.metric("Akurasi Model (R2)", f"{rf.score(X_test, y_test)*100:.1f}%")

st.markdown("---")

# Layout Dashboard
tab1, tab2 = st.tabs(["📊 Visualisasi Data", "🤖 Perbandingan Algoritma"])

with tab1:
    st.subheader("Distribusi Volume Pengiriman")
    fig = px.histogram(df, x="Volume_Paket", nbins=50, color_discrete_sequence=['#2E8B57'])
    st.plotly_chart(fig, use_container_width=True)
    

with tab2:
    st.subheader("Evaluasi Performa (MAE)")
    pred_dt = dt.predict(X_test)
    pred_rf = rf.predict(X_test)
    
    comp_df = pd.DataFrame({
        "Algoritma": ["Decision Tree", "Random Forest"],
        "Error (MAE)": [mean_absolute_error(y_test, pred_dt), mean_absolute_error(y_test, pred_rf)]
    })
    
    fig_comp = px.bar(comp_df, x="Algoritma", y="Error (MAE)", color="Algoritma", text_auto='.2f')
    st.plotly_chart(fig_comp, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Prediksi Paket Terkirim", layout="wide")

st.title("Prediksi Volume Paket Terkirim")
st.subheader("Perbandingan Random Forest Regressor vs Linear Regression")

# ===============================
# LOAD DATA (ANTI ERROR)
# ===============================
@st.cache_data
def load_data():
    file_path = "data_historis_paket.csv"

    # cek file lokal
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='cp1252')
            except:
                df = pd.read_csv(file_path, encoding='latin-1')
    else:
        return None

    # bersihkan kolom
    df.columns = df.columns.str.lower().str.strip()

    return df

df = load_data()

# ===============================
# UPLOAD FILE (BACKUP)
# ===============================
if df is None:
    st.warning("File tidak ditemukan. Upload dataset CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df = pd.read_csv(uploaded_file, encoding='latin-1')

        df.columns = df.columns.str.lower().str.strip()
    else:
        st.stop()

# ===============================
# DEBUG KOLOM
# ===============================
st.write("Kolom dataset:", df.columns)

# ===============================
# DETEKSI KOLOM TANGGAL
# ===============================
possible_date_cols = ['created_at', 'tanggal', 'date', 'tgl', 'waktu']

date_col = None
for col in possible_date_cols:
    if col in df.columns:
        date_col = col
        break

if date_col is None:
    st.error("Kolom tanggal tidak ditemukan!")
    st.stop()

st.success(f"Menggunakan kolom waktu: {date_col}")

# ===============================
# PREPROCESSING
# ===============================
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])

# ===============================
# AGREGASI HARIAN
# ===============================
df['tanggal'] = df[date_col].dt.date

df_harian = df.groupby('tanggal').size().reset_index(name='jumlah_paket')

df_harian['tanggal'] = pd.to_datetime(df_harian['tanggal'])

# ===============================
# FEATURE ENGINEERING
# ===============================
df_harian['hari'] = df_harian['tanggal'].dt.day
df_harian['bulan'] = df_harian['tanggal'].dt.month
df_harian['tahun'] = df_harian['tanggal'].dt.year
df_harian['hari_dalam_minggu'] = df_harian['tanggal'].dt.weekday()

df_harian = df_harian.sort_values('tanggal')

# ===============================
# SPLIT DATA
# ===============================
train_size = int(len(df_harian) * 0.8)

train = df_harian[:train_size]
test = df_harian[train_size:]

X_train = train[['hari','bulan','tahun','hari_dalam_minggu']]
y_train = train['jumlah_paket']

X_test = test[['hari','bulan','tahun','hari_dalam_minggu']]
y_test = test['jumlah_paket']

# ===============================
# MODEL
# ===============================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
lr_model = LinearRegression()

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

pred_rf = rf_model.predict(X_test)
pred_lr = lr_model.predict(X_test)

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("Input Prediksi")

tanggal_input = st.sidebar.date_input("Pilih Tanggal")

hari = tanggal_input.day
bulan = tanggal_input.month
tahun = tanggal_input.year
hari_dalam_minggu = tanggal_input.weekday()

input_data = np.array([[hari, bulan, tahun, hari_dalam_minggu]])

# ===============================
# PREDIKSI
# ===============================
pred_rf_input = rf_model.predict(input_data)[0]
pred_lr_input = lr_model.predict(input_data)[0]

# ===============================
# TAB
# ===============================
tab1, tab2, tab3 = st.tabs(["Prediksi", "Dataset", "Evaluasi"])

# ===============================
# TAB 1
# ===============================
with tab1:
    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Random Forest", f"{int(pred_rf_input)} Paket")

    with col2:
        st.metric("Linear Regression", f"{int(pred_lr_input)} Paket")

# ===============================
# TAB 2
# ===============================
with tab2:
    st.subheader("Data Agregasi Harian")
    st.dataframe(df_harian)

    fig = px.line(df_harian, x="tanggal", y="jumlah_paket", title="Trend Paket")
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# TAB 3
# ===============================
with tab3:
    st.subheader("Evaluasi Model")

    mae_rf = mean_absolute_error(y_test, pred_rf)
    mae_lr = mean_absolute_error(y_test, pred_lr)

    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))

    r2_rf = r2_score(y_test, pred_rf)
    r2_lr = r2_score(y_test, pred_lr)

    eval_df = pd.DataFrame({
        "Model": ["Random Forest", "Linear Regression"],
        "MAE": [mae_rf, mae_lr],
        "RMSE": [rmse_rf, rmse_lr],
        "R2": [r2_rf, r2_lr]
    })

    st.dataframe(eval_df)

    fig = px.bar(eval_df, x="Model", y="MAE", color="Model", title="Perbandingan MAE")
    st.plotly_chart(fig, use_container_width=True)
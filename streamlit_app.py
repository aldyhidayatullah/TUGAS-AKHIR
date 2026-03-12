import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# KONFIGURASI HALAMAN
# ===============================

st.set_page_config(
    page_title="Prediksi Volume Pengiriman Paket",
    layout="wide"
)

st.title("Perbandingan Prediksi Volume Pengiriman Paket")
st.subheader("Random Forest vs Decision Tree - Borneo Express Pontianak")

# ===============================
# LOAD DATASET
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_pengiriman.csv")
    return df

df = load_data()

# ===============================
# PREPROCESSING
# ===============================

le = LabelEncoder()

df["Jenis_Layanan"] = le.fit_transform(df["Jenis_Layanan"])
df["Tujuan_Pulau"] = le.fit_transform(df["Tujuan_Pulau"])

X = df.drop("Volume_Paket", axis=1)
y = df["Volume_Paket"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# MODEL MACHINE LEARNING
# ===============================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

dt_model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

pred_rf = rf_model.predict(X_test)
pred_dt = dt_model.predict(X_test)

# ===============================
# SIDEBAR INPUT DATA
# ===============================

st.sidebar.header("Input Data Pengiriman")

tahun = st.sidebar.slider("Tahun", 2022, 2030, 2026)

hari = st.sidebar.slider("Hari dalam Minggu", 1, 7, 3)

berat = st.sidebar.slider("Berat Paket (KG)", 0.5, 30.0, 5.0)

jarak = st.sidebar.slider("Jarak Pengiriman (KM)", 1, 100, 20)

layanan = st.sidebar.selectbox(
    "Jenis Layanan",
    ["Reguler", "Kilat", "SameDay"]
)

pulau = st.sidebar.selectbox(
    "Tujuan Pulau",
    ["Kalimantan", "Jawa", "Sumatera", "Sulawesi", "Papua"]
)

# encoding input
layanan_encode = {"Reguler":0,"Kilat":1,"SameDay":2}
pulau_encode = {
    "Kalimantan":0,
    "Jawa":1,
    "Sumatera":2,
    "Sulawesi":3,
    "Papua":4
}

input_data = np.array([[

    tahun,
    hari,
    berat,
    jarak,
    layanan_encode[layanan],
    pulau_encode[pulau]

]])

# ===============================
# PREDIKSI
# ===============================

pred_rf_input = rf_model.predict(input_data)[0]
pred_dt_input = dt_model.predict(input_data)[0]

# ===============================
# TABS DASHBOARD
# ===============================

tab1,tab2,tab3 = st.tabs(["Prediksi","Dataset","Evaluasi Model"])

# ===============================
# TAB PREDIKSI
# ===============================

with tab1:

    st.subheader("Hasil Prediksi Volume Pengiriman")

    col1,col2 = st.columns(2)

    with col1:
        st.metric(
            "Random Forest",
            f"{pred_rf_input:.0f} Paket"
        )

    with col2:
        st.metric(
            "Decision Tree",
            f"{pred_dt_input:.0f} Paket"
        )

    st.subheader("Feature Importance")

    features = X.columns

    col3,col4 = st.columns(2)

    with col3:

        st.write("Random Forest")

        importance = rf_model.feature_importances_

        fig,ax = plt.subplots()

        ax.bar(features,importance)

        plt.xticks(rotation=45)

        st.pyplot(fig)

    with col4:

        st.write("Decision Tree")

        importance = dt_model.feature_importances_

        fig,ax = plt.subplots()

        ax.bar(features,importance)

        plt.xticks(rotation=45)

        st.pyplot(fig)

# ===============================
# TAB DATASET
# ===============================

with tab2:

    st.subheader("Dataset Pengiriman")

    st.dataframe(df)

    fig = px.histogram(
        df,
        x="Volume_Paket",
        nbins=40,
        title="Distribusi Volume Pengiriman"
    )

    st.plotly_chart(fig,use_container_width=True)

# ===============================
# TAB EVALUASI MODEL
# ===============================

with tab3:

    st.subheader("Evaluasi Model")

    mae_rf = mean_absolute_error(y_test,pred_rf)
    mae_dt = mean_absolute_error(y_test,pred_dt)

    mse_rf = mean_squared_error(y_test,pred_rf)
    mse_dt = mean_squared_error(y_test,pred_dt)

    r2_rf = r2_score(y_test,pred_rf)
    r2_dt = r2_score(y_test,pred_dt)

    eval_df = pd.DataFrame({

        "Model":["Random Forest","Decision Tree"],
        "MAE":[mae_rf,mae_dt],
        "MSE":[mse_rf,mse_dt],
        "R2 Score":[r2_rf,r2_dt]

    })

    st.dataframe(eval_df)

    fig = px.bar(
        eval_df,
        x="Model",
        y="R2 Score",
        color="Model",
        title="Perbandingan Performa Model"
    )

    st.plotly_chart(fig,use_container_width=True)   
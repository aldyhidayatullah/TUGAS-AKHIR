import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Borneo Express Analytics",
    layout="wide"
)

st.title("📦 Prediksi Volume Pengiriman Paket")
st.subheader("Borneo Express Pontianak")


# =========================
# GENERATE DATASET SIMULASI
# =========================

@st.cache_data
def generate_borneo_data():

    np.random.seed(42)
    n = 5000

    data = {

        "Hari_dalam_Minggu": np.random.randint(1,8,n),

        "Jenis_Layanan": np.random.choice(
            ["Reguler","Kilat","SameDay"], n
        ),

        "Berat_Paket_KG": np.round(np.random.uniform(0.5,20,n),2),

        "Jarak_KM": np.round(np.random.uniform(5,80,n),2),

        "Volume_Paket": np.random.randint(50,500,n)

    }

    df = pd.DataFrame(data)

    return df


df = generate_borneo_data()


# =========================
# PREPROCESSING DATA
# =========================

df["Jenis_Layanan"] = df["Jenis_Layanan"].map({
    "Reguler":0,
    "Kilat":1,
    "SameDay":2
})

X = df.drop("Volume_Paket", axis=1)
y = df["Volume_Paket"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


# =========================
# SIDEBAR PARAMETER MODEL
# =========================

st.sidebar.title("⚙️ Parameter Model")

n_estimators = st.sidebar.slider(
    "Jumlah Pohon Random Forest",
    50,
    300,
    100
)

max_depth = st.sidebar.slider(
    "Kedalaman Decision Tree",
    3,
    20,
    10
)


# =========================
# TRAIN MODEL
# =========================

# Decision Tree
dt_model = DecisionTreeRegressor(
    max_depth=max_depth,
    random_state=42
)

dt_model.fit(X_train,y_train)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=42
)

rf_model.fit(X_train,y_train)


# =========================
# PREDIKSI
# =========================

pred_dt = dt_model.predict(X_test)
pred_rf = rf_model.predict(X_test)


# =========================
# EVALUASI MODEL
# =========================

mae_dt = mean_absolute_error(y_test,pred_dt)
mae_rf = mean_absolute_error(y_test,pred_rf)

r2_dt = r2_score(y_test,pred_dt)
r2_rf = r2_score(y_test,pred_rf)


# =========================
# METRIK UTAMA
# =========================

col1,col2,col3 = st.columns(3)

col1.metric("Total Data", len(df))

col2.metric(
    "Rata-rata Volume Paket",
    f"{df['Volume_Paket'].mean():.1f}"
)

col3.metric(
    "Akurasi Random Forest (R²)",
    f"{rf_model.score(X_test,y_test)*100:.1f}%"
)


st.markdown("---")


# =========================
# TABS DASHBOARD
# =========================

tab1, tab2, tab3 = st.tabs([
    "📊 Visualisasi Data",
    "🤖 Perbandingan Algoritma",
    "🔮 Prediksi Manual"
])


# =========================
# VISUALISASI DATA
# =========================

with tab1:

    st.subheader("Distribusi Volume Pengiriman")

    fig = px.histogram(
        df,
        x="Volume_Paket",
        nbins=40,
        color_discrete_sequence=["#2E8B57"]
    )

    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Pengiriman Berdasarkan Jenis Layanan")

    layanan_map = {0:"Reguler",1:"Kilat",2:"SameDay"}

    df_temp = df.copy()
    df_temp["Jenis_Layanan"] = df_temp["Jenis_Layanan"].map(layanan_map)

    fig2 = px.box(
        df_temp,
        x="Jenis_Layanan",
        y="Volume_Paket",
        color="Jenis_Layanan"
    )

    st.plotly_chart(fig2, use_container_width=True)


# =========================
# PERBANDINGAN MODEL
# =========================

with tab2:

    st.subheader("Perbandingan Algoritma")

    comp_df = pd.DataFrame({

        "Algoritma":[
            "Decision Tree",
            "Random Forest"
        ],

        "MAE":[
            mae_dt,
            mae_rf
        ],

        "R2":[
            r2_dt,
            r2_rf
        ]

    })

    st.dataframe(comp_df)

    fig_comp = px.bar(
        comp_df,
        x="Algoritma",
        y="MAE",
        color="Algoritma",
        text_auto=".2f"
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    if mae_rf < mae_dt:

        st.success(
            "Random Forest memiliki performa lebih baik berdasarkan MAE"
        )

    else:

        st.success(
            "Decision Tree memiliki performa lebih baik berdasarkan MAE"
        )


# =========================
# PREDIKSI MANUAL
# =========================

with tab3:

    st.subheader("Input Prediksi Pengiriman")

    col1,col2 = st.columns(2)

    hari = col1.slider("Hari dalam Minggu",1,7)
    layanan = col2.selectbox(
        "Jenis Layanan",
        ["Reguler","Kilat","SameDay"]
    )

    berat = st.slider("Berat Paket (KG)",0.5,20.0,5.0)
    jarak = st.slider("Jarak Pengiriman (KM)",5,80,20)

    layanan_encoded = {
        "Reguler":0,
        "Kilat":1,
        "SameDay":2
    }[layanan]

    input_data = np.array([[hari,layanan_encoded,berat,jarak]])

    if st.button("Prediksi"):

        pred_dt = dt_model.predict(input_data)[0]
        pred_rf = rf_model.predict(input_data)[0]

        st.success(f"Prediksi Decision Tree: {int(pred_dt)} paket")

        st.success(f"Prediksi Random Forest: {int(pred_rf)} paket")
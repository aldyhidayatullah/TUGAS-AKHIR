import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# =========================
# 1. KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Dashboard Prediksi Borneo Express",
    page_icon="📦",
    layout="wide"
)

# =========================
# 2. STYLE CSS (FIX VISUAL)
# =========================
st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    /* Memaksa teks metrik dan label berwarna gelap agar terlihat */
    [data-testid="stMetricValue"] { color: white !important; font-weight: bold !important; }
    [data-testid="stMetricLabel"] { color: white !important; font-weight: 600 !important; }
    
    div[data-testid="metric-container"] {
        background: white; border-radius: 12px; padding: 15px;
        border: 1px solid #eaeaea; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] { font-weight: bold; font-size: 16px; }
    .explanation-box {
        background-color: #ffffff; padding: 20px; border-radius: 10px;
        border-left: 5px solid #1e293b; margin-top: 10px; color: #333;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. LOAD & AGREGASI DATA
# =========================
@st.cache_data
def load_data(file):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(file, sep=';', encoding=enc, on_bad_lines='skip')
            break
        except: continue

    df.columns = df.columns.str.lower().str.strip()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at'])
    df['tanggal'] = df['created_at'].dt.date

    df.columns = df.columns.str.lower().str.strip()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at'])
   
    # Ekstrasi tanggal awal bersih
    df['tanggal'] = df['created_at'].dt.date
    
    # Agreagasi Harian
    df = df.groupby('tanggal').agg({'no_resi': 'count'}).reset_index()
    df.columns =['Tanggal', 'Volume']
    
    # Feature Engineering (Menggunakan Temporary datetime untuk ektraksi angka)
    temp_dt = pd.to_datetime(df['Tanggal'])
    df['Hari_'] = temp_dt.dt.dayofweek
    df['Bulan_'] = temp_dt.dt.month
    df['Tanggal_'] = temp_dt.dt.day
    
    # Ubah kembali kolom tanggal menjadi tipe data agar jam 00:00 Hilang
    df['Tanggal'] = pd.to_datetime(df['Tanggal']).dt.date
    
    return df.sort_values('Tanggal')

# Coba muat file otomatis
file_path = "data_historis_paket.csv"
df_main = None

if os.path.exists(file_path):
    df_main = load_data(file_path)
else:
    upload = st.sidebar.file_uploader("Upload CSV Borneo Express", type="csv")
    if upload:
        df_main = load_data(upload)

# =========================
# 4. MAIN APP
# =========================
if df_main is not None:
    # SIDEBAR
    with st.sidebar:
        st.title("📦 Borneo Express")
        menu = st.radio("Navigasi Menu", ["📊 Dashboard & Dataset", "🤖 Evaluasi Model", "🔮 Prediksi Multi-Skala"])
        st.divider()
        st.caption("Aldy Hidayatullah - Teknik Informatika")

    # Siapkan Data Model
    X = df_main[['Hari_', 'Bulan_', 'Tanggal_']]
    y = df_main['Volume']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training Model (Global)
    rf_final = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    lr_final = LinearRegression().fit(X, y)

    # --- MENU: DASHBOARD ---
    if menu == "📊 Dashboard & Dataset":
        st.title("📊 Dashboard & Analisis Data")
        tab1, tab2 = st.tabs(["📈 Trend & KPI", "📑 Data Agregasi & Korelasi"])
        
        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Observasi", f"{len(df_main)} Hari")
            c2.metric("Rata-rata Paket/Hari", f"{int(df_main['Volume'].mean())}")
            c3.metric("Total Volume", f"{int(df_main['Volume'].sum()):,}")
            
            avg_volume = df_main["Volume"].mean()
            st.subheader("Trend Volume Pengiriman Harian")
            fig_trend = px.line(df_main, x="Tanggal", y="Volume", markers=True, template="plotly_white")
            fig_trend.update_layout(font=dict(color="black"))
            st.plotly_chart(fig_trend, use_container_width=True)

        with tab2:
            st.subheader("📑 Tabel Data Hasil Agregasi")
            st.dataframe(df_main, use_container_width=True)
            st.divider()
            st.subheader("🔗 Korelasi Fitur (Heatmap)")
            corr = df_main[['Volume', 'Hari_', 'Bulan_', 'Tanggal_']].corr()
            fig_corr, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig_corr)

    # --- MENU: EVALUASI ---
    elif menu == "🤖 Evaluasi Model":
        st.title("🤖 Perbandingan Performa Algoritma")

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        model_lr = LinearRegression().fit(X_tr, y_tr)

        p_rf = model_rf.predict(X_te)
        p_lr = model_lr.predict(X_te)

        metrics = {
            "Metrik": ["MAE", "RMSE", "R² Score"],
            "Random Forest": [
                mean_absolute_error(y_te, p_rf),
                np.sqrt(mean_squared_error(y_te, p_rf)),
                r2_score(y_te, p_rf)
            ],
            "Linear Regression": [
                mean_absolute_error(y_te, p_lr),
                np.sqrt(mean_squared_error(y_te, p_lr)),
                r2_score(y_te, p_lr)
            ]
        }
            # 1. Menampilkan tabel (tetap seperti kode Anda)
        st.table(pd.DataFrame(metrics).set_index("Metrik"))
            
        df_plot = pd.DataFrame(metrics).iloc[:2].melt(id_vars="Metrik", var_name="Model", value_name="Nilai")

            # 3. Membuat grafik dengan pemetaan warna khusus
        fig_perf = px.bar(
                df_plot, 
                x="Metrik", 
                y="Nilai", 
                color="Model", 
                barmode="group", 
                text_auto='.2f', 
                title="Model Performance Comparison",
                # Menggunakan warna spesifik Anda
                color_discrete_map={
                    "Random Forest": "#0056B3",
                    "Linear Regression": "#FF6347"
                },
                template="plotly_white" # Memastikan tema latar belakang putih bersih
            )

                # --- UPDATE PADA BAGIAN LAYOUT ---
        fig_perf.update_layout(
            plot_bgcolor='white', 
            font=dict(color="black", size=13), # Font diperbesar sedikit agar lebih tegas
            yaxis_title="Nilai Error",
            xaxis_title="",
            legend_title_text="Algoritma",
            
            # Mempertebal garis sumbu X dan Y (Garis Hitam Utama)
            xaxis=dict(
                showline=True, 
                linewidth=2, 
                linecolor='black', 
                mirror=True,
                showgrid=False
            ),
            yaxis=dict(
                showline=True, 
                linewidth=2, 
                linecolor='black', 
                mirror=True,
                showgrid=True, 
                gridcolor='Gray' # Garis bantu tetap tipis agar tidak mengganggu
            ),
            
            # Mengatur posisi legend agar tidak menutupi grafik
            legend=dict(
                bordercolor="Black",
                borderwidth=1
            )
        )

        # Mempertebal outline pada batang grafik agar warna biru dan merah lebih "keluar"
        fig_perf.update_traces(
            marker_line_color='black',
            marker_line_width=1.5,
            textfont_size=13,
            textfont_color='black',
            textposition="outside", 
            cliponaxis=False
        )

        st.plotly_chart(fig_perf, use_container_width=True)

   # --- MENU: PREDIKSI & FEATURE IMPORTANCE ---
    elif menu == "🔮 Prediksi Multi-Skala":
        st.title("🔮 Prediksi & Feature Importance Analysis")

        # 1. Inisialisasi session state untuk menyimpan hasil prediksi
        if 'hasil_prediksi' not in st.session_state:
            st.session_state.hasil_prediksi = None
            
        if 'metrics_eval' not in st.session_state:
            # Hitung metrik evaluasi (validasi) satu kali saja
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            m_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
            m_lr = LinearRegression().fit(X_tr, y_tr)
            p_rf, p_lr = m_rf.predict(X_te), m_lr.predict(X_te)
            
            st.session_state.metrics_eval = {
                'rf': {'mae': mean_absolute_error(y_te, p_rf), 'rmse': np.sqrt(mean_squared_error(y_te, p_rf)), 'r2': r2_score(y_te, p_rf)},
                'lr': {'mae': mean_absolute_error(y_te, p_lr), 'rmse': np.sqrt(mean_squared_error(y_te, p_lr)), 'r2': r2_score(y_te, p_lr)}
            }    

        col_input, col_results = st.columns([1, 2])
        
        with col_input:
            st.write("### Input Prediksi")
            selected_date = st.date_input("Pilih Tanggal Prediksi", value=df_main['Tanggal'].max())
            predict_btn = st.button("Jalankan Prediksi")

        # 2. Simpan hasil ke dalam session_state saat tombol ditekan
        if predict_btn:
            d = pd.to_datetime(selected_date)
            features = np.array([[d.dayofweek, d.month, d.day]])
            st.session_state.hasil_prediksi = {
                'tgl': selected_date.strftime('%d %B %Y'),
                'res_rf': rf_final.predict(features)[0],
                'res_lr': lr_final.predict(features)[0]
            }
            
          

        # 3. Tampilkan hasil dari session_state (jika ada)
        if st.session_state.hasil_prediksi:
            h = st.session_state.hasil_prediksi
            m = st.session_state.metrics_eval
            
            rf_day = f"{int(h['res_rf']*7):,}".replace(",", ".")
            lr_day = f"{int(h['res_lr']*7):,}".replace(",", ".")
            
            # 1. Hitung dan format angka untuk Mingguan
            rf_week = f"{int(h['res_rf']*7):,}".replace(",", ".")
            lr_week = f"{int(h['res_lr']*7):,}".replace(",", ".")
            
            # 2. Hitung dan format angka untuk Bulanan
            rf_month = f"{int(h['res_rf']*30):,}".replace(",", ".")
            lr_month = f"{int(h['res_lr']*30):,}".replace(",", ".")
            
            with col_results:
                st.subheader(f"📅 Hasil Prediksi: {h['tgl']}")
                p1, p2, p3 = st.columns(3)
                
            with p1:
                st.markdown("### ☀️ Harian")
                st.metric("Random Forest", f"{rf_day} Paket")
                st.metric("Linear Regression", f"{lr_day} Paket")

            with p2:
                st.markdown("### 📅 Mingguan")
                st.metric("Random Forest", f"{rf_week} Paket")
                st.metric("Linear Regression", f"{lr_week} Paket")

            with p3:
                st.markdown("### 🌙 Bulanan")
                st.metric("Random Forest", f"{rf_month} Paket")
                st.metric("Linear Regression", f"{lr_month} Paket")
                
                
            # --- MENAMPILKAN SKOR EVALUASI MODEL ---
            st.write("---")

            # Menggunakan Expander
            with st.expander("📊 Lihat Detail Skor Performa Model", expanded=True):
                # Baris Random Forest
                st.markdown("🌲 **Random Forest Regressor**")
                st.info(f"MAE: **{m['rf']['mae']:.2f}** | RMSE: **{m['rf']['rmse']:.2f}** | R²: **{m['rf']['r2']:.4f}**")
                
                # Baris Linear Regression
                st.markdown("📈 **Linear Regression**")
                st.success(f"MAE: **{m['lr']['mae']:.2f}** | RMSE: **{m['lr']['rmse']:.2f}** | R²: **{m['lr']['r2']:.4f}**")

            st.divider()
            
        show_fi = st.checkbox("🔍 Tampilkan Analisis Feature Importance")
        
        if show_fi:
            # Bagian Feature Importance Anda tetap di sini...
            pass
            
            # Persiapan Data Importance
            rf_imp = rf_final.feature_importances_
            lr_coef = np.abs(lr_final.coef_)
            lr_imp = lr_coef / np.sum(lr_coef) # Normalisasi agar total=1
            
            features_list = ['Hari', 'Bulan', 'Tanggal']

            # --- MEMBUAT KOLOM TERPISAH UNTUK MASING-MASING ALGORITMA ---
            col_rf, col_lr = st.columns(2)

           # --- KOLOM KIRI: RANDOM FOREST ---
            with col_rf:
                st.markdown("### 🌲 Random Forest Regressor")
                
                fi_rf_df = pd.DataFrame({
                    'Fitur': features_list,
                    'Score': rf_imp
                }).sort_values('Score', ascending=False)

                # PERBAIKAN: x='Fitur', y='Score', dan HAPUS orientation='h'
                fig_rf = px.bar(fi_rf_df, 
                                x='Fitur', 
                                y='Score', 
                                title="Feature Importance: Random Forest",
                                labels={'Score': 'Importance Score'},
                                color_discrete_sequence=['#0056B3']) 
                
                # Gunakan xaxis untuk mengurutkan karena sekarang grafik vertikal
                fig_rf.update_layout(plot_bgcolor='white', font=dict(color="black"), xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_rf, use_container_width=True)

                with st.expander("Lihat Penjelasan Teknis (Random Forest)", expanded=True):
                    st.write("**Hasil:**")
                    for _, row in fi_rf_df.iterrows():
                        st.write(f"- **{row['Fitur']}**: {row['Score']:.4f}")
                    st.markdown("**Interpretasi:** Model mengukur seberapa sering fitur digunakan untuk membagi data guna mengurangi varians.")

            # --- KOLOM KANAN: LINEAR REGRESSION ---
            with col_lr:
                st.markdown("### 📈 Linear Regression")
                
                fi_lr_df = pd.DataFrame({
                    'Fitur': features_list,
                    'Score': lr_imp
                }).sort_values('Score', ascending=False)

                # PERBAIKAN: x='Fitur', y='Score', dan HAPUS orientation='h'
                fig_lr = px.bar(fi_lr_df, 
                                x='Fitur', 
                                y='Score', 
                                title="Feature Importance: Linear Regression",
                                labels={'Score': 'Normalized Coefficient Score'},
                                color_discrete_sequence=['#FF6347']) 

                # Gunakan xaxis untuk mengurutkan karena sekarang grafik vertikal
                fig_lr.update_layout(plot_bgcolor='white', font=dict(color="black"), xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_lr, use_container_width=True)

                with st.expander("Lihat Penjelasan Teknis (Linear Regression)", expanded=True):
                    st.write("**Hasil:**")
                    for _, row in fi_lr_df.iterrows():
                        st.write(f"- **{row['Fitur']}**: {row['Score']:.4f}")
                    st.markdown("**Interpretasi:** Skor berasal dari koefisien absolut yang dinormalisasi, menunjukkan hubungan linier terkuat.")

            # --- PERBANDINGAN & IMPLIKASI (TETAP DI BAWAH, FULL WIDTH) ---
            st.markdown("---")
            st.markdown("### 🔍 Analisis Perbandingan & Implikasi Praktis")
            
            c_comp, c_impl = st.columns(2)
            with c_comp:
                st.markdown("**Perbedaan Utama:**")
                st.write("""
                1. **Kompleksitas Pola**: Random Forest mampu menangkap pola **non-linier** (misal: lonjakan paket hanya pada hari Senin), sedangkan Linear Regression hanya melihat tren kenaikan atau penurunan yang bersifat **konstan**.
                2. **Stabilitas**: Random Forest biasanya memberikan distribusi bobot yang lebih merata karena merupakan gabungan dari 100 pohon keputusan, menjadikannya lebih tahan terhadap data pencilan (*outliers*).
                """)
            
            with c_impl:
                st.markdown("**Implikasi Praktis (Manajerial):**")
                st.write("""
                1. **Prioritas Operasional**: Fokuskan alokasi kurir pada fitur waktu (Hari/Tanggal) dengan skor tertinggi. Jika 'Hari' dominan, maka jadwal kerja mingguan harus diperketat.
                2. **Perencanaan Kapasitas**: Variabel dengan skor rendah dapat dianggap sebagai faktor pendukung yang tidak menyebabkan lonjakan paket secara drastis, sehingga tidak memerlukan pengawasan ekstra.
                """)

else:
    st.warning("Silakan muat dataset Borneo Express melalui sidebar untuk memulai.")

st.markdown("<div style='text-align: center; color: gray; margin-top: 50px;'>© 2026 Borneo Express Pontianak - Aldy Hidayatullah</div>", unsafe_allow_html=True)
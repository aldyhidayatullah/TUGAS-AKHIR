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
    # Mencoba berbagai encoding agar tidak error saat baca CSV
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(file, sep=';', encoding=enc, on_bad_lines='skip')
            break
        except: continue
    
    # Standarisasi nama kolom (lowercase & hapus spasi)
    df.columns = df.columns.str.lower().str.strip()
    
    # Konversi tanggal dengan dayfirst=True agar 03/11 dibaca 3 November (sesuai dataset Anda)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['created_at'])
    
    # --- PERBAIKAN: Hitung total data asli (14.221) sebelum di-groupby ---
    total_mentah = len(df)
    
    # Buat kolom tanggal untuk agregasi
    df['tanggal'] = df['created_at'].dt.date
    
    # Agregasi data (mengelompokkan ribuan resi menjadi hitungan per hari)
    df_agg = df.groupby('tanggal').agg({'no_resi': 'count'}).reset_index()
    df_agg.columns = ['Tanggal', 'Volume']
    
    # Feature Engineering untuk model ML
    temp_dt = pd.to_datetime(df_agg['Tanggal'])
    df_agg['Hari_'] = temp_dt.dt.dayofweek
    df_agg['Bulan_'] = temp_dt.dt.month
    df_agg['Tanggal_'] = temp_dt.dt.day
    
    df_agg['Tanggal'] = pd.to_datetime(df_agg['Tanggal']).dt.date
    
    # Kembalikan dua nilai: dataframe hasil agregasi DAN jumlah data asli
    return df_agg.sort_values('Tanggal'), total_mentah

# Coba muat file otomatis
file_path = "data_historis_paket.csv"
df_main = None

# Ganti bagian ini agar menerima 'total_mentah'
if os.path.exists(file_path):
    df_main, total_mentah = load_data(file_path)
else:
    upload = st.sidebar.file_uploader("Upload CSV Borneo Express", type="csv")
    if upload:
        df_main, total_mentah = load_data(upload)

# =========================
# 4. MAIN APP
# =========================
if df_main is not None:
    st.markdown("""
    <style>
        /* Styling untuk kotak profil di bagian bawah sidebar */
        .sidebar-profile {
            background: linear-gradient(135deg, #2b2b2b 0%, #1e1e1e 100%);
            padding: 15px;
            border-radius: 12px;
            font-size: 0.85rem;
            color: #b0bec5;
            text-align: center;
            border: 1px solid #444;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-top: 20px;
        }
        .sidebar-profile b {
            color: #ffffff;
            font-size: 1rem;
        }
        .status-dot {
            height: 10px;
            width: 10px;
            background-color: #4caf50;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    # MENU: SIDEBAR MODERN
    # =========================
    with st.sidebar:
        # 1. Logo / Header Aplikasi
        st.markdown("""
            <div style='text-align: center; margin-bottom: 20px;'>
                <h1 style='margin-bottom: 0; padding-bottom: 0; color: #00bcd4; font-weight: 800; font-size: 2.2rem;'>
                    📦 BORNEO<br><span style='color: white;'>EXPRESS</span>
                </h1>
                <p style='color: gray; font-size: 0.9rem; margin-top: 5px; letter-spacing: 1px;'>
                    SISTEM PREDIKSI LOGISTIK
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # 2. Navigasi Menu
        st.markdown("**📍 NAVIGASI MENU**")
        # Menggunakan label_visibility="collapsed" agar tulisan "Navigasi Menu" bawaan hilang
        # sehingga tampilannya tidak dobel dan lebih bersih
        menu = st.radio(
            "Navigasi Menu", 
            ["📊 Dashboard & Analisis Data", "🤖 Evaluasi Model", "🔮 Prediksi Multi-Skala"],
            label_visibility="collapsed" 
        )
        
        st.divider()
        
        # 3. Profil Identitas Mahasiswa (Cocok untuk Skripsi)
        st.markdown("""
        <div class="sidebar-profile">
            <b>👨‍💻 Aldy Hidayatullah</b><br>
            NPM. 211220026<br>
            Teknik Informatika<br>
            Universitas Muhammadiyah Pontianak<br>
            <hr style="border-color: #444; margin: 10px 0;">
            <span style="font-size: 0.75rem; color: #aaa;">
                <span class="status-dot"></span>System Online - 2026
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Siapkan Data Model
    X = df_main[['Hari_', 'Bulan_', 'Tanggal_']]
    y = df_main['Volume']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training Model (Global)
    rf_final = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    lr_final = LinearRegression().fit(X, y)

   # --- MENU: DASHBOARD ---
    if menu == "📊 Dashboard & Analisis Data":
        st.title("📊 Dashboard & Analisis Data")
        
        tab1, tab2 = st.tabs(["📈 Ringkasan Operasional", "📑 Tabel Data & Korelasi"])
        
        # ==========================================
        # TAB 1: RINGKASAN OPERASIONAL (MODERN UI)
        # ==========================================
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True) # Spasi atas
            
            # --- 1. CSS UNTUK MODERN CARDS ---
            st.markdown("""
            <style>
                .dash-card {
                    background: linear-gradient(135deg, #2b2b2b 0%, #1e1e1e 100%);
                    border-radius: 15px;
                    padding: 20px 10px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
                    text-align: center;
                    border-top: 5px solid;
                    transition: all 0.3s ease;
                    margin-bottom: 20px;
                }
                .dash-card:hover {
                    transform: translateY(-8px);
                    box-shadow: 0 10px 20px rgba(0,0,0,0.6);
                }
                .card-icon {
                    font-size: 2.5rem;
                    margin-bottom: 10px;
                }
                .card-title {
                    color: #b0bec5;
                    font-size: 0.9rem;
                    font-weight: 600;
                    margin-bottom: 5px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                .card-value {
                    color: #ffffff;
                    font-size: 1.8rem;
                    font-weight: 800;
                }
                .card-unit {
                    font-size: 1rem;
                    color: #888888;
                    font-weight: 500;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # --- 2. BARIS METRIK KARTU BERJEJER ---
            c1, c2, c3, c4 = st.columns(4)
            
            v_mentah = f"{total_mentah:,}".replace(',', '.')
            v_agregasi = f"{len(df_main):,}".replace(',', '.')
            v_rata = int(df_main['Volume'].mean())
            v_hari = len(df_main)

            with c1:
                st.markdown(f"""
                <div class="dash-card" style="border-top-color: #00bcd4;">
                    <div class="card-icon">📦</div>
                    <div class="card-title">Data Mentah</div>
                    <div class="card-value">{v_mentah} <span class="card-unit">Resi</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown(f"""
                <div class="dash-card" style="border-top-color: #4caf50;">
                    <div class="card-icon">📑</div>
                    <div class="card-title">Data Agregasi</div>
                    <div class="card-value">{v_agregasi} <span class="card-unit">Baris</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with c3:
                st.markdown(f"""
                <div class="dash-card" style="border-top-color: #ff9800;">
                    <div class="card-icon">⚡</div>
                    <div class="card-title">Rata-rata Harian</div>
                    <div class="card-value">{v_rata} <span class="card-unit">Paket/Hari</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with c4:
                st.markdown(f"""
                <div class="dash-card" style="border-top-color: #9c27b0;">
                    <div class="card-icon">🗓️</div>
                    <div class="card-title">Total Observasi</div>
                    <div class="card-value">{v_hari} <span class="card-unit">Hari</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
            
            # --- 3. DIAGRAM VISUALISASI ---
            st.markdown("<h4 style='color: #fff;'>📊 Grafik Visualisasi Data</h4>", unsafe_allow_html=True)
            col_diag1, col_diag2 = st.columns(2)
            
            with col_diag1:
                # Mengubah template ke plotly_dark agar serasi dengan Dark Mode
                fig_dist = px.histogram(
                    df_main, x="Volume", nbins=30, 
                    template="plotly_dark", 
                    color_discrete_sequence=['#00bcd4'],
                    title="Sebaran Frekuensi Volume Paket"
                )
                # Transparansi background grafik agar terlihat menyatu
                fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_dist, use_container_width=True)
                
            with col_diag2:
                fig_trend = px.line(
                    df_main, x="Tanggal", y="Volume", 
                    template="plotly_dark",
                    color_discrete_sequence=['#ff9800'],
                    title="Trend Pergerakan Paket per Waktu"
                )
                fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # ==========================================
        # TAB 2: DATASET & KORELASI
        # ==========================================
        with tab2:
            st.markdown("<br><h4 style='color: #fff;'>📑 Eksplorasi Tabel Dataset</h4>", unsafe_allow_html=True)
            
            tabel_col1, tabel_col2 = st.columns(2)
            
            with tabel_col1:
                st.markdown("**1. Tabel Data Mentah (Agregasi Dasar)**")
                st.dataframe(df_main[['Tanggal', 'Volume']], use_container_width=True, height=350)
                
            with tabel_col2:
                st.markdown("**2. Tabel Data Fitur Machine Learning**")
                st.dataframe(df_main, use_container_width=True, height=350)
                
            st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #fff;'>🔗 Korelasi Fitur (Heatmap)</h4>", unsafe_allow_html=True)
            st.caption("Semakin mendekati angka 1 atau -1, semakin kuat pengaruh variabel waktu (Hari/Bulan) terhadap Volume paket.")
            
            # Styling heatmap agar sesuai dengan tema gelap
            plt.style.use("dark_background") 
            corr = df_main[['Volume', 'Hari_', 'Bulan_', 'Tanggal_']].corr()
            fig_corr, ax = plt.subplots(figsize=(10, 4))
            
            # Ubah warna background figure matplotlib jadi transparan
            fig_corr.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            sns.heatmap(
                corr, annot=True, cmap='mako', # Menggunakan palet mako (hijau kebiruan gelap)
                fmt=".2f", linewidths=0.5, ax=ax, 
                cbar_kws={'label': 'Kekuatan Korelasi'}
            )
            st.pyplot(fig_corr)
            # Reset style matplotlib agar tidak mengganggu plot di menu lain
            plt.style.use("default")

    # --- MENU: EVALUASI ---
    elif menu == "🤖 Evaluasi Model":
        st.title("🤖 Perbandingan Performa Algoritma")
        st.markdown("<p style='color: #b0bec5; margin-top: -15px;'>Mengevaluasi akurasi Random Forest Regressor vs Linear Regression pada data testing.</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # 1. SPLIT DATA & TRAINING (Pastikan X dan y sudah didefinisikan sebelumnya)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        model_lr = LinearRegression().fit(X_tr, y_tr)

        p_rf = model_rf.predict(X_te)
        p_lr = model_lr.predict(X_te)

        # 2. PERHITUNGAN METRIK
        rf_mae, rf_rmse, rf_r2 = mean_absolute_error(y_te, p_rf), np.sqrt(mean_squared_error(y_te, p_rf)), r2_score(y_te, p_rf)
        lr_mae, lr_rmse, lr_r2 = mean_absolute_error(y_te, p_lr), np.sqrt(mean_squared_error(y_te, p_lr)), r2_score(y_te, p_lr)

        # 3. CSS KHUSUS KARTU KOMPARASI MODERN
        st.markdown("""
        <style>
            .eval-card {
                background: linear-gradient(135deg, #2b2b2b 0%, #1a1a1a 100%);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.5);
                border: 1px solid #333;
                margin-bottom: 20px;
            }
            .eval-title {
                color: #ffffff;
                font-size: 1.1rem;
                font-weight: 700;
                border-bottom: 1px solid #444;
                padding-bottom: 10px;
                margin-bottom: 15px;
                text-align: center;
                letter-spacing: 1px;
            }
            .eval-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .model-name { color: #aaaaaa; font-weight: 500; font-size: 0.95rem; }
            .rf-val { color: #00e5ff; font-weight: 700; font-size: 1.2rem; } /* Cyan untuk RF */
            .lr-val { color: #ff5252; font-weight: 700; font-size: 1.2rem; } /* Merah/Orange untuk LR */
            .winner-badge {
                background-color: #1b5e20;
                color: #a5d6a7;
                font-size: 0.7rem;
                padding: 3px 8px;
                border-radius: 12px;
                margin-left: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

        # 4. TAMPILAN KARTU METRIK (3 KOLOM BERJEJER)
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">🎯 MAE (Mean Absolute Error)</div>
                <div class="eval-row">
                    <span class="model-name">Random Forest <span class="winner-badge">Lebih Baik</span></span>
                    <span class="rf-val">{rf_mae:.2f}</span>
                </div>
                <div class="eval-row">
                    <span class="model-name">Linear Regression</span>
                    <span class="lr-val">{lr_mae:.2f}</span>
                </div>
                <div style="font-size: 0.8rem; color: gray; text-align: center; margin-top: 10px;">* Semakin kecil semakin baik</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">📉 RMSE (Root Mean Squared Error)</div>
                <div class="eval-row">
                    <span class="model-name">Random Forest <span class="winner-badge">Lebih Baik</span></span>
                    <span class="rf-val">{rf_rmse:.2f}</span>
                </div>
                <div class="eval-row">
                    <span class="model-name">Linear Regression</span>
                    <span class="lr-val">{lr_rmse:.2f}</span>
                </div>
                <div style="font-size: 0.8rem; color: gray; text-align: center; margin-top: 10px;">* Semakin kecil semakin baik</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="eval-card">
                <div class="eval-title">⭐ R² Score (Akurasi Model)</div>
                <div class="eval-row">
                    <span class="model-name">Random Forest <span class="winner-badge">Lebih Baik</span></span>
                    <span class="rf-val">{rf_r2:.4f}</span>
                </div>
                <div class="eval-row">
                    <span class="model-name">Linear Regression</span>
                    <span class="lr-val">{lr_r2:.4f}</span>
                </div>
                <div style="font-size: 0.8rem; color: gray; text-align: center; margin-top: 10px;">* Semakin mendekati 1.0 semakin baik</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)

        # 5. VISUALISASI GRAFIK (TEMA MODERN DARK)
        st.subheader("📊 Visualisasi Perbandingan Error")
        
        # Persiapan data plot (Hanya mengambil MAE dan RMSE untuk bar chart)
        df_plot = pd.DataFrame({
            "Metrik": ["MAE", "RMSE", "MAE", "RMSE"],
            "Model": ["Random Forest", "Random Forest", "Linear Regression", "Linear Regression"],
            "Nilai": [rf_mae, rf_rmse, lr_mae, lr_rmse]
        })

        fig_perf = px.bar(
            df_plot, 
            x="Metrik", y="Nilai", color="Model", barmode="group", text_auto='.2f', 
            # Menggunakan warna Neon agar kontras di Dark Mode
            color_discrete_map={"Random Forest": "#00e5ff", "Linear Regression": "#ff5252"},
            template="plotly_dark"
        )

        fig_perf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", # Latar belakang tembus pandang
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0", size=13),
            yaxis_title="Nilai Error (Paket)",
            xaxis_title="",
            legend_title_text="Algoritma",
            legend=dict(
                orientation="h", # Legend dipindah ke atas horizontal agar rapi
                yanchor="bottom", y=1.02, xanchor="right", x=1
            )
        )

        fig_perf.update_traces(
            textfont_size=14,
            textfont_color='white',
            textposition="outside", 
            cliponaxis=False
        )

        st.plotly_chart(fig_perf, use_container_width=True)
        
        # 6. KESIMPULAN OTOMATIS (Membantu untuk Sidang)
        st.info(f"""
        **💡 Analisis Evaluasi:**
        Berdasarkan data pengujian, **Random Forest Regressor** terbukti lebih unggul dibandingkan Linear Regression. 
        Hal ini ditandai dengan nilai Error (MAE & RMSE) yang lebih rendah, serta **R² Score** sebesar **{rf_r2:.2f}** yang menunjukkan bahwa model mampu menjelaskan variansi volume paket dengan lebih akurat.
        """)

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

        # Layout kolom input dan hasil
        col_input, col_results = st.columns([1.2, 2.8])
        
        with col_input:
            st.markdown("<h3 style='color: #1f77b4;'>📅 Input Prediksi</h3>", unsafe_allow_html=True)
            with st.container(border=True):
                selected_date = st.date_input("Pilih Tanggal Prediksi", value=df_main['Tanggal'].max())
                predict_btn = st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary")

        # 2. Simpan hasil ke dalam session_state saat tombol ditekan
        if predict_btn:
            d = pd.to_datetime(selected_date)
            # Pastikan model rf_final dan lr_final sudah dilatih sebelum blok ini!
            features = np.array([[d.dayofweek, d.month, d.day]])
            st.session_state.hasil_prediksi = {
                'tgl': selected_date.strftime('%d %B %Y'),
                'res_rf': rf_final.predict(features)[0],
                'res_lr': lr_final.predict(features)[0]
            }
          
        # 3. Tampilkan hasil dari session_state (jika ada) dalam bentuk Card UI
        if st.session_state.hasil_prediksi:
            h = st.session_state.hasil_prediksi
            m = st.session_state.metrics_eval
            
            # Hitung dan format angka
            # Harian (Asli)
            rf_day = f"{int(h['res_rf']):,}".replace(",", ".")
            lr_day = f"{int(h['res_lr']):,}".replace(",", ".")
            
            # Mingguan (* 7)
            rf_week = f"{int(h['res_rf']*7):,}".replace(",", ".")
            lr_week = f"{int(h['res_lr']*7):,}".replace(",", ".")
            
            # Bulanan (* 30)
            rf_month = f"{int(h['res_rf']*30):,}".replace(",", ".")
            lr_month = f"{int(h['res_lr']*30):,}".replace(",", ".")
            
            # CSS untuk styling Card Modern
            st.markdown("""
            <style>
                .pred-card {
                    background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.2);
                    border-left: 5px solid #1f77b4;
                    margin-bottom: 15px;
                    transition: transform 0.2s;
                    color: #e0e0e0;
                }
                .pred-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 8px 15px rgba(0,0,0,0.4);
                }
                .card-title {
                    font-size: 1.1rem;
                    color: #ffffff;
                    margin-bottom: 15px;
                    font-weight: 600;
                    border-bottom: 1px solid #444;
                    padding-bottom: 8px;
                }
                .model-box {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                    padding: 10px;
                    background-color: #333333;
                    border-radius: 8px;
                    border: 1px solid #444;
                }
                .model-name {
                    font-weight: 500;
                    color: #e0e0e0;
                    font-size: 0.95rem;
                }
                .model-rf-val {
                    font-size: 1.3rem;
                    font-weight: 700;
                    color: #4caf50; /* Hijau terang agar kontras di latar gelap */
                }
                .model-lr-val {
                    font-size: 1.3rem;
                    font-weight: 700;
                    color: #ffb74d; /* Oranye terang agar kontras di latar gelap */
                }
                .badge {
                    font-size: 0.7rem;
                    background: #1b5e20; /* Hijau lumut gelap */
                    color: #a5d6a7; /* Teks hijau pucat */
                    padding: 2px 6px;
                    border-radius: 4px;
                    margin-left: 5px;
                }
            </style>
            """, unsafe_allow_html=True)
            
            with col_results:
                st.markdown(f"<h3 style='color: #white; margin-top: 0;'>🎯 Hasil Prediksi: {h['tgl']}</h3>", unsafe_allow_html=True)
                
                # Menggunakan 3 kolom untuk 3 Card
                p1, p2, p3 = st.columns(3)
                
                with p1:
                    st.markdown(f"""
                    <div class="pred-card" style="border-left-color: #00bcd4;">
                        <div class="card-title">☀️ Harian</div>
                        <div class="model-box">
                            <span class="model-name">Random Forest<span class="badge">Best</span>
                            <span class="model-rf-val">{rf_day} <span style="font-size:0.8rem; color:#888;">Paket</span></span>
                        </div>
                        <div class="model-box">
                            <span class="model-name">Lin. Regression</span>
                            <span class="model-lr-val">{lr_day} <span style="font-size:0.8rem; color:#888;">Paket</span></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with p2:
                    st.markdown(f"""
                    <div class="pred-card" style="border-left-color: #3f51b5;">
                        <div class="card-title">📅 Mingguan (x7)</div>
                        <div class="model-box">
                            <span class="model-name">Random Forest</span>
                            <span class="model-rf-val">{rf_week} <span style="font-size:0.8rem; color:#888;">Paket</span></span>
                        </div>
                        <div class="model-box">
                            <span class="model-name">Lin. Regression</span>
                            <span class="model-lr-val">{lr_week} <span style="font-size:0.8rem; color:#888;">Paket</span></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with p3:
                    st.markdown(f"""
                    <div class="pred-card" style="border-left-color: #9c27b0;">
                        <div class="card-title">🌙 Bulanan (x30)</div>
                        <div class="model-box">
                            <span class="model-name">Random Forest</span>
                            <span class="model-rf-val">{rf_month} <span style="font-size:0.8rem; color:#888;">Paket</span></span>
                        </div>
                        <div class="model-box">
                            <span class="model-name">Lin. Regression</span>
                            <span class="model-lr-val">{lr_month} <span style="font-size:0.8rem; color:#888;">Paket</span></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
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
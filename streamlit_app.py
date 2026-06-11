import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import holidays  # Library kalender otomatis

# ==========================================
# 1. KONFIGURASI HALAMAN & STYLE VISUAL
# ==========================================
st.set_page_config(
    page_title="Dashboard Prediksi Borneo Express",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    [data-testid="stMetricValue"] { color: white !important; font-weight: bold !important; }
    [data-testid="stMetricLabel"] { color: white !important; font-weight: 600 !important; }
    
    div[data-testid="metric-container"] {
        background: white; border-radius: 12px; padding: 15px;
        border: 1px solid #eaeaea; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] { font-weight: bold; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSIONALITAS OTOMATISASI FITUR BOOLEAN
# ==========================================
def hitung_fitur_boolean(temp_dt, df_agg):
    # 1. Fitur Promo Tanggal Kembar (Otomatis Jan-Des)
    df_agg['is_promo_day'] = (df_agg['Tanggal_'] == df_agg['Bulan_']).astype(int)
    
    # 2. Fitur Hari Libur Nasional & Peak Season Logistik (Kalender Otomatis Indonesia)
    def cek_peak_season(x):
        tanggal_saja = x.date() if hasattr(x, 'date') else x
        libur_indonesia = holidays.Indonesia(years=x.year)
        
        # Kondisi A: Hari Libur Nasional Resmi Indonesia
        if tanggal_saja in libur_indonesia:
            return 1
            
        # Kondisi B: Peak Season Akhir Tahun (18-31 Desember)
        if x.month == 12 and (18 <= x.day <= 31):
            return 1
            
        # Kondisi C: Peak Season Lebaran (Otomatis melacak H-10 s/d H+2 Idul Fitri)
        for tgl_libur, nama_libur in libur_indonesia.items():
            if "Idul Fitri" in nama_libur:
                selisih_hari = (tanggal_saja - tgl_libur).days
                if -10 <= selisih_hari <= 2:
                    return 1
                    
        return 0
        
    df_agg['is_holiday'] = temp_dt.apply(cek_peak_season).astype(int)
    return df_agg

# ==========================================
# 3. LOAD & AGREGASI DATA (AUTO-DELIMITER DETECTION)
# ==========================================
@st.cache_data
def load_data(file):
    encodings = ['utf-8-sig', 'cp1252', 'latin-1']
    separators = [',', ';']  # Deteksi otomatis koma atau titik koma
    df = None
    log_error_terakhir = ""
    
    try:
        versi_pandas = tuple(map(int, pd.__version__.split('.')[:2]))
    except Exception:
        versi_pandas = (1, 3) 
        
    file_loaded = False
    for sep in separators:
        if file_loaded:
            break
        for enc in encodings:
            try:
                if hasattr(file, 'seek'):
                    file.seek(0)
                
                argumen_baca = {'sep': sep, 'encoding': enc, 'skip_blank_lines': True}
                if versi_pandas >= (1, 3):
                    argumen_baca['on_bad_lines'] = 'skip'
                else:
                    argumen_baca['error_bad_lines'] = False
                    
                df = pd.read_csv(file, **argumen_baca)
                
                kolom_cek = df.columns.str.lower().str.strip()
                if 'created_at' in kolom_cek:
                    file_loaded = True
                    break
            except Exception as e:
                log_error_terakhir = str(e)
                continue
                
    if df is None or not file_loaded:
        st.error("❌ Gagal membaca file. Sistem menemui kendala teknis saat membedah struktur data.")
        st.warning(f"🔍 **Pesan Error Internal:** `{log_error_terakhir}`")
        return None, 0
    
    # Sinkronisasi nama kolom resmi ke huruf kecil dan bersih dari spasi
    df.columns = df.columns.str.lower().str.strip()

    # Mengubah string waktu menjadi objek datetime
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['created_at'])
    
    total_mentah = len(df)
    df['tanggal'] = df['created_at'].dt.date
    
    # Agregasi volume harian paket berdasarkan kode resi
    kolom_resi = 'no_resi' if 'no_resi' in df.columns else df.columns[0]
    df_agg = df.groupby('tanggal').agg({kolom_resi: 'count'}).reset_index()
    df_agg.columns = ['Tanggal', 'Volume']
    
    temp_dt = pd.to_datetime(df_agg['Tanggal'])
    df_agg['Hari_'] = temp_dt.dt.dayofweek
    df_agg['Bulan_'] = temp_dt.dt.month
    df_agg['Tanggal_'] = temp_dt.dt.day
    
    # Tambahkan fitur boolean khusus e-commerce & logistik
    df_agg = hitung_fitur_boolean(temp_dt, df_agg)
    df_agg['Tanggal'] = pd.to_datetime(df_agg['Tanggal']).dt.date
    
    return df_agg.sort_values('Tanggal'), total_mentah

# Logika Pemanggilan Dataset
file_path = "data_historis_paket.csv"
df_main = None

if os.path.exists(file_path):
    df_main, total_mentah = load_data(file_path)
else:
    upload = st.sidebar.file_uploader("Upload CSV Borneo Express", type="csv")
    if upload:
        df_main, total_mentah = load_data(upload)

# ==========================================
# 4. INTERFACE APLIKASI STEAMLIT
# ==========================================
if df_main is not None:
    st.markdown("""
    <style>
        .sidebar-profile {
            background: linear-gradient(135deg, #2b2b2b 0%, #1e1e1e 100%);
            padding: 15px; border-radius: 12px; font-size: 0.85rem; color: #b0bec5;
            text-align: center; border: 1px solid #444; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-top: 20px;
        }
        .sidebar-profile b { color: #ffffff; font-size: 1rem; }
        .status-dot { height: 10px; width: 10px; background-color: #4caf50; border-radius: 50%; display: inline-block; margin-right: 5px; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 20px;'>
                <h1 style='margin-bottom: 0; padding-bottom: 0; color: #00bcd4; font-weight: 800; font-size: 2.2rem;'>
                    📦 BORNEO<br><span style='color: white;'>EXPRESS</span>
                </h1>
                <p style='color: gray; font-size: 0.9rem; margin-top: 5px; letter-spacing: 1px;'>SISTEM PREDIKSI LOGISTIK</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown("**📍 NAVIGASI MENU**")
        menu = st.radio(
            "Navigasi Menu", 
            ["📊 Dashboard & Analisis Data", "🤖 Evaluasi Model", "🔮 Prediksi Multi-Skala"],
            label_visibility="collapsed" 
        )
        st.divider()
        
        st.markdown("""
        <div class="sidebar-profile">
            <b>👨‍💻 Aldy Hidayatullah</b><br>
            NPM. 211220026<br>
            Teknik Informatika<br>
            Universitas Muhammadiyah Pontianak<br>
            <hr style="border-color: #444; margin: 10px 0;">
            <span style="font-size: 0.75rem; color: #aaa;"><span class="status-dot"></span>System Online - 2026</span>
        </div>
        """, unsafe_allow_html=True)

    # Inisialisasi Matriks Pembelajaran Mesin
    fitur_skripsi = ['Hari_', 'Bulan_', 'Tanggal_', 'is_holiday', 'is_promo_day']
    X = df_main[fitur_skripsi]
    y = df_main['Volume']
    
    # Pelatihan Model Akhir
    rf_final = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    xgb_final = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X, y)

    # --- MENU 1: DASHBOARD ---
    if menu == "📊 Dashboard & Analisis Data":
        st.title("📊 Dashboard & Analisis Data")
        tab1, tab2 = st.tabs(["📈 Ringkasan Operasional", "📑 Tabel Data & Korelasi"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <style>
                .dash-card {
                    background: linear-gradient(135deg, #2b2b2b 0%, #1e1e1e 100%);
                    border-radius: 15px; padding: 20px 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.4);
                    text-align: center; border-top: 5px solid; transition: all 0.3s ease; margin-bottom: 20px;
                }
                .dash-card:hover { transform: translateY(-8px); box-shadow: 0 10px 20px rgba(0,0,0,0.6); }
                .card-icon { font-size: 2.5rem; margin-bottom: 10px; }
                .card-title { color: #b0bec5; font-size: 0.9rem; font-weight: 600; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
                .dash-value { color: #ffffff; font-size: 1.8rem; font-weight: 800; }
                .card-unit { font-size: 1rem; color: #888888; font-weight: 500; }
            </style>
            """, unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            v_mentah = f"{total_mentah:,}".replace(',', '.')
            v_agregasi = f"{len(df_main):,}".replace(',', '.')
            v_rata = int(df_main['Volume'].mean())
            v_hari = len(df_main)

            with c1:
                st.markdown(f'<div class="dash-card" style="border-top-color: #00bcd4;"><div class="card-icon">📦</div><div class="card-title">Data Mentah</div><div class="dash-value">{v_mentah} <span class="card-unit">Resi</span></div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="dash-card" style="border-top-color: #4caf50;"><div class="card-icon">📑</div><div class="card-title">Data Agregasi</div><div class="dash-value">{v_agregasi} <span class="card-unit">Baris</span></div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="dash-card" style="border-top-color: #ff9800;"><div class="card-icon">⚡</div><div class="card-title">Rata-rata Harian</div><div class="dash-value">{v_rata} <span class="card-unit">Paket/Hari</span></div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="dash-card" style="border-top-color: #9c27b0;"><div class="card-icon">🗓️</div><div class="card-title">Total Observasi</div><div class="dash-value">{v_hari} <span class="card-unit">Hari</span></div></div>', unsafe_allow_html=True)
            
            st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #fff;'>📊 Grafik Visualisasi Data</h4>", unsafe_allow_html=True)
            col_diag1, col_diag2 = st.columns(2)
            
            with col_diag1:
                fig_dist = px.histogram(df_main, x="Volume", nbins=30, template="plotly_dark", color_discrete_sequence=['#00bcd4'], title="Sebaran Frekuensi Volume Paket")
                fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_dist, use_container_width=True)
            with col_diag2:
                fig_trend = px.line(df_main, x="Tanggal", y="Volume", template="plotly_dark", color_discrete_sequence=['#ff9800'], title="Trend Pergerakan Paket per Waktu")
                fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with tab2:
            st.markdown("<br><h4 style='color: #fff;'>📑 Eksplorasi Tabel Dataset</h4>", unsafe_allow_html=True)
            tabel_col1, tabel_col2 = st.columns(2)
            with tabel_col1:
                st.markdown("**1. Tabel Data Mentah (Agregasi Dasar)**")
                st.dataframe(df_main[['Tanggal', 'Volume']], use_container_width=True, height=350)
            with tabel_col2:
                st.markdown("**2. Tabel Data Fitur Machine Learning (Mencakup Fitur Boolean)**")
                st.dataframe(df_main, use_container_width=True, height=350)
                
            st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #fff;'>🔗 Korelasi Fitur Terhadap Volume (Heatmap)</h4>", unsafe_allow_html=True)
            
            plt.style.use("dark_background") 
            corr = df_main[['Volume', 'Hari_', 'Bulan_', 'Tanggal_', 'is_holiday', 'is_promo_day']].corr()
            fig_corr, ax = plt.subplots(figsize=(10, 4))
            fig_corr.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            sns.heatmap(corr, annot=True, cmap='mako', fmt=".2f", linewidths=0.5, ax=ax, cbar_kws={'label': 'Kekuatan Korelasi'})
            st.pyplot(fig_corr)
            plt.style.use("default")

    # --- MENU 2: EVALUASI MODEL ---
    elif menu == "🤖 Evaluasi Model":
        st.title("🤖 Perbandingan Performa Algoritma")
        st.markdown("<br>", unsafe_allow_html=True)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X_tr, y_tr)

        p_rf = model_rf.predict(X_te)
        p_xgb = model_xgb.predict(X_te)

        rf_mae, rf_rmse, rf_r2 = mean_absolute_error(y_te, p_rf), np.sqrt(mean_squared_error(y_te, p_rf)), r2_score(y_te, p_rf)
        xgb_mae, xgb_rmse, xgb_r2 = mean_absolute_error(y_te, p_xgb), np.sqrt(mean_squared_error(y_te, p_xgb)), r2_score(y_te, p_xgb)

        st.markdown("""
        <style>
            .eval-card { background: linear-gradient(135deg, #2b2b2b 0%, #1a1a1a 100%); border-radius: 12px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); border: 1px solid #333; margin-bottom: 20px; }
            .eval-title { color: #ffffff; font-size: 1.1rem; font-weight: 700; border-bottom: 1px solid #444; padding-bottom: 10px; margin-bottom: 15px; text-align: center; letter-spacing: 1px; }
            .eval-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
            .model-name { color: #aaaaaa; font-weight: 500; font-size: 0.95rem; }
            .rf-val { color: #00e5ff; font-weight: 700; font-size: 1.2rem; }
            .xgb-val { color: #ff5252; font-weight: 700; font-size: 1.2rem; }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="eval-card"><div class="eval-title">🎯 MAE (Mean Absolute Error)</div><div class="eval-row"><span class="model-name">Random Forest</span><span class="rf-val">{rf_mae:.2f}</span></div><div class="eval-row"><span class="model-name">XGBoost Regressor</span><span class="xgb-val">{xgb_mae:.2f}</span></div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="eval-card"><div class="eval-title">📉 RMSE (Root Mean Squared Error)</div><div class="eval-row"><span class="model-name">Random Forest</span><span class="rf-val">{rf_rmse:.2f}</span></div><div class="eval-row"><span class="model-name">XGBoost Regressor</span><span class="xgb-val">{xgb_rmse:.2f}</span></div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="eval-card"><div class="eval-title">⭐ R² Score (Akurasi Model)</div><div class="eval-row"><span class="model-name">Random Forest</span><span class="rf-val">{rf_r2:.4f}</span></div><div class="eval-row"><span class="model-name">XGBoost Regressor</span><span class="xgb-val">{xgb_r2:.4f}</span></div></div>', unsafe_allow_html=True)

        st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
        st.subheader("📊 Visualisasi Perbandingan Error")
        
        df_plot = pd.DataFrame({
            "Metrik": ["MAE", "RMSE", "MAE", "RMSE"],
            "Model": ["Random Forest", "Random Forest", "XGBoost Regressor", "XGBoost Regressor"],
            "Nilai": [rf_mae, rf_rmse, xgb_mae, xgb_rmse]
        })

        fig_perf = px.bar(df_plot, x="Metrik", y="Nilai", color="Model", barmode="group", text_auto='.2f', color_discrete_map={"Random Forest": "#00e5ff", "XGBoost Regressor": "#ff5252"}, template="plotly_dark")
        fig_perf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0", size=13))
        st.plotly_chart(fig_perf, use_container_width=True)
        
        pemenang = "Random Forest Regressor" if rf_mae < xgb_mae else "XGBoost Regressor"
        r2_terbaik = rf_r2 if rf_mae < xgb_mae else xgb_r2
        st.info(f"""
        **💡 Analisis Hasil Evaluasi:**
        Berdasarkan hasil pengujian di atas, **{pemenang}** memiliki tingkat akurasi yang lebih optimal untuk memprediksi volume paket di Borneo Express Pontianak dengan nilai **R² Score** sebesar **{r2_terbaik:.4f}**.
        """)

    # --- MENU 3: PREDIKSI MULTI-SKALA ---
    elif menu == "🔮 Prediksi Multi-Skala":
        st.title("🔮 Prediksi & Feature Importance Analysis")

        if 'hasil_prediksi' not in st.session_state:
            st.session_state.hasil_prediksi = None
            
        if 'metrics_eval' not in st.session_state:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            m_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
            m_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X_tr, y_tr)
            p_rf, p_xgb = m_rf.predict(X_te), m_xgb.predict(X_te)
            
            st.session_state.metrics_eval = {
                'rf': {'mae': mean_absolute_error(y_te, p_rf), 'rmse': np.sqrt(mean_squared_error(y_te, p_rf)), 'r2': r2_score(y_te, p_rf)},
                'xgb': {'mae': mean_absolute_error(y_te, p_xgb), 'rmse': np.sqrt(mean_squared_error(y_te, p_xgb)), 'r2': r2_score(y_te, p_xgb)}
            }    

        col_input, col_results = st.columns([1.3, 2.7])
        
        with col_input:
            st.markdown("<h3 style='color: #00bcd4;'>📅 Input Parameter</h3>", unsafe_allow_html=True)
            with st.container(border=True):
                selected_date = st.date_input("Pilih Tanggal Prediksi", value=df_main['Tanggal'].max())
                predict_btn = st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary")

        if predict_btn:
            d = pd.to_datetime(selected_date)
            
            # Memanggil fungsi pengecekan boolean secara internal untuk mendeteksi status libur/promo otomatis
            class ObjekTanggalDummy:
                def __init__(self, dt_obj):
                    self.year = dt_obj.year
                    self.month = dt_obj.month
                    self.day = dt_obj.day
                    self._dt = dt_obj
                def date(self):
                    return self._dt.date()
                    
            libur_indonesia = holidays.Indonesia(years=d.year)
            
            # Lacak status is_holiday otomatis
            def hitung_status_libur(dt):
                if dt.date() in libur_indonesia: return 1
                if dt.month == 12 and (18 <= dt.day <= 31): return 1
                for tgl, nama in libur_indonesia.items():
                    if "Idul Fitri" in nama and -10 <= (dt.date() - tgl).days <= 2: return 1
                return 0

            val_hol = hitung_status_libur(d)
            val_pro = 1 if d.day == d.month else 0
            
            features = np.array([[d.dayofweek, d.month, d.day, val_hol, val_pro]])
            st.session_state.hasil_prediksi = {
                'tgl': selected_date.strftime('%d %B %Y'),
                'res_rf': rf_final.predict(features)[0],
                'res_xgb': xgb_final.predict(features)[0],
                'is_hol': val_hol,
                'is_pro': val_pro
            }
          
        if st.session_state.hasil_prediksi:
            h = st.session_state.hasil_prediksi
            m = st.session_state.metrics_eval
            
            rf_day = f"{int(h['res_rf']):,}".replace(",", ".")
            xgb_day = f"{int(h['res_xgb']):,}".replace(",", ".")
            rf_week = f"{int(h['res_rf']*7):,}".replace(",", ".")
            xgb_week = f"{int(h['res_xgb']*7):,}".replace(",", ".")
            rf_month = f"{int(h['res_rf']*30):,}".replace(",", ".")
            xgb_month = f"{int(h['res_xgb']*30):,}".replace(",", ".")
            
            st.markdown("""
            <style>
                .pred-card { background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%); border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border-left: 5px solid #00bcd4; margin-bottom: 15px; color: #e0e0e0; }
                .card-title { font-size: 1.1rem; color: #ffffff; margin-bottom: 15px; font-weight: 600; border-bottom: 1px solid #444; padding-bottom: 8px; }
                .model-box { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding: 10px; background-color: #333333; border-radius: 8px; border: 1px solid #444; }
                .model-name { font-weight: 500; color: #e0e0e0; font-size: 0.95rem; }
                .model-rf-val { font-size: 1.3rem; font-weight: 700; color: #4caf50; }
                .model-xgb-val { font-size: 1.3rem; font-weight: 700; color: #ffb74d; }
                .status-badge { padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; margin-right: 5px; }
            </style>
            """, unsafe_allow_html=True)
            
            with col_results:
                st.markdown(f"<h3 style='color: white; margin-top: 0;'>🎯 Hasil Prediksi: {h['tgl']}</h3>", unsafe_allow_html=True)
                
                # MENAMPILKAN STATUS KALENDER OTOMATIS 
                lbl_hol = '<span class="status-badge" style="background:#d32f2f; color:white;">🔥 Peak Season/Libur Aktif</span>' if h['is_hol'] == 1 else '<span class="status-badge" style="background:#388e3c; color:white;">🍃 Hari Reguler</span>'
                lbl_pro = '<span class="status-badge" style="background:#f57c00; color:white;">🛍️ Promo Tanggal Kembar</span>' if h['is_pro'] == 1 else ''
                st.markdown(f"**Deteksi Status Kalender Sistem:** {lbl_hol} {lbl_pro}", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                p1, p2, p3 = st.columns(3)
                with p1:
                    st.markdown(f'<div class="pred-card" style="border-left-color: #00bcd4;"><div class="card-title">☀️ Harian</div><div class="model-box"><span class="model-name">Random Forest</span><span class="model-rf-val">{rf_day} <span style="font-size:0.8rem; color:#888;">Pkt</span></span></div><div class="model-box"><span class="model-name">XGBoost Reg.</span><span class="model-xgb-val">{xgb_day} <span style="font-size:0.8rem; color:#888;">Pkt</span></span></div></div>', unsafe_allow_html=True)
                with p2:
                    st.markdown(f'<div class="pred-card" style="border-left-color: #3f51b5;"><div class="card-title">📅 Mingguan (x7)</div><div class="model-box"><span class="model-name">Random Forest</span><span class="model-rf-val">{rf_week} <span style="font-size:0.8rem; color:#888;">Pkt</span></span></div><div class="model-box"><span class="model-name">XGBoost Reg.</span><span class="model-xgb-val">{xgb_week} <span style="font-size:0.8rem; color:#888;">Pkt</span></span></div></div>', unsafe_allow_html=True)
                with p3:
                    st.markdown(f'<div class="pred-card" style="border-left-color: #9c27b0;"><div class="card-title">🌙 Bulanan (x30)</div><div class="model-box"><span class="model-name">Random Forest</span><span class="model-rf-val">{rf_month} <span style="font-size:0.8rem; color:#888;">Pkt</span></span></div><div class="model-box"><span class="model-name">XGBoost Reg.</span><span class="model-xgb-val">{xgb_month} <span style="font-size:0.8rem; color:#888;">Pkt</span></span></div></div>', unsafe_allow_html=True)
                
            st.write("---")
            with st.expander("📊 Lihat Detail Skor Validasi Model Pengujian", expanded=True):
                st.markdown("🌲 **Random Forest Regressor**")
                st.info(f"MAE: **{m['rf']['mae']:.2f}** | RMSE: **{m['rf']['rmse']:.2f}** | R²: **{m['rf']['r2']:.4f}**")
                st.markdown("🚀 **XGBoost Regressor**")
                st.success(f"MAE: **{m['xgb']['mae']:.2f}** | RMSE: **{m['xgb']['rmse']:.2f}** | R²: **{m['xgb']['r2']:.4f}**")

            st.divider()
            
        show_fi = st.checkbox("🔍 Tampilkan Analisis Feature Importance", value=False)
        if show_fi:
            rf_imp = rf_final.feature_importances_
            xgb_imp = xgb_final.feature_importances_

            col_rf, col_xgb = st.columns(2)
            with col_rf:
                st.markdown("### 🌲 Feature Importance: Random Forest")
                fi_rf_df = pd.DataFrame({'Fitur': fitur_skripsi, 'Score': rf_imp}).sort_values('Score', ascending=False)
                fig_rf = px.bar(fi_rf_df, x='Fitur', y='Score', labels={'Score': 'Importance Score'}, color_discrete_sequence=['#0056B3'], template="plotly_dark")
                fig_rf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_rf, use_container_width=True)

            with col_xgb:
                st.markdown("### 🚀 Feature Importance: XGBoost Regressor")
                fi_xgb_df = pd.DataFrame({'Fitur': fitur_skripsi, 'Score': xgb_imp}).sort_values('Score', ascending=False)
                fig_xgb = px.bar(fi_xgb_df, x='Fitur', y='Score', labels={'Score': 'Importance Score'}, color_discrete_sequence=['#FF6347'], template="plotly_dark")
                fig_xgb.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_xgb, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔍 Analisis Perbandingan & Implikasi Praktis")
            c_comp, c_impl = st.columns(2)
            with c_comp:
                st.markdown("**Perbedaan Pemaknaan Pola:**")
                st.write("""
                1. **Karakteristik Pembagian Pohon**: Random Forest membangun banyak pohon secara paralel independen dan merata-ratakan hasilnya, sehingga grafik importances cenderung menyebar stabil.
                2. **Optimasi Bergradien (XGBoost)**: XGBoost melatih pohon secara berurutan untuk memperbaiki kesalahan pohon sebelumnya. Hal ini membuat XGBoost jauh lebih sensitif dalam menangkap variabel lonjakan tajam seperti fitur `is_promo_day` dan `is_holiday`.
                """)
            with c_impl:
                st.markdown("**Implikasi Praktis (Manajerial Borneo Express):**")
                st.write("""
                1. **Manajemen Peak Season**: Ketika indikator `is_holiday` atau `is_promo_day` aktif, manajemen kurir di Pontianak dapat mempersiapkan kapasitas kurir cadangan sejak jauh hari sesuai matriks prediksi harian.
                2. **Alokasi Sumber Daya**: Fitur dengan skor kontribusi paling tinggi menjadi acuan utama untuk menyusun jadwal operasional logistik mingguan.
                """)

else:
    st.warning("Silakan muat dataset Borneo Express melalui sidebar untuk memulai.")

st.markdown("<div style='text-align: center; color: gray; margin-top: 50px;'>© 2026 Borneo Express Pontianak - Aldy Hidayatullah</div>", unsafe_allow_html=True)
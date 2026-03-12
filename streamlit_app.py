import streamlit as st

st.title("🎈 My new app")
st.write("Let's start building!")

# Contoh interaksi sederhana
nama = st.text_input("Siapa nama Anda?")
if st.button("Sapa Saya"):
    st.write(f"Halo {nama}, selamat membangun aplikasi!")
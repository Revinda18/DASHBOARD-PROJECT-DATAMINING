import streamlit as st
import pandas as pd
import joblib
from CONVERT_TO_PKL.feature_importance import show_feature_importance_page
from overview import show_overview
from song_recommendation import show_song_recommendation
from favorite_song_prediction import show_label_analysis

@st.cache_data
def load_data():
    df = pd.read_csv("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/DATASET/dataset.csv")
    return df

df = load_data()

model = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/model_spotify.pkl")
feature_names = [
    "explicit", "danceability", "energy", "mode", "speechiness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature"
]

st.sidebar.title("Spotify Dashboard")
page = st.sidebar.radio("Pilih Halaman", ["Overview Dashboard", "Aplikasi Prediksi Kesukaan Lagu Spotify", "Rekomendasi Lagu", "Feature Importance"])

if page == "Overview Dashboard":
    show_overview(df)
elif page ==  "Aplikasi Prediksi Kesukaan Lagu Spotify":
    show_label_analysis(df)
elif page == "Feature Importance":
    show_feature_importance_page(df, model, feature_names)
elif page == "Rekomendasi Lagu":
    show_song_recommendation(df)
else:
    st.title(f"Halaman {page} sedang dalam pengembangan")

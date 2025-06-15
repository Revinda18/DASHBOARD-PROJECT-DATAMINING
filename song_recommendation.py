import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@st.cache_resource
def load_model_scaler():
    model = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/model_spotify.pkl")
    scaler = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/scaler_spotify.pkl")
    feature_list = joblib.load("D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/fitur_spotify.pkl")
    return model, scaler, feature_list

def align_features(df_input, feature_list):
    df_new = df_input.copy()
    for col in feature_list:
        if col not in df_new.columns:
            df_new[col] = 0
    df_new = df_new[feature_list]
    return df_new

def create_popularity_label(df, threshold=50):
    df['popularity_class'] = df['popularity'].apply(lambda x: 1 if x >= threshold else 0)
    return df

def show_song_recommendation(df):
    st.title("🎵 Rekomendasi Lagu Mirip Berdasarkan Pilihan Lagu")

    model, scaler, feature_list = load_model_scaler()

    if 'popularity_class' not in df.columns:
        df = create_popularity_label(df)

    # List lagu unik untuk pilihan
    song_options = df[['track_name', 'artists']].drop_duplicates()
    song_options['display'] = song_options['track_name'] + " - " + song_options['artists']
    selected_song = st.selectbox("Pilih lagu favorit kamu:", song_options['display'].tolist())

    if selected_song:
        track_name = song_options[song_options['display'] == selected_song]['track_name'].values[0]
        artist_name = song_options[song_options['display'] == selected_song]['artists'].values[0]

        st.write("Lagu yang kamu pilih:")
        selected_song_data = df[(df['track_name'] == track_name) & (df['artists'] == artist_name)]
        st.dataframe(selected_song_data[['track_name', 'artists', 'popularity']].drop_duplicates())

        # Ambil fitur lagu seluruh dataset, dan fitur lagu yang dipilih
        df_features = align_features(df, feature_list).astype(float)
        df_scaled = scaler.transform(df_features)

        # Cari index lagu yang dipilih
        idx_selected = selected_song_data.index[0]
        selected_vec = df_scaled[idx_selected].reshape(1, -1)

        # Hitung cosine similarity seluruh lagu dengan lagu yang dipilih
        cosine_sim = cosine_similarity(selected_vec, df_scaled).flatten()

        # Buat dataframe hasil similarity
        df_sim = df.copy()
        df_sim['similarity'] = cosine_sim

        # Kecualikan lagu yang sama (cosine similarity 1)
        df_sim = df_sim[df_sim.index != idx_selected]

        # Urutkan berdasarkan similarity tertinggi dan popularitas
        df_sim = df_sim.sort_values(by=['similarity', 'popularity'], ascending=False)

        st.subheader("Lagu Rekomendasi Mirip:")
        st.dataframe(df_sim[['track_name', 'artists', 'popularity', 'similarity']].head(20))

    else:
        st.info("Silakan pilih lagu favorit di atas untuk mendapatkan rekomendasi.")


import streamlit as st
import pandas as pd
import joblib
import numpy as np

def show_label_analysis(df):
    st.title("🎧 Aplikasi Prediksi Kesukaan Lagu Spotify")

    # Load model dan scaler
    model_path = "D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/model_spotify.pkl"
    scaler_path = "D:/SEMESTER 6/PENGGALIAN DATA/DASHBOARD/FILE_PKL/scaler_spotify.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    fitur_model = [
        "explicit", "danceability", "energy", "mode", "speechiness", 
        "instrumentalness", "liveness", "valence", "tempo", "time_signature"
    ]

    def align_features(df_input):
        for fitur in fitur_model:
            if fitur not in df_input.columns:
                df_input[fitur] = 0.0
        return df_input[fitur_model]

    mode = st.radio("Pilih Metode Input", ["Pilih Lagu", "Pilih Fitur (Manual)", "Pilih Fitur (.pkl)"])

    if mode == "Pilih Lagu":
        song_list = df[['track_name', 'artists']].drop_duplicates()
        song_list['display'] = song_list['track_name'] + " - " + song_list['artists']
        selected_song = st.selectbox("Pilih lagu:", song_list['display'].tolist())

        if selected_song:
            track = song_list[song_list['display'] == selected_song]['track_name'].values[0]
            artist = song_list[song_list['display'] == selected_song]['artists'].values[0]
            song_data = df[(df['track_name'] == track) & (df['artists'] == artist)]
            features = align_features(song_data).astype(float)
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            label = "Suka" if pred == 1 else "Tidak Suka"
            st.success(f"Prediksi: {label}")
            st.write(f"Probabilitas suka: {proba[1]*100:.2f}%")
            st.write(f"Probabilitas tidak suka: {proba[0]*100:.2f}%")

    elif mode == "Pilih Fitur (Manual)":
        st.write("Pilih fitur yang ingin Anda masukkan nilai:")

        fitur_terpilih = st.multiselect(
            "Pilih fitur:", 
            fitur_model, 
            default=fitur_model
        )

        input_dict = {}

        def input_feature(fitur):
            if fitur == "explicit":
                return st.selectbox("Explicit (1 = Explicit, 0 = Non-explicit)", [0,1], key=fitur)
            elif fitur == "danceability":
                return st.slider("Danceability (0.0-1.0)", 0.0, 1.0, 0.5, key=fitur)
            elif fitur == "energy":
                return st.slider("Energy (0.0-1.0)", 0.0, 1.0, 0.5, key=fitur)
            elif fitur == "mode":
                return st.selectbox("Mode (0 = Minor, 1 = Major)", [0,1], key=fitur)
            elif fitur == "speechiness":
                return st.slider("Speechiness (0.0-1.0)", 0.0, 1.0, 0.05, key=fitur)
            elif fitur == "instrumentalness":
                return st.slider("Instrumentalness (0.0-1.0)", 0.0, 1.0, 0.0, key=fitur)
            elif fitur == "liveness":
                return st.slider("Liveness (0.0-1.0)", 0.0, 1.0, 0.1, key=fitur)
            elif fitur == "valence":
                return st.slider("Valence (0.0-1.0)", 0.0, 1.0, 0.5, key=fitur)
            elif fitur == "tempo":
                return st.slider("Tempo (50-200)", 50.0, 200.0, 120.0, key=fitur)
            elif fitur == "time_signature":
                return st.slider("Time Signature (3-7)", 3, 7, 4, key=fitur)
            else:
                return 0.0

        if fitur_terpilih:
            for fitur in fitur_terpilih:
                input_dict[fitur] = [input_feature(fitur)]

            # Tambahan dari permintaanmu:
            for fitur in fitur_model:
                if fitur not in input_dict:
                    input_dict[fitur] = [0.0]

            input_df = pd.DataFrame(input_dict)

            # Pastikan urutan sesuai fitur_model
            input_df = input_df[fitur_model]

            features_scaled = scaler.transform(input_df)
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            label = "Suka" if pred == 1 else "Tidak Suka"
            st.success(f"Prediksi: {label}")
            st.write(f"Probabilitas suka: {proba[1]*100:.2f}%")
            st.write(f"Probabilitas tidak suka: {proba[0]*100:.2f}%")
        else:
            st.warning("Silakan pilih minimal satu fitur untuk memasukkan nilai.")

    else:  # mode == "Upload File (.pkl)"
        uploaded_file = st.file_uploader("Upload file .pkl yang berisi DataFrame fitur lagu", type=["pkl"])

        if uploaded_file is not None:
            try:
                uploaded_df = joblib.load(uploaded_file)

                if not isinstance(uploaded_df, pd.DataFrame):
                    st.error("File .pkl harus berisi DataFrame.")
                    return

                features = align_features(uploaded_df).astype(float)
                features_scaled = scaler.transform(features)
                preds = model.predict(features_scaled)
                probas = model.predict_proba(features_scaled)

                results = []
                for i in range(len(preds)):
                    label = "Suka" if preds[i] == 1 else "Tidak Suka"
                    prob_suka = probas[i][1]*100
                    prob_tidak = probas[i][0]*100
                    results.append({
                        "Prediksi": label,
                        "Probabilitas Suka (%)": f"{prob_suka:.2f}",
                        "Probabilitas Tidak Suka (%)": f"{prob_tidak:.2f}"
                    })

                results_df = pd.DataFrame(results)
                st.write("Hasil Prediksi:")
                st.dataframe(pd.concat([uploaded_df.reset_index(drop=True), results_df], axis=1))

            except Exception as e:
                st.error(f"Gagal memuat file: {e}")

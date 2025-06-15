import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_overview(df):
    st.subheader("📊INFORMASI DATA")

    # Total jumlah lagu unik yang dianalisis
    if 'track_id' in df.columns:
        total_songs = df['track_id'].nunique()
        st.write(f"Total jumlah lagu yang dianalisis: {total_songs}")
    else:
        st.write("Kolom 'track_id' tidak ditemukan.")

    # Jumlah genre unik
    if 'track_genre' in df.columns:
        total_genres = df['track_genre'].nunique()
        st.write(f"Jumlah genre unik: {total_genres}")
    else:
        st.write("Kolom 'track_genre' tidak ditemukan.")

    # Visualisasi distribusi popularitas lagu
    if 'popularity' in df.columns:
        st.subheader("Distribusi Popularitas Lagu")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['popularity'], bins=30, kde=False, ax=ax1)
        ax1.set_xlabel("Popularitas")
        ax1.set_ylabel("Jumlah Lagu")
        st.pyplot(fig1)
    else:
        st.write("Kolom 'popularity' tidak ditemukan.")

    # Visualisasi jumlah lagu per genre (Top 10)
    if 'track_genre' in df.columns:
        st.subheader("Jumlah Lagu per Genre (Top 10)")
        fig2, ax2 = plt.subplots(figsize=(12,6))
        genre_count = df['track_genre'].value_counts().head(20)
        sns.barplot(x=genre_count.index, y=genre_count.values, ax=ax2, palette="viridis")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_ylabel("Jumlah Lagu")
        st.pyplot(fig2)

        # Visualisasi pie chart distribusi genre (Top 10)
        st.subheader("Distribusi Genre (Top 10)")
        fig3, ax3 = plt.subplots(figsize=(8,8))
        genre_pie = genre_count.head(10)
        ax3.pie(genre_pie.values, labels=genre_pie.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", 10))
        ax3.axis('equal')
        st.pyplot(fig3)
    else:
        st.write("Kolom 'track_genre' tidak ditemukan.")

    # Visualisasi jumlah lagu per artis (Top 10)
    if 'artists' in df.columns:
        st.subheader("Jumlah Lagu per Artis (Top 10)")
        artist_count = df['artists'].value_counts().head(10)
        fig4, ax4 = plt.subplots(figsize=(12,6))
        sns.barplot(x=artist_count.values, y=artist_count.index, ax=ax4, palette="magma")
        ax4.set_xlabel("Jumlah Lagu")
        ax4.set_ylabel("Artis")
        st.pyplot(fig4)

        # Visualisasi distribusi lagu per artis (Top 10)
        st.subheader("Distribusi Lagu per Artis (Top 10)")
        artist_pie = artist_count.head(10)
        fig5, ax5 = plt.subplots(figsize=(8,8))
        ax5.pie(artist_pie.values, labels=artist_pie.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("magma", 10))
        ax5.axis('equal')
        st.pyplot(fig5)
    else:
        st.write("Kolom 'artists' tidak ditemukan.")

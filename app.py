import streamlit as st
import pickle
import requests
import creds
import gzip

with gzip.open("similarity.pkl", "rb") as f:
    p = pickle.Unpickler(f)
    similarity = p.load()


movies_df = pickle.load(open("movies.pkl", "rb"))


def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={creds.api_key}"
    data = requests.get(url)
    data = data.json()
    posterPath = data["poster_path"]
    full_path = "https://image.tmdb.org/t/p/w500/" + posterPath
    return full_path


def recommend(movie):
    index = movies_df[movies_df["title"] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1]
    )
    recommended_movies = []
    RMP = []

    for i in distances[1:6]:
        movie_id = movies_df.iloc[i[0]].movie_id
        RMP.append(fetch_poster(movie_id))
        recommended_movies.append(movies_df.iloc[i[0]].title)

    return recommended_movies, RMP


st.header("Movie Recommender System")

movies_list = movies_df["title"].values

selected_movie = st.selectbox("Type or select a movie", movies_list)

if st.button("Show Recommendations"):
    recommended_movies, RMP = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movies[0])
        st.image(RMP[0])

    with col2:
        st.text(recommended_movies[1])
        st.image(RMP[1])

    with col3:
        st.text(recommended_movies[2])
        st.image(RMP[2])

    with col4:
        st.text(recommended_movies[3])
        st.image(RMP[3])

    with col5:
        st.text(recommended_movies[4])
        st.image(RMP[4])

fetch_poster(5)

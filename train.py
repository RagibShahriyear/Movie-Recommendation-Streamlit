import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# movies["title"][0] == credits["title"][0]
# looks like both dataframes have the same column name 'title'. We can use this to merge these 2 dataframes.

movies = movies.merge(credits, on="title")


# Feature selection
movies = movies[["title", "genres", "keywords", "overview", "movie_id", "cast", "crew"]]

# checking for missing data
movies.isnull().sum()

# checking which movies's features are missing
null_data = movies[movies.isnull().any(axis=1)]

# dropping the movies with missing features
movies.dropna(inplace=True)

# checking if there are any duplicates
movies.duplicated().sum()


# data preprocessing

# making a function to remove the 'id' and only have the 'names' of genres and keywords.


def my_convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L


movies["genres"] = movies["genres"].apply(my_convert)
movies["keywords"] = movies["keywords"].apply(my_convert)

# function for getting the first 3 casts


def my_casts(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i["name"])
            counter += 1
        else:
            break
    return L


movies["cast"] = movies["cast"].apply(my_casts)

# function for getting the director's name


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L


movies["crew"] = movies["crew"].apply(fetch_director)

# Turning the overview into a list

movies["overview"] = movies["overview"].apply(lambda x: x.split())

# removing spaces from genres, keywords, crew, and cast columns

movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])

# creating one single column for everything
movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

# New dataframe with just title, id, and tags
new_df = movies[["title", "movie_id", "tags"]]

# converting the tags into a string from a list and making the lower
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))

new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())


# stemming
ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


new_df["tags"] = new_df["tags"].apply(stem)

# Vectorization
# technique used = bag of words

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

# measuring similary between vectors
similarity = cosine_similarity(vectors)


def recommend(movie):
    movie_index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[
        1:6
    ]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


import gzip, pickle, pickletools

filepath = "similarity.pkl"
with gzip.open(filepath, "wb") as f:
    pickled = pickle.dumps(similarity)
    optimized_pickle = pickletools.optimize(pickled)
    f.write(optimized_pickle)

pickle.dump(new_df, open("movies.pkl", "wb"))
# pickle.dump(similarity, open("similarity.pkl", "wb"))

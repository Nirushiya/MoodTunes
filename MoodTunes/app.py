from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the song dataset
songs = pd.read_csv("songs.csv")

# Combine song information to represent each song
songs["features"] = (
    songs["mood"].fillna("") + " " +
    songs["genre"].fillna("") + " " +
    songs["artist"].fillna("")
)

# Convert song features into numerical vectors
vectorizer = TfidfVectorizer()
song_vectors = vectorizer.fit_transform(songs["features"])

# Calculate similarity between songs
similarity_matrix = cosine_similarity(song_vectors)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    selected_mood = request.form.get("mood", "").strip().title()

    # Check that the selected mood exists in our dataset
    available_moods = songs["mood"].dropna().unique()

    if selected_mood not in available_moods:
        return "Invalid mood selected. Please return and choose a valid mood.", 400

    # Find songs that match the selected mood
    matching_indices = songs.index[
        songs["mood"].str.lower() == selected_mood.lower()
    ].tolist()

    # Use the first matching song as the starting point
    selected_index = matching_indices[0]

    # Rank songs according to their similarity to the starting song
    similarity_scores = list(enumerate(similarity_matrix[selected_index]))

    similarity_scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores:
        # Skip the starting song itself
        if index == selected_index:
            continue

        song = songs.iloc[index]

        recommendations.append({
            "title": song["title"],
            "artist": song["artist"],
            "mood": song["mood"],
            "genre": song["genre"],
            "score": round(float(score) * 100, 1)
        })

        if len(recommendations) == 5:
            break

    return render_template(
        "index.html",
        recommendations=recommendations,
        selected_mood=selected_mood
    )


if __name__ == "__main__":
    app.run(debug=True)



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Movie dataset
data = {
    'Movie': ['Avatar', 'Titanic', 'Avengers',
              'Iron Man', 'Frozen'],
    'Genre': ['Action Sci-Fi',
              'Romance Drama',
              'Action Superhero',
              'Action Superhero',
              'Animation Family']
}

df = pd.DataFrame(data)

# Convert genres into numerical values
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df['Genre'])

# Calculate similarity
similarity = cosine_similarity(matrix)

# Get movie recommendation
def recommend(movie):
    if movie not in df['Movie'].values:
        print("Movie not found!")
        return

    index = df[df['Movie'] == movie].index[0]

    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1],
                    reverse=True)

    print("Recommended Movies:")

    count = 0
    for i, score in scores:
        if i != index:
            print(df.iloc[i]['Movie'])
            count += 1
        if count == 3:
            break

# User input
movie = input("Enter your favourite movie: ")
recommend(movie)

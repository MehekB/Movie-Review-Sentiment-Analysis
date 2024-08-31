import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews
from tmdbv3api import TMDb, Movie
import requests

class MovieSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=2000)
        self.model = MultinomialNB()
        self._prepare_data()

    def _prepare_data(self):
        # nltk.download('movie_reviews') -> can also be downloaded directly from nltk corpus
        documents = [
            (" ".join(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)
        ]

        df = pd.DataFrame(documents, columns=['review', 'sentiment'])
        x = self.vectorizer.fit_transform(df['review'])
        y = df['sentiment']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

    def predict_sentiment(self, text):
        text_vector = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vector)
        return prediction[0]
    
    def analyze_reviews(self, movie):
        positive_count = 0
        for review in movie.reviews:
            sentiment = self.predict_sentiment(review)
            if sentiment == 'pos':
                positive_count += 1

        total_reviews = len(movie.reviews)
        positive_ratio = positive_count / total_reviews
        rating = round(positive_ratio * 10, 3)
        return rating, positive_count, total_reviews

    
class Movie:
    def __init__(self, title):
        self.title = title
        self.reviews = []

    def fetch_reviews(self, api_key):
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={self.title}"
        search_response = requests.get(search_url).json()
       
        if search_response['results']:
            movie_id = search_response['results'][0]['id']
        else:
            print("Movie not found!")
            raise SystemExit
        
        reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}"
        reviews_response = requests.get(reviews_url).json()
        self.reviews = [review['content'] for review in reviews_response['results']]

        if not self.reviews:
            print("No reviews found for this movie")
            raise SystemExit



if __name__ == '__main__':

    API_KEY = ' ' # Replace with your TMDb API key!
    movie_title = input("Enter a movie title: ")

    movie = Movie(movie_title)
    reviews = movie.fetch_reviews(API_KEY)

    analyzer = MovieSentimentAnalyzer()
    rating, positive_count, total_reviews = analyzer.analyze_reviews(movie)

    print(f"Movie: {movie.title}")
    print(f"Rating: {rating}/10 based on {total_reviews} reviews")
    print(f"Positive Reviews: {positive_count}")
    print(f"Negative Reviews: {total_reviews - positive_count}")

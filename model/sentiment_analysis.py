
import nltk
import pandas as pd
import os

df=pd.read_csv(r'D:\Hackathon\SIH(2025)\version_17\econsultation_comments.csv')

input_path = r'D:\Hackathon\SIH(2025)\version_17\econsultation_comments.csv'
output_path = r'D:\Hackathon\SIH(2025)\version_17\econsultation_comments_with_sentiment.csv'

df = pd.read_csv(input_path)

# VADER Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df['vader_sentiment'] = df['comment'].apply(lambda x: sia.polarity_scores(x)['compound'])

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
    # ...existing code...
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)
df['cleaned_comment'] = df['comment'].apply(preprocess_text)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
X = df['cleaned_comment']
y = df['vader_sentiment'].apply(lambda x: 1 if x >
    0 else (0 if x == 0 else -1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

import joblib
joblib.dump(pipeline, 'sentiment_model.pkl')
# Load the model
model = joblib.load('sentiment_model.pkl')

def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    prediction = model.predict([cleaned_text])
    return prediction[0]

# Example usage
example_text = "I love this product! It's absolutely wonderful."
predicted_sentiment = predict_sentiment(example_text)
print(f"Predicted sentiment for the example text: {predicted_sentiment}")



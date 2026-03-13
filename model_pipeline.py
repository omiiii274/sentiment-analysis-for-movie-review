import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from preprocessor import clean_text

def train_sentiment_model(data_path):
    df = pd.read_csv(data_path)
    
    print("🧹 Cleaning text data...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    X = df['cleaned_review']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline: TF-IDF with Bigrams + Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', MultinomialNB())
    ])
    
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline

if __name__ == "__main__":
    # Example usage
    # model = train_sentiment_model('movie_reviews.csv')
    pass

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd


def train(train_df: pd.DataFrame, mode: str ='tf-idf', stop_words = None):
    if mode=='tf-idf':
        vectorizer = TfidfVectorizer(max_features=4000, stop_words=stop_words)
    else:
        vectorizer = CountVectorizer(max_features=4000, stop_words=stop_words)

    x = train_df['text']
    y = train_df['labels']
    vectorizer.fit(x)

    # turn text into bag of words / tf-idf vector
    x = vectorizer.transform(x)

    model = MultinomialNB()
    model.fit(x, y)
    return model, vectorizer

def predict(df: pd.DataFrame,model, vectorizer):
    x = df['text']
    y = df['labels']
    # turn text into bag of words / tf-idf vector
    x = vectorizer.transform(x)
    prediction = model.predict(x)

    return prediction

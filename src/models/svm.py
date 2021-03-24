import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def train(train_df: pd.DataFrame, mode: str ='tf-idf', stop_words = None):
    '''
    Trains a bag of words or tf-idf based SVM on the provided training dataframe
    :param train_df: pandas dataframe with column 'text' and 'label'
    :param mode: when set to 'tf-idf' uses a tf-idf vectorizer, else a bag of words vectorizer
    :return: returns the trained model an the used vectorizer
    '''
    if mode=='tf-idf':
        vectorizer = TfidfVectorizer(max_features=2000, stop_words=stop_words)
    else:
        vectorizer = CountVectorizer(max_features=2000, stop_words=stop_words)

    x = train_df['text']
    y = train_df['labels']
    vectorizer.fit(x)

    # turn text into bag of words / tf-idf vector
    x = vectorizer.transform(x)

    model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    model.fit(x, y)

    return model, vectorizer

def predict(df: pd.DataFrame,model: svm.SVC, vectorizer):
    x = df['text']
    y = df['labels']
    # turn text into bag of words / tf-idf vector
    x = vectorizer.transform(x)
    prediction = model.predict(x)

    return prediction

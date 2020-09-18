import pandas as pd
import click
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, svm

def load_data(trainingdata, testdata, validdata):
    train = pd.read_csv(trainingdata, delimiter=',')
    test = pd.read_csv(testdata, delimiter=',')
    valid = pd.read_csv(validdata, delimiter=',')
    return train, test, valid


@click.command()
@click.option('--trainingdata', default='../../data/processed/in-cat-train.csv', help='path to the training data.')
@click.option('--testdata', default='../../data/processed/in-cat-test.csv', help='path to the test data.')
@click.option('--validata', default='../../data/processed/out-of-cat-valid.csv', help='path to the validation data.')
def main(trainingdata, testdata, validata):
    train, test, valid = load_data(trainingdata, testdata, validata)
    print("data loaded")
    print(f"train: {len(train)}, test: {len(test)}, valid: {len(valid)}")
    x_train = train['text']
    y_train = train['labels']
    x_test = test['text']
    y_test = test['labels']
    x_valid = valid['text']
    y_valid = valid['labels']

    #bag of words
    vectorizer = CountVectorizer(max_features=2000)
    vectorizer.fit(x_train)

    x_train_bow = vectorizer.transform(x_train)
    x_test_bow = vectorizer.transform(x_test)
    x_valid_bow = vectorizer.transform(x_valid)

    #tf idf
    tfidf_vect = TfidfVectorizer(max_features=2000)
    tfidf_vect.fit(x_train)

    x_train_tfidf = tfidf_vect.transform(x_train)
    x_test_tfidf = tfidf_vect.transform(x_test)
    x_valid_tfidf = tfidf_vect.transform(x_valid)

    SVM_bow = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM_tfidf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM_bow.fit(x_train_bow, y_train)
    SVM_tfidf.fit(x_train_tfidf, y_train)

    prediction_svm_test_bow = SVM_bow.predict(x_test_bow)
    prediction_svm_valid_bow = SVM_bow.predict(x_valid_bow)

    prediction_svm_test_tfidf = SVM_tfidf.predict(x_test_tfidf)
    prediction_svm_valid_tfidf = SVM_tfidf.predict(x_valid_tfidf)

    acc_test_bow = accuracy_score(prediction_svm_test_bow, y_test)*100
    acc_valid_bow = accuracy_score(prediction_svm_valid_bow, y_valid)*100
    f1_test_bow = f1_score(prediction_svm_test_bow, y_test) * 100
    f1_valid_bow = f1_score(prediction_svm_valid_bow, y_valid) * 100

    acc_test_tfidf = accuracy_score(prediction_svm_test_tfidf, y_test) * 100
    acc_valid_tfidf = accuracy_score(prediction_svm_valid_tfidf, y_valid) * 100
    f1_test_tfidf = f1_score(prediction_svm_test_tfidf, y_test) * 100
    f1_valid_tfidf = f1_score(prediction_svm_valid_tfidf, y_valid) * 100

    print("test set evaluation BOW")
    print(f"acc: {acc_test_bow}, f1: {f1_test_bow}")
    print("valid set evaluation BOW")
    print(f"acc: {acc_valid_bow}, f1: {f1_valid_bow}")

    print("test set evaluation TFIDF")
    print(f"acc: {acc_test_tfidf}, f1: {f1_test_tfidf}")
    print("valid set evaluation TFIDF")
    print(f"acc: {acc_valid_tfidf}, f1: {f1_valid_tfidf}")


if __name__ == '__main__':
    main()

import click
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from data_helpers import *
import roberta
import svm
import textcnn


@click.command()
@click.option('--datapath', default='../../data/processed/labels.csv', help='path to the labels.')
def eval(datapath):
    # load data
    df = load_labels(datapath)
    categories = df.category.unique()
    training_cat, testing_cat = train_test_split(categories, test_size=0.5)
    train_df, test_df = split_dataframe_on_categories(df,training_cat)
    test_labels = test_df['need'].tolist()

    # train text-cnn
    data, max_sentence_len = textcnn.createData(df)
    data_train, _ = textcnn.createData(train_df)
    data_test, _ = textcnn.createData(test_df)
    w2i, i2w, vocab_size = textcnn.createVocab(data)
    model = textcnn.train2(data_train, max_sentence_len, w2i, vocab_size)
    prediction = textcnn.predict(model, data_test, w2i, 3)
    cnn_f1 = f1_score(test_labels, prediction)

    #train svm
    svm_model, vectorizer = svm.train(train_df)
    svm_pred = svm.predict(test_df, svm_model, vectorizer)

    svm_f1 = f1_score(test_labels, svm_pred)
    print(svm_f1)


    # train roberta model
    roberta_model = roberta.train(train_df)
    roberta_pred = roberta.predict(test_df['sentece'].tolist(), roberta_model)

    roberta_f1 = f1_score(test_labels,roberta_pred)
    print("results f1")
    print(cnn_f1)
    print(svm_f1)
    print(roberta_f1)




if __name__ == '__main__':
    eval()
import click
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from data_helpers import *
import roberta
import svm
import textcnn

def eval_text_cnn(train_df, test_df):
    test_labels = test_df['labels'].tolist()
    # train text-cnn
    data_train, max_sentence_len = textcnn.createData(train_df)
    data_test, _ = textcnn.createData(test_df)
    w2i, i2w, vocab_size = textcnn.createVocab(data_train)
    model = textcnn.train(data_train, max_sentence_len, w2i, vocab_size)
    prediction = textcnn.predict(model, data_test, w2i, 3)
    f1 = f1_score(test_labels, prediction)
    return f1

def eval_roberta(train_df, test_df):
    test_labels = test_df['labels'].tolist()
    # train roberta model
    roberta_model = roberta.train(train_df)
    roberta_pred = roberta.predict(test_df['text'].tolist(), roberta_model)

    f1 = f1_score(test_labels, roberta_pred)
    return f1

def eval_SVM(train_df, test_df):
    test_labels = test_df['labels'].tolist()
    # train svm
    svm_model, vectorizer = svm.train(train_df)
    svm_pred = svm.predict(test_df, svm_model, vectorizer)

    f1 = f1_score(test_labels, svm_pred)
    return f1


@click.command()
@click.option('--datapath', default='../../data/processed/labels.csv', help='path to the labels.')
def eval(datapath):
    # load data
    df = load_labels(datapath)
    df = remove_row_based_on_category(df,'Baby')
    df_in_cat_train, df_in_cat_test = load_in_cat_labels('../../data/processed/baby-train-labels.csv', '../../data/processed/baby-test-labels.csv')

    train_df, test_df = train_test_split(df, train_size=2000, stratify=df[['category']])
    test_labels = test_df['labels'].tolist()

    # train roberta model
    roberta_f1_1 = eval_roberta(train_df, df_in_cat_test)
    roberta_f1_2 = eval_roberta(df_in_cat_train, df_in_cat_test)
    roberta_f1_3 = eval_roberta(df, df_in_cat_test)
    print(roberta_f1_1)
    print(roberta_f1_2)
    print(roberta_f1_3)

    # train text-cnn
    cnn_f1_1 = eval_text_cnn(train_df, df_in_cat_test)
    cnn_f1_2 = eval_text_cnn(df_in_cat_train, df_in_cat_test)
    print(cnn_f1_1)
    print(cnn_f1_2)

    #train svm
    svm_f1_1  =eval_SVM(train_df, df_in_cat_test)
    svm_f1_2  =eval_SVM(df_in_cat_train, df_in_cat_test)
    print(svm_f1_1)
    print(svm_f1_2)

    print("---results cnn---")
    print(cnn_f1_1)
    print(cnn_f1_2)
    print("---results svm---")
    print(svm_f1_1)
    print(svm_f1_2)
    print("---results roberta---")
    print(roberta_f1_1)
    print(roberta_f1_2)
    print(roberta_f1_3)

if __name__ == '__main__':
    eval()

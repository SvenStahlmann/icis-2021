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

def eval(datapath):
    # load data
    df = load_labels(datapath)
    df = remove_row_based_on_category(df,'Baby')
    df_in_cat_train, df_in_cat_test = load_in_cat_labels('../../data/processed/baby-train-labels.csv',
                                                         '../../data/processed/baby-test-labels.csv')

    train_df, test_df = train_test_split(df, train_size=2000, stratify=df[['category']])

    # train svm
    svm_f1_1 = eval_SVM(train_df, df_in_cat_test)
    svm_f1_2 = eval_SVM(df_in_cat_train, df_in_cat_test)
    svm_f1_3 = eval_SVM(df, df_in_cat_test)

    # train text-cnn
    cnn_f1_1 = eval_text_cnn(train_df, df_in_cat_test)
    cnn_f1_2 = eval_text_cnn(df_in_cat_train, df_in_cat_test)
    cnn_f1_3 = eval_text_cnn(df, df_in_cat_test)

    # train roberta model
    roberta_f1_1 = eval_roberta(train_df, df_in_cat_test)
    roberta_f1_2 = eval_roberta(df_in_cat_train, df_in_cat_test)
    roberta_f1_3 = eval_roberta(df, df_in_cat_test)

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

    return roberta_f1_1, roberta_f1_2, roberta_f1_3, cnn_f1_1, cnn_f1_2, cnn_f1_3, svm_f1_1, svm_f1_2, svm_f1_3


def main():
    number_of_runs = 5
    roberta_1_results = []
    roberta_2_results = []
    roberta_3_results = []
    cnn_1_results = []
    cnn_2_results = []
    cnn_3_results = []
    svm_1_results = []
    svm_2_results = []
    svm_3_results = []
    for i in range(number_of_runs):
        print("Run: " + str(i))
        roberta_f1_1, roberta_f1_2, roberta_f1_3, cnn_f1_1, cnn_f1_2, cnn_f1_3, svm_f1_1, svm_f1_2, svm_f1_3 = eval('../../data/processed/labels.csv')
        roberta_1_results.append(roberta_f1_1)
        roberta_2_results.append(roberta_f1_2)
        roberta_3_results.append(roberta_f1_3)
        cnn_1_results.append(cnn_f1_1)
        cnn_2_results.append(cnn_f1_2)
        cnn_3_results.append(cnn_f1_3)
        svm_1_results.append(svm_f1_1)
        svm_2_results.append(svm_f1_2)
        svm_3_results.append(svm_f1_3)
        print("End Run: " + str(i))

    roberta_1_average = sum(roberta_1_results) / len(roberta_1_results)
    roberta_2_average = sum(roberta_2_results) / len(roberta_2_results)
    roberta_3_average = sum(roberta_3_results) / len(roberta_3_results)
    cnn_1_average = sum(cnn_1_results) / len(cnn_1_results)
    cnn_2_average = sum(cnn_2_results) / len(cnn_2_results)
    cnn_3_average = sum(cnn_3_results) / len(cnn_3_results)
    svm_1_average = sum(svm_1_results) / len(svm_1_results)
    svm_2_average = sum(svm_2_results) / len(svm_2_results)
    svm_3_average = sum(svm_3_results) / len(svm_3_results)

    d = {'name': ['roberta-1', 'roberta-2', 'roberta-3', 'cnn-1', 'cnn-2', 'cnn-3', 'svm-1', 'svm-2', 'svm-3'],
         'f1-average': [roberta_1_average, roberta_2_average, roberta_3_average,
                        cnn_1_average, cnn_2_average, cnn_3_average,
                        svm_1_average, svm_2_average, svm_3_average]}
    df_results = pd.DataFrame(data=d)
    print(df_results)
    df_results.to_csv('../../reports/results-5-avg.csv', index=False)

if __name__ == '__main__':
    main()

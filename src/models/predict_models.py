from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
from nltk.corpus import stopwords
import nltk
from data_helpers import *
import roberta
import svm
import textcnn
import naivebayes

def eval_text_cnn(train_df, test_df):
    test_labels = test_df['labels'].tolist()
    # train text-cnn
    data_train, max_sentence_len = textcnn.createData(train_df)
    data_test, _ = textcnn.createData(test_df)
    w2i, i2w, vocab_size = textcnn.createVocab(data_train)
    model = textcnn.train(data_train, max_sentence_len, w2i, vocab_size)
    prediction = textcnn.predict(model, data_test, w2i, 5)
    f1 = f1_score(test_labels, prediction)
    return f1

def eval_roberta(train_df, test_df):
    test_labels = test_df['labels'].tolist()
    # train roberta model
    roberta_model = roberta.train(train_df)
    roberta_pred = roberta.predict(test_df['text'].tolist(), roberta_model)

    f1 = f1_score(test_labels, roberta_pred)
    return f1

def eval_SVM(train_df, test_df, stop_words = True):
    if stop_words:
        #nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    else:
        stop_words= None
    test_labels = test_df['labels'].tolist()
    # train svm
    svm_model, vectorizer = svm.train(train_df, stop_words=stop_words)
    svm_pred = svm.predict(test_df, svm_model, vectorizer)

    f1 = f1_score(test_labels, svm_pred)
    return f1

def eval_NB(train_df, test_df, stop_words = True):
    if stop_words:
        #nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    else:
        stop_words= None

    test_labels = test_df['labels'].tolist()
    # train svm
    svm_model, vectorizer = naivebayes.train(train_df, stop_words=stop_words)
    svm_pred = naivebayes.predict(test_df, svm_model, vectorizer)

    f1 = f1_score(test_labels, svm_pred)
    return f1

def eval(train_df_small, train_df_big ,df_in_cat_train, df_in_cat_test):
    result_dict = {}

    # train svm
    result_dict['svm_f1_1'] = eval_SVM(train_df_small, df_in_cat_test)
    result_dict['svm_f1_2'] = eval_SVM(df_in_cat_train, df_in_cat_test)
    result_dict['svm_f1_3'] = eval_SVM(train_df_big, df_in_cat_test)

    # train naive bayes
    result_dict['nb_f1_1'] = eval_NB(train_df_small, df_in_cat_test)
    result_dict['nb_f1_2'] = eval_NB(df_in_cat_train, df_in_cat_test)
    result_dict['nb_f1_3'] = eval_NB(train_df_big, df_in_cat_test)

    # train text-cnn
    result_dict['cnn_f1_1'] = eval_text_cnn(train_df_small, df_in_cat_test)
    result_dict['cnn_f1_2'] = eval_text_cnn(df_in_cat_train, df_in_cat_test)
    result_dict['cnn_f1_3'] = eval_text_cnn(train_df_big, df_in_cat_test)

    # train roberta model
    result_dict['roberta_f1_1'] = eval_roberta(train_df_small, df_in_cat_test)
    result_dict['roberta_f1_2'] = eval_roberta(df_in_cat_train, df_in_cat_test)
    result_dict['roberta_f1_3'] = eval_roberta(train_df_big, df_in_cat_test)

    print("---results---")
    print(result_dict)

    return result_dict


def main():
    number_of_folds = 3

    df = load_labels('../../data/processed/labels.csv')
    df = remove_row_based_on_category(df,'Baby')
    df_in_cat = load_labels('../../data/processed/baby-labels.csv')
    df_small, _ = train_test_split(df, train_size=1800, stratify=df[['category']])

    kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=2)
    df_in_cat_splits = list(kf.split(df_in_cat))


    for i in range(number_of_folds):
        print("Run: " + str(i))

        df_in_cat_train = df_in_cat.iloc[df_in_cat_splits[i][0]]
        df_in_cat_test = df_in_cat.iloc[df_in_cat_splits[i][1]]

        run_results = eval(df_small,df,df_in_cat_train,df_in_cat_test)
        if i == 0:
            #create dataframe
            df_results = pd.DataFrame(data=[run_results])
        else:
            df_results = df_results.append(run_results, ignore_index=True)
        print("End Run: " + str(i))

    print(df_results)
    print('---mean---')
    print(df_results.mean())
    df_results.to_csv('../../reports/results-5-cv.csv', index=False)

if __name__ == '__main__':
    main()

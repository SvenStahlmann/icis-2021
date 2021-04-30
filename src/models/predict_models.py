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


def eval_out_of_cat(df_heterogeneous ,df_homogeneous, categories):
    df_results = None
    for category in categories:
        print(category)
        test, train_heterogeneous_big = split_dataframe_on_categories(df_heterogeneous,[category])
        train_heterogeneous_small, _ = train_test_split(train_heterogeneous_big, train_size=len(df_homogeneous),
                                                        stratify=train_heterogeneous_big[['category']])
        result_dict = {'category': category}
        # train svm
        result_dict['svm_f1_1'] = eval_SVM(train_heterogeneous_small, test)
        result_dict['svm_f1_2'] = eval_SVM(df_homogeneous, test)
        result_dict['svm_f1_3'] = eval_SVM(train_heterogeneous_big, test)

        # train naive bayes
        result_dict['nb_f1_1'] = eval_NB(train_heterogeneous_small, test)
        result_dict['nb_f1_2'] = eval_NB(df_homogeneous, test)
        result_dict['nb_f1_3'] = eval_NB(train_heterogeneous_big, test)

        # train text-cnn
        result_dict['cnn_f1_1'] = eval_text_cnn(train_heterogeneous_small, test)
        result_dict['cnn_f1_2'] = eval_text_cnn(df_homogeneous, test)
        result_dict['cnn_f1_3'] = eval_text_cnn(train_heterogeneous_big, test)

        # train roberta model
        result_dict['roberta_f1_1'] = eval_roberta(train_heterogeneous_small, test)
        result_dict['roberta_f1_2'] = eval_roberta(df_homogeneous, test)
        result_dict['roberta_f1_3'] = eval_roberta(train_heterogeneous_big, test)

        if df_results is None:
            #create dataframe
            df_results = pd.DataFrame(data=[result_dict])
        else:
            df_results = df_results.append(result_dict, ignore_index=True)
            
        df_results.to_csv('../../reports/results-out-of-training.csv', index=False)

    df_results.to_csv('../../reports/results-out-of-training.csv', index=False)

def main():
    # eval when predictiing the baby category
    number_of_folds = 5

    df_heterogeneous = load_labels('../../data/processed/labels.csv')
    df_homogeneous = load_labels('../../data/processed/baby-labels.csv')

    kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=2)
    df_homogeneous_splits = list(kf.split(df_homogeneous))

    for i in range(number_of_folds):
        print("Run: " + str(i))

        df_homogeneous_train = df_homogeneous.iloc[df_homogeneous_splits[i][0]]
        df_homogeneous_test = df_homogeneous.iloc[df_homogeneous_splits[i][1]]

        df_heterogeneous_small, _ = train_test_split(df_heterogeneous, train_size=len(df_homogeneous_train),
                                                     stratify=df_heterogeneous[['category']])

        run_results = eval(df_heterogeneous_small,df_heterogeneous,df_homogeneous_train,df_homogeneous_test)
        if i == 0:
            #create dataframe
            df_results = pd.DataFrame(data=[run_results])
        else:
            df_results = df_results.append(run_results, ignore_index=True)

        print("End Run: " + str(i))

    print(df_results)
    print('---mean---')
    print(df_results.mean())
    df_results.to_csv('../../reports/results-5-cv-in-training.csv', index=False)

if __name__ == '__main__':
    main()
    #df_heterogeneous = load_labels('../../data/processed/labels.csv')
    #df_homogeneous = load_labels('../../data/processed/baby-labels.csv')
    #categories = list(df_heterogeneous['category'].unique())
    #categories.remove('Baby')
    #eval_out_of_cat(df_heterogeneous,df_homogeneous,categories)
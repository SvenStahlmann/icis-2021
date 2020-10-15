import click
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, accuracy_score


def load_validation_data(valid_in_cat_path, valid_out_of_cat_path):
    valid_in_cat = pd.read_csv(valid_in_cat_path, delimiter=',')
    valid_out_of_cat = pd.read_csv(valid_out_of_cat_path, delimiter=',')

    return valid_in_cat, valid_out_of_cat


def load_fold_data(path, current_fold):
    fold_train = pd.read_csv(path + 'fold-' + str(current_fold) + '-train.csv', delimiter=',')
    fold_test = pd.read_csv(path + 'fold-' + str(current_fold) + '-test.csv', delimiter=',')

    return fold_train, fold_test


def prediction_svm(train, test, valid, report_df, current_fold, categories, use_tfidf=False):
    name = 'fold-' + str(current_fold)
    x_train = train['text']
    y_train = train['labels']
    x_test = test['text']
    y_test = test['labels']
    x_valid = valid['text']
    y_valid = valid['labels']

    if use_tfidf:
        vectorizer = TfidfVectorizer(max_features=2000)
    else:
        vectorizer = CountVectorizer(max_features=2000)

    vectorizer.fit(x_train)

    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_valid = vectorizer.transform(x_valid)

    model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    model.fit(x_train, y_train)

    prediction_svm_test = model.predict(x_test)
    prediction_svm_valid = model.predict(x_valid)

    acc_test = accuracy_score(prediction_svm_test, y_test) * 100
    acc_valid = accuracy_score(prediction_svm_valid, y_valid) * 100
    f1_test = f1_score(prediction_svm_test, y_test) * 100
    f1_valid = f1_score(prediction_svm_valid, y_valid) * 100

    # per category prediction
    cat_dict = {}
    for category in categories:
        mask = valid['category'] == category
        category_df = valid[mask]
        cat_x_valid = category_df['text']
        cat_x_valid = vectorizer.transform(cat_x_valid)
        cat_y_valid = category_df['labels']
        cat_prediction = model.predict(cat_x_valid)
        cat_acc_valid = accuracy_score(cat_prediction, cat_y_valid) * 100
        cat_f1_valid = f1_score(cat_prediction, cat_y_valid) * 100
        cat_dict[category + "-f1"] = cat_f1_valid
        cat_dict[category + "-acc"] = cat_acc_valid

    report_df.append([name, acc_test, f1_test, acc_valid, f1_valid] + list(cat_dict.values()))

    # return category columns
    return list(cat_dict.keys())


@click.command()
@click.option('--path', default='../../data/processed/', help='path to the training data.')
@click.option('--valid_in_cat_path', default='../../data/processed/in-cat-test.csv', help='path to the training data.')
@click.option('--valid_out_of_cat_path', default='../../data/processed/out-of-cat-valid.csv',
              help='path to the training data.')
def main(path, valid_in_cat_path, valid_out_of_cat_path):
    report_df_bow = []
    report_df_tfidf = []
    valid_in_cat, valid_out_of_cat = load_validation_data(valid_in_cat_path, valid_out_of_cat_path)
    categories = valid_out_of_cat.category.unique()

    print("validation data loaded")
    print(f"in cat: {len(valid_in_cat)}, out of cat: {len(valid_out_of_cat)}")
    print("starting fold training")
    for i in range(10):
        name = 'fold-' + str(i)
        print("working on " + name)
        train, test = load_fold_data(path, i)
        # BOW
        prediction_svm(train, valid_in_cat, valid_out_of_cat, report_df_bow, i, categories)
        # TFIDF
        category_columns = prediction_svm(train, valid_in_cat, valid_out_of_cat, report_df_tfidf, i, categories, True)
        print("done with " + name)

    report_df_bow = pd.DataFrame(report_df_bow,
                                 columns=['name', 'acc_in_cat', 'f1_in_cat', 'acc_out_of_cat',
                                          'f1_out_of_cat'] + category_columns)
    report_df_tfidf = pd.DataFrame(report_df_tfidf,
                                   columns=['name', 'acc_in_cat', 'f1_in_cat', 'acc_out_of_cat',
                                            'f1_out_of_cat'] + category_columns)

    print("BOW evaluation in cat")
    print(f"acc: {report_df_bow['acc_in_cat'].mean()}, f1: {report_df_bow['f1_in_cat'].mean()}")
    print("BOW evaluation out of cat")
    print(f"acc: {report_df_bow['acc_out_of_cat'].mean()}, f1: {report_df_bow['f1_out_of_cat'].mean()}")

    print("TFIDF evaluation in cat")
    print(f"acc: {report_df_tfidf['acc_in_cat'].mean()}, f1: {report_df_tfidf['f1_in_cat'].mean()}")
    print("TFIDF evaluation out of cat")
    print(f"acc: {report_df_tfidf['acc_out_of_cat'].mean()}, f1: {report_df_tfidf['f1_out_of_cat'].mean()}")

    report_df_bow.to_csv('../../reports/SVM-bow.csv', index=False)
    report_df_tfidf.to_csv('../../reports/SVM-tfidf.csv', index=False)


if __name__ == '__main__':
    main()

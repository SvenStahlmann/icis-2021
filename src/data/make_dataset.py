# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
import pandas as pd

def kfoldize(df, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True)
    train_folds = []
    test_folds = []
    for train_index, test_index in kf.split(df):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        train_folds.append(train_df)
        test_folds.append(test_df)
    return train_folds, test_folds

def load_in_cat_data(in_category_data_path):
    # load in category data
    names = ['text', 'Rating', 'Title', 'Purchase type', 'sentenceID', 'reviewID', 'labels']
    in_cat_data = pd.read_csv(in_category_data_path, delimiter=',', header=None, names=names)
    # drop unused columns from the dataframe
    in_cat_data = in_cat_data.filter(['labels', 'text'])
    #change labels
    in_cat_data['labels'] = in_cat_data['labels'].replace(['n', 'esc'], 0)
    in_cat_data['labels'] = in_cat_data['labels'].replace(['y'], 1)
    
    in_cat_data = preprocess_dataframe(in_cat_data)

    # split into train in test in category
    train_in_cat, test_in_cat = train_test_split(in_cat_data, test_size=0.1,stratify=in_cat_data[['labels']])
    print(train_in_cat['labels'].mean())
    print(test_in_cat['labels'].mean())
    train_folds, test_folds = kfoldize(train_in_cat)

    return train_folds,test_folds, test_in_cat

def load_out_of_cat_data(out_category_data_path):
    out_of_cat_data = pd.read_csv(out_category_data_path, index_col=False)
    # drop unused columns from the dataframe
    out_of_cat_data = out_of_cat_data.filter(['labels', 'text', 'category'])
    #change labels
    out_of_cat_data['labels'] = out_of_cat_data['labels'].replace(['n', 'esc'], 0)
    out_of_cat_data['labels'] = out_of_cat_data['labels'].replace(['y'], 1)
    
    out_of_cat_data = preprocess_dataframe(out_of_cat_data)

    return out_of_cat_data

def preprocess_dataframe(df):
    #lower
    df["text"] = df.text.apply(lambda x: str.lower(x))
    #keep only text
    df['text'] = df['text'].str.replace(r"[^a-zA-Z ]",'')
    df = df[df['text'].str.strip().astype(bool)]
    
    return df


@click.command()
@click.argument('in_category_data_path', default='../../data/raw/needs-kindersitz-labeled.csv', type=click.Path(exists=True))
@click.argument('out_category_data_path', default='../../data/raw/validation-sample-labeled.csv', type=click.Path(exists=True))
@click.argument('output_path', default='../../data/processed/', type=click.Path(exists=True))
def main(in_category_data_path, out_category_data_path, output_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #generate training/test folds and test data
    train_folds,test_folds, test_in_cat = load_in_cat_data(in_category_data_path)

    # load out of category data
    out_of_cat_data = load_out_of_cat_data(out_category_data_path)
    for i, (train, test) in enumerate(zip(train_folds, test_folds)):
        print("working on fold " +str(i))
        train.to_csv(output_path+"fold-" + str(i) +"-train.csv", index=False)
        test.to_csv(output_path+"fold-" + str(i)+"-test.csv", index=False)

    test_in_cat.to_csv(output_path+"in-cat-test.csv",index=False)
    out_of_cat_data.to_csv(output_path+"out-of-cat-valid.csv",index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()

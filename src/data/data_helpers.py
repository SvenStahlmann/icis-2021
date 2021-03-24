import pandas as pd
from sklearn.model_selection import train_test_split

def select_columns(df):
    df['need']= df['need'].replace({'y': '1', 'n': '0', 'X': 'x', '2': 'x'})
    df = df.rename(columns={"sentece": "text", "need": "labels"})
    return df[["text", "labels","category"]]

def create_labels():
    # load dataframes
    df = select_columns(pd.read_csv('../../data/raw/amazon-reviews4.csv', delimiter=';', index_col=0))
    df_5000_6200 = select_columns(pd.read_csv('../../data/raw/df_5000_6200-comma-RG-Tags.csv', delimiter=';'))
    df_6000_7200 = select_columns(pd.read_csv('../../data/raw/df_6000_7200_OR-Tags.csv', delimiter=';'))
    df_7000_8000 = select_columns(pd.read_csv('../../data/raw/sören_amz_reviews7000-800-tagged.csv', delimiter=','))
    # concatenate the different dataframes to create the whole labelled data
    df_processed = df.iloc[0:5000].append(df_5000_6200.iloc[0:1000]).append(df_6000_7200.iloc[0:999]).append(df_7000_8000)

    print(df_processed['labels'].value_counts())
    print(len(df_processed))

    df_processed.to_csv('../../data/processed/labels.csv', index=False)

def create_in_cat_labels():
    # load dataframes
    df = pd.read_csv('../../data/raw/needs-kindersitz-labeled.csv', names=['text','rating','title','purchase','id','id2','labels'])
    df['labels'] = df['labels'].replace({'y': '1', 'n': '0', 'X': 'x', '2': 'x', 'esc': 'x'})
    df = df[["text", "labels"]]
    train_df, test_df = train_test_split(df, train_size=0.8)
    df.to_csv('../../data/processed/baby-labels.csv', index=False)
    train_df.to_csv('../../data/processed/baby-train-labels.csv', index=False)
    test_df.to_csv('../../data/processed/baby-test-labels.csv', index=False)


if __name__ == '__main__':
    create_in_cat_labels()
    create_labels()










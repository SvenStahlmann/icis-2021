import pandas as pd

def select_columns(df1):
    df1['need']= df1['need'].replace({'y': '1', 'n': '0', 'X': 'x', '2': 'x'})
    return df1[["sentece", "need","category"]]

def create_labels():
    # load dataframes
    df = select_columns(pd.read_csv('../../data/raw/amazon-reviews4.csv', delimiter=';', index_col=0))
    df_5000_6200 = select_columns(pd.read_csv('../../data/raw/df_5000_6200-comma-RG-Tags.csv', delimiter=';'))
    df_6000_7200 = select_columns(pd.read_csv('../../data/raw/df_6000_7200_OR-Tags.csv', delimiter=';'))
    df_7000_8000 = select_columns(pd.read_csv('../../data/raw/sören_amz_reviews7000-800-tagged.csv', delimiter=','))
    # concatenate the different dataframes to create the whole labelled data
    df_processed = df.iloc[0:5000].append(df_5000_6200.iloc[0:1000]).append(df_6000_7200.iloc[0:999]).append(df_7000_8000)

    print(df_processed['need'].value_counts())
    print(len(df_processed))

    df_processed.to_csv('../../data/processed/labels.csv', index=False)


if __name__ == '__main__':
    create_labels()










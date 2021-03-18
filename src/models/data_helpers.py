import pandas as pd

def split_dataframe_on_categories(df, categories):
        categories = df['category'].isin(categories)
        withCategories = df[categories]
        withOutCategories = df[~categories]
        return withCategories, withOutCategories

def remove_non_numeric_rows(df,columnName):
        df_clean = df[df[columnName].str.isnumeric()]
        return df_clean

def remove_row_based_on_category(df,category):
        df = df[df.category != category]
        return df

def load_labels(path):
        df = pd.read_csv(path)
        df = remove_non_numeric_rows(df, 'labels')
        df["labels"] = pd.to_numeric(df["labels"])
        return df

def load_in_cat_labels(train_path, test_path):
        df_train = pd.read_csv(train_path)
        df_train = remove_non_numeric_rows(df_train, 'labels')
        df_train["labels"] = pd.to_numeric(df_train["labels"])
        df_test = pd.read_csv(test_path)
        df_test = remove_non_numeric_rows(df_test, 'labels')
        df_test["labels"] = pd.to_numeric(df_test["labels"])

        return df_train, df_test















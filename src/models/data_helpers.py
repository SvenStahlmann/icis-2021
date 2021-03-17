import pandas as pd

def split_dataframe_on_categories(df, categories):
        categories = df['category'].isin(categories)
        withCategories = df[categories]
        withOutCategories = df[~categories]
        return withCategories, withOutCategories

def remove_non_numeric_rows(df,columnName):
        df_clean = df[df[columnName].str.isnumeric()]
        return df_clean

def load_labels(path):
        df = pd.read_csv(path)
        df = remove_non_numeric_rows(df, 'need')
        df["need"] = pd.to_numeric(df["need"])
        return df














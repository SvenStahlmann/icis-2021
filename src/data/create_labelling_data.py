import pandas as pd

def get_subset_of_data(df,start,end):
    return df.iloc[start:end]


df = pd.read_csv('../../data/raw/amazon-reviews4.csv', delimiter=';', index_col=0)

df_5000_6200 = get_subset_of_data(df,5000,6200)

df_6000_7200 = get_subset_of_data(df,6000,7200)

print(df_5000_6200.head())

print(df_6000_7200.head())

sep = ';'

df_5000_6200.to_csv('../../data/raw/df_5000_6200-semicolon.csv', index=False, sep = ';')
df_5000_6200.to_csv('../../data/raw/df_5000_6200-comma-RG-Tags.csv', index=False)

df_6000_7200.to_csv('../../data/raw/df_6000_7200-semicolon.csv', index=False, sep = ';')
df_6000_7200.to_csv('../../data/raw/df_6000_7200-comma.csv', index=False)




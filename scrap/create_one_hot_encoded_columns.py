import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('./data/full_training_set.csv')

mlb = MultiLabelBinarizer()

df_judges = pd.DataFrame(mlb.fit_transform(df[['href', 'judges']]['judges'].apply(lambda x: x.split(';'))),columns=mlb.classes_, index=df.href)

df = pd.merge(df, df_judges, how='inner', on='href')

df_lower_court = pd.get_dummies(df[['href', 'lower_court']], columns=['lower_court', ])

df = pd.merge(df, df_lower_court, how='inner', on='href')

df.to_csv('./data/full_one_hot_encoded.csv')
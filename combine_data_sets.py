import pandas as pd

left_table = pd.read_csv('./data/entity_df.csv')
right_table = pd.read_csv('./data/oyez_judges_advocates_question_conclusion.csv')

joined_table = pd.merge(left_table, right_table, on='href')

joined_table.to_csv('./data/full_training_set.csv')
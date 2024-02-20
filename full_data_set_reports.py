import pandas as pd
import sqlite3 as db
import requests
import numpy as np
from ydata_profiling import ProfileReport
import sweetviz as sv
# from autoviz.AutoViz_Class import AutoViz_Class

raw_df = pd.read_csv('for_training.csv')
raw_df.info()

# raw_df['fpw'] = raw_df['first_party_winner'].map({'True': 1,'False' :0 })
# min_df = raw_df[['href', 'fpw', 'issue_area']]
# sim1_facts_df = pd.merge(raw_df, min_df, how="left", suffixes=(None, '_sim1_facts'), left_on='sim1_facts_href', right_on='href')
# sim2_facts_df = pd.merge(sim1_facts_df, min_df, how="left", suffixes=(None, '_sim2_facts'), left_on='sim2_facts_href', right_on='href')

# sim1_issue_area_df = pd.merge(sim2_facts_df, min_df, how="left", suffixes=(None, '_sim1_issue_area'), left_on='sim1_issue_area_href', right_on='href')
# sim2_issue_area_df = pd.merge(sim1_issue_area_df, min_df, how="left", suffixes=(None, '_sim2_issue_area'), left_on='sim2_issue_area_href', right_on='href')

# sim1_legal_question_df = pd.merge(raw_df, min_df, how="left", suffixes=(None, '_sim1_legal_question'), left_on='sim1_legal_question_href', right_on='href')

# sim1_conclusion_df = pd.merge(sim1_legal_question_df, min_df, how="left", suffixes=(None, '_sim1_conclusion'), left_on='sim1_conclusion_href', right_on='href')

# merged_df = sim1_conclusion_df

# print(merged_df.columns)
# merged_df = merged_df[merged_df.columns.drop(list(merged_df.filter(regex='Unnamed')))]
# merged_df = merged_df[merged_df.columns.drop(list(merged_df.filter(regex='href')))]
# merged_df = merged_df.drop('ID', axis=1)
# merged_df = merged_df.drop('name', axis=1)
# merged_df = merged_df.drop('docket', axis=1)
# merged_df = merged_df.drop('facts', axis=1)
# merged_df = merged_df.drop('facts_len', axis=1)
# merged_df = merged_df.drop('majority_vote', axis=1)
# merged_df = merged_df.drop('minority_vote', axis=1)
# merged_df = merged_df.drop('disposition', axis=1)
# merged_df = merged_df.drop('decision_type', axis=1)
# merged_df = merged_df.drop('first_party_winner', axis=1)

# merged_df['fpw'] = merged_df[merged_df['fpw'].notna()]['fpw'].astype(int)

# joining_df = raw_df[['href']]
# raw_df["first_party_winner_bin"] = raw_df["first_party_winner"].astype(int)

profile = ProfileReport(raw_df, title="FV Train Set Report")
profile.to_file("train_set_report.html")


# sw_df = merged_df[merged_df['issue_area'].notna()]

# issue_area = sv.analyze(source=sw_df,
#             target_feat='issue_area',
#             pairwise_analysis = 'auto')

# issue_area.show_html(  filepath='issue_area_report.html', 
#             open_browser=True, 
#             layout='widescreen', 
#             scale=None)
# AV = AutoViz_Class()
# df = AV.AutoViz("filevine_case_study_justice - train set_source data set.csv")

# sw_df = raw_df[raw_df['fpw'].notna()]

fpw = sv.analyze(source=raw_df,
            target_feat='fpw',
            pairwise_analysis = 'on')

fpw.show_html( filepath='fpw_report.html', 
            open_browser=True, 
            layout='widescreen', 
            scale=None)
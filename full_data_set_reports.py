import pandas as pd
import sqlite3 as db
import requests
import numpy as np
from ydata_profiling import ProfileReport
import sweetviz as sv
# from autoviz.AutoViz_Class import AutoViz_Class

raw_df = pd.read_csv('./data/full_one_hot_encoded.csv')
raw_df.info()

# profile = ProfileReport(raw_df, title="FV Train Set Report")
# profile.to_file("train_set_report.html")


# sw_df = raw_df[raw_df['issue_area'].notna()]

# issue_area = sv.analyze(source=sw_df,
#             target_feat='issue_area',
#             pairwise_analysis = 'auto')

# issue_area.show_html(  filepath='issue_area_report.html', 
#             open_browser=True, 
#             layout='widescreen', 
#             scale=None)
# AV = AutoViz_Class()
# df = AV.AutoViz("filevine_case_study_justice - train set_source data set.csv")

sw_df = raw_df[raw_df['first_party_winner'].notna()]

fpw = sv.analyze(source=sw_df,
            target_feat='first_party_winner',
            pairwise_analysis = 'on')

fpw.show_html( filepath='fpw_report.html', 
            open_browser=True, 
            layout='widescreen', 
            scale=None)
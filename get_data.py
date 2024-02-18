
import pandas as pd
from pandas import json_normalize
import requests
import sqlite3
from tqdm import tqdm
import csv


urls = pd.read_csv('../data/filevine_case_study_justice - train set_source data set.csv')['href']

oyez_df = pd.DataFrame()
bad_urls = list()
for url in tqdm(urls[0]):
    try:
        response = requests.get(url)
        row = json_normalize(response.json())
        oyez_df = pd.concat([oyez_df, row], ignore_index=True)
        del row
    except:
        print(f"error occured with {url}")
        bad_urls.append(url)
        continue

# conn = sqlite3.connect('filevine_casestudy.db')
# oyez_df.to_sql('oyez', conn, if_exists='replace')

oyez_df.to_csv('oyez_one.csv')

with open('bad_urls_one.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(bad_urls)


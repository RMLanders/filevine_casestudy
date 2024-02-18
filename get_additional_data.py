
import pandas as pd
from pandas import json_normalize
import requests
import sqlite3
from tqdm import tqdm
import csv
import re

def get_legal_question(json_data):
    if not json_data["question"]:
        question = "Legal Question Not Present"
    else: 
        question = json_data["question"]
    return question

def get_conclusion(json_data):
    if not json_data["conclusion"]:
        court_conclusion = "Conclusion Not Present"
    else: 
        court_conclusion = json_data["conclusion"]
    return court_conclusion

def get_judges(json_data):
    judge_list = []
    if (not json_data["decisions"] or not json_data["decisions"][0] or not json_data["decisions"][0]["votes"]):
        judge_list = ["no judges found"]
    else:
        number_of_judges = len(json_data["decisions"][0]["votes"])
        for person in range(number_of_judges):
            half = json_data["decisions"][0]["votes"][person]["member"]["name"]
            judge_list.append(half)
    return judge_list

def get_lower_court(json_data):
    if (not json_data["lower_court"] or not json_data["lower_court"]["name"]):
        lower_court = "Lower Court Not Present"
    else:
        lower_court = json_data["lower_court"]["name"]
    return lower_court

urls = pd.read_csv('./data/filevine_case_study_justice - train set_source data set.csv')['href']
additional_data_df = pd.DataFrame()
bad_urls = list()
for url in tqdm(urls):
    try:
        response = requests.get(url)
        res = response.json()
    except:
        print(f"error occured with {url}")
        bad_urls.append(url)
        continue
        
    judges = get_judges(res)
    legal_question = get_legal_question(res)
    conclusion = get_conclusion(res)
    lower_court = get_lower_court(res)
    
    new_row = pd.DataFrame({"href": url, "judges": judges, "lower_court": lower_court, "legal_question": legal_question, "conclusion": conclusion})
    additional_data_df = pd.concat([additional_data_df, new_row])    
    del new_row  

# conn = sqlite3.connect('filevine_casestudy.db')
# oyez_df.to_sql('oyez', conn, if_exists='replace')

additional_data_df.to_csv('./data/oyez_smaller.csv')

with open('bad_urls_one.csv', 'w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerows(bad_urls)

# # https://api.oyez.org/cases/1977/77-404
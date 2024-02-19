import pandas as pd
import csv
import re
from bs4 import BeautifulSoup
import marqo
import pprint
from tqdm import tqdm

file = './data/full_one_hot_encoded.csv'
df = pd.read_csv(file)

url = "http://localhost:8882"
mq = marqo.Client(url=url)

df = pd.read_csv('./data/full_drop.csv')

def clean_text(txt):
    if pd.isna(txt):
        clean_text = ""
    else:   
        soup = BeautifulSoup(str(txt).strip(), 'html.parser')
        clean_text = soup.get_text()
        clean_text = re.sub('[^a-zA-Z0-9\s]', '', clean_text)
        clean_text = clean_text.lower()
    return clean_text

df['facts_clean'] = df['facts'].apply(clean_text)
df['issue_area'] = df['issue_area'].apply(clean_text)
df['legal_question'] = df['legal_question'].apply(clean_text)
df['conclusion'] = df['conclusion'].apply(clean_text)

df['sim1_facts_score'] = None
df['sim1_issue_area_score'] = None
df['sim1_legal_question_score'] = None
df['sim1_conclusion_score'] = None

df['sim2_facts_score'] = None
df['sim2_issue_area_score'] = None

df['sim1_facts_href'] = None
df['sim2_facts_href'] = None

df['sim1_issue_area_href'] = None
df['sim2_issue_area_href'] = None

df['sim1_legal_question_href'] = None

df['sim1_conclusion_href'] = None

for index, row in tqdm(df.iterrows()):
    search_href = row['href']
    # facts = row['facts_clean']
    # results = mq.index("whole").search(facts, search_method='TENSOR')
    # results_df = pd.DataFrame(results['hits'])
    # same_doc_index = results_df[results_df['href']==search_href].index
    # results_df.drop(same_doc_index , inplace=True)
    # results_df[0:3]

    facts = row['facts_clean']
    # print("===============")
    # print(index)
    # print(facts)
    # print(search_href)
    # print("===============")
    results = mq.index("whole_legal").search(facts, search_method='TENSOR')
    results_df = pd.DataFrame(results['hits'])
    same_doc_index = results_df[results_df['href']==search_href].index
    results_df.drop(same_doc_index , inplace=True)
    results_df = results_df[0:3]
    # print(results_df[['href', '_score']])
    df.loc[df['href']==search_href, 'sim1_facts_score'] = results_df['_score'].iloc[0]
    df.loc[df['href']==search_href, 'sim2_facts_score'] = results_df['_score'].iloc[1]

    df.loc[df['href']==search_href, 'sim1_facts_href'] = results_df['href'].iloc[0]
    df.loc[df['href']==search_href, 'sim2_facts_href'] = results_df['href'].iloc[1]
    # print("===============")
    # print(results_df['_score'].iloc[0])
    # print(results_df['_score'].iloc[1])
    # print(results_df['_score'].iloc[2])
    # print("***********************")
    # print(df[df['href']==search_href]['sim1_facts_score'])
    # print(df[df['href']==search_href]['sim2_facts_score'])

    # print(df[df['href']==search_href]['sim1_facts_href'])
    # print(df[df['href']==search_href]['sim2_facts_href'])
    # print("===============")
    # print("===============")
    # print("===============")

    issue_area = row['issue_area']
    # print("===============")
    # print(index)
    # print(issue_area)
    # print(search_href)
    # print("===============")
    results = mq.index("whole_legal").search(issue_area, search_method='TENSOR')
    results_df = pd.DataFrame(results['hits'])
    same_doc_index = results_df[results_df['href']==search_href].index
    results_df.drop(same_doc_index , inplace=True)
    results_df = results_df[0:3]
    # print(results_df[['href', '_score']])
    df.loc[df['href']==search_href, 'sim1_issue_area_score'] = results_df['_score'].iloc[0]
    df.loc[df['href']==search_href, 'sim2_issue_area_score'] = results_df['_score'].iloc[1]

    df.loc[df['href']==search_href, 'sim1_issue_area_href'] = results_df['href'].iloc[0]
    df.loc[df['href']==search_href, 'sim2_issue_area_href'] = results_df['href'].iloc[1]
    # print("===============")
    # print(results_df['_score'].iloc[0])
    # print(results_df['_score'].iloc[1])
    # print("***********************")
    # print(df[df['href']==search_href]['sim1_issue_area_score'])
    # print(df[df['href']==search_href]['sim2_issue_area_score'])

    # print(df[df['href']==search_href]['sim1_issue_area_href'])
    # print(df[df['href']==search_href]['sim2_issue_area_href'])
    # print("===============")
    # print("===============")
    # print("===============")

    legal_question = row['legal_question']
    # print("===============")
    # print(index)
    # print(legal_question)
    # print(search_href)
    # print("===============")
    results = mq.index("whole_legal").search(legal_question, search_method='TENSOR')
    results_df = pd.DataFrame(results['hits'])
    same_doc_index = results_df[results_df['href']==search_href].index
    results_df.drop(same_doc_index , inplace=True)
    results_df = results_df[0:3]
    # print(results_df[['href', '_score']])
    df.loc[df['href']==search_href, 'sim1_legal_question_score'] = results_df['_score'].iloc[0]

    df.loc[df['href']==search_href, 'sim1_legal_question_href'] = results_df['href'].iloc[0]
    # print("===============")
    # print(results_df['_score'].iloc[0])
    # print("***********************")
    # print(df[df['href']==search_href]['sim1_legal_question_score'])

    # print(df[df['href']==search_href]['sim1_legal_question_href'])
    # print("===============")
    # print("===============")
    # print("===============")

    conclusion = row['conclusion']
    # print("===============")
    # print(index)
    # print(conclusion)
    # print(search_href)
    # print("===============")
    results = mq.index("whole_legal").search(conclusion, search_method='TENSOR')
    results_df = pd.DataFrame(results['hits'])
    same_doc_index = results_df[results_df['href']==search_href].index
    results_df.drop(same_doc_index , inplace=True)
    results_df = results_df[0:3]
    # print(results_df[['href', '_score']])
    df.loc[df['href']==search_href, 'sim1_conclusion_score'] = results_df['_score'].iloc[0]

    df.loc[df['href']==search_href, 'sim1_conclusion_href'] = results_df['href'].iloc[0]
    # print("===============")
    # print(results_df['_score'].iloc[0])
    # print("***********************")
    # print(df[df['href']==search_href]['sim1_conclusion_score'])

    # print(df[df['href']==search_href]['sim1_conclusion_href'])
    # print("===============")
    # print("===============")
    # print("===============")

df.to_csv('./data/final_script.csv')
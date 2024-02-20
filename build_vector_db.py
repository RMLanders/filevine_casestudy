import pandas as pd
import csv
import re
from bs4 import BeautifulSoup
import marqo
import pprint

#run in bash first
# docker pull marqoai/marqo:latest
# docker rm -f marqo
# docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest

# df = pd.read_csv('./data/full_training_set.csv')
df = pd.read_csv("./data/prediction/filevine_case_study_justice - prediction set (2).csv")
model = "hf/multilingual-e5-large"
url = "http://localhost:8882"


def find_tags(txt):
    soup = BeautifulSoup(txt, 'html.parser')
    tags =[tag.name for tag in soup.find_all()]
    return tags

def clean_text(txt):
    if pd.isna(txt):
        clean_text = "no text found"
    else:   
        soup = BeautifulSoup(str(txt).strip(), 'html.parser')
        clean_text = soup.get_text()
        clean_text = re.sub('[^a-zA-Z0-9\s]', '', clean_text)
        clean_text = clean_text.lower()
    return clean_text

def create_marqo_client(url):
    mq = marqo.Client(url=url)
    return mq

df['facts'] = df['facts'].apply(clean_text)
df['issue_area'] = df['issue_area'].apply(clean_text)
df['legal_question'] = df['legal_question'].apply(clean_text)
df['conclusion'] = df['conclusion'].apply(clean_text)
clean_df = df[['href', 'first_party_winner', 'facts', 'issue_area', 'legal_question', 'conclusion']].to_dict('records')

mq = create_marqo_client(url)

# mq.create_index("whole", model=model)
# mq.index("whole").add_documents(
#     clean_df,
#     tensor_fields=['facts_clean', 'issue_area', 'legal_question', 'conclusion'],
#     client_batch_size=50
# )
ind = 'train_test'
settings = {
    "textPreprocessing": {
        "splitLength": 2,
        "splitOverlap": 0,
        "splitMethod": "sentence",
    },
}
# mq.create_index(ind, 
#                 model=model,
#                 settings_dict=settings
# )
mq.index(ind).add_documents(
    clean_df,
    tensor_fields=['facts', 'issue_area', 'legal_question', 'conclusion'],
    client_batch_size=50
)
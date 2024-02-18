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

df = pd.read_csv('./data/filevine_case_study_justice - train set_source data set.csv')

def find_tags(txt):
    soup = BeautifulSoup(txt, 'html.parser')
    tags =[tag.name for tag in soup.find_all()]
    return tags

def clean_text(txt):
    soup = BeautifulSoup(txt.strip(), 'html.parser')
    clean_text = soup.get_text()
    clean_text = re.sub('[^a-zA-Z0-9\s]', '', clean_text)
    clean_text = clean_text.lower()
    return clean_text

df['facts'].apply(find_tags).to_csv('tags.csv')

df['facts_clean'] = df['facts'].apply(clean_text)

clean_df = df[['href', 'facts_clean', 'first_party_winner']].to_dict('records')

mq = marqo.Client(url="http://localhost:8882")

mq.create_index("filevine_docs_whole", model="hf/multilingual-e5-large")

mq.index("filevine_docs_whole").add_documents(
    clean_df,
    tensor_fields=["facts_clean"],
    client_batch_size=50,
    auto_refresh=False
)

mq = marqo.Client(url="http://localhost:8882")

settings = {
    "textPreprocessing": {
        "splitLength": 2,
        "splitOverlap": 0,
        "splitMethod": "sentence",
    },
}

mq.create_index("filevine_docs_sentences", 
                model="hf/multilingual-e5-large",
                settings_dict=settings
)

mq.index("filevine_docs_sentences").add_documents(
    clean_df,
    tensor_fields=["facts_clean"],
    client_batch_size=50
)
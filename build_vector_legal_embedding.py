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

df = pd.read_csv('./data/full_training_set.csv')
# model = "hf/multilingual-e5-large"
model = "nlpaueb/legal-bert-base-uncased"
url = "http://localhost:8882"

def find_tags(txt):
    soup = BeautifulSoup(txt, 'html.parser')
    tags =[tag.name for tag in soup.find_all()]
    return tags

def clean_text(txt):
    if pd.isna(txt):
        clean_text = "no text found"
    soup = BeautifulSoup(str(txt).strip(), 'html.parser')
    clean_text = soup.get_text()
    clean_text = re.sub('[^a-zA-Z0-9\s]', '', clean_text)
    clean_text = clean_text.lower()
    return clean_text

def create_marqo_client(url):
    mq = marqo.Client(url=url)
    return mq

df['facts_clean'] = df['facts'].apply(clean_text)
df['issue_area'] = df['issue_area'].apply(clean_text)
df['legal_question'] = df['legal_question'].apply(clean_text)
df['conclusion'] = df['conclusion'].apply(clean_text)
clean_df = df[['href', 'first_party_winner', 'facts_clean', 'issue_area', 'legal_question', 'conclusion']].to_dict('records')

# settings = {
#     "index_defaults": {
#         "treat_urls_and_pointers_as_images": True,
#         "model": "generic-clip-test-model-1",
#         "model_properties": {
#             "name": "ViT-B-32-quickgelu",
#             "dimensions": 512,
#             "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
#             "type": "open_clip",
#         },
#         "normalize_embeddings": True,
#     },
# }

# settings = {
#     "index_defaults": {
#         "model": "legal-bert-base-uncased",
#         "model_properties": {
#             "name": "nlpaueb/legal-bert-base-uncased",
#             "dimensions": 512,
#             "type": "hf",
#         },
#     },
# }
# mq = create_marqo_client(url)
# mq.create_index("whole_legal", model=model)
# mq.index("whole_legal").add_documents(
#     clean_df,
#     tensor_fields=['facts_clean', 'issue_area', 'legal_question', 'conclusion'],
#     client_batch_size=50
# )

settings = {
    "textPreprocessing": {
        "splitLength": 7,
        "splitOverlap": 0,
        "splitMethod": "sentence",
    },
    "index_defaults": {
        "model": "legal-bert-base-uncased",
        "model_properties": {
            "name": "nlpaueb/legal-bert-base-uncased",
            "dimensions": 512,
            "type": "hf",
        },
    },
}
mq = create_marqo_client(url)
mq.create_index("legal_sentences_7", 
                model=model,
                settings_dict=settings
)
mq.index("legal_sentences_7").add_documents(
    clean_df,
    tensor_fields=['facts_clean', 'issue_area', 'legal_question', 'conclusion'],
    client_batch_size=50
)
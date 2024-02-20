import spacy
import pandas as pd


nlp = spacy.load("en_core_web_trf")

def extract_entity(party):
    party = str(party).strip()
    entity = ''
    if not party or pd.isna(party):
        entity = "No parties found"
    doc = nlp(party)
    for ent in doc.ents:
        entity = ent.label_
    print(f"{party}, {entity}")
    return entity

# data = pd.read_csv('./data/filevine_case_study_justice - train set_source data set.csv')
data = pd.read_csv("./data/prediction/filevine_case_study_justice - prediction set (2).csv")
data['first_party_entity'] = data['first_party'].apply(lambda row: extract_entity(row))
data['second_party_entity'] = data['second_party'].apply(lambda row: extract_entity(row))

print(data)
data.to_csv('./data/prediction/prediction_entity_df.csv')
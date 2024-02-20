
import pandas as pd
from pandas import json_normalize
import requests
import sqlite3
from tqdm import tqdm
import csv

url = 'https://api.oyez.org/cases/1984/83-1919'
response = requests.get(url)
row = json_normalize(response.json())
row.to_csv('oyez_one.csv')


import re
import pandas as pd


'''
Preprocess the data
'''

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove trailing whitespace
    """
    s = s.lower()
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\"\(\)\\\/\!])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?\'\.]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\!\|•«\n])', ' ', s)
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Data preprocessing

def data_preprocessing(path):
    df = pd.read_csv(path, sep=',', encoding='utf-8')
    df = df[['Review_id','Commentaire', 'Note']]
    df = df.dropna()
    df = df[:1000]
    df.Commentaire = df.Commentaire.apply(lambda x:deEmojify(x))
    df.Commentaire = df.Commentaire.str.lower()
    df.Commentaire = df.Commentaire.apply(lambda x: text_preprocessing(x))
    # These are the labels
    labels_index ={'0,5':0, '1,0':1, '1,5':2, '2,0':3, '2,5':4, '3,0':5, '3,5':6, '4,0':7, '4,5':8, '5,0':9}
    df['Note'] = df['Note'].replace(labels_index)
    return df


# def test_data_process(path):
#     df = pd.read_csv(path, sep=',', encoding='utf-8')
#     df = df[['Review_id','Commentaire']]
#     df.Commentaire = df.Commentaire.fillna('vide')
#     df = df.dropna()
#     df = df[:1000]
#     df.Commentaire = df.Commentaire.apply(lambda x:deEmojify(x))
#     df.Commentaire = df.Commentaire.str.lower()
#     df.Commentaire = df.Commentaire.apply(lambda x: text_preprocessing(x))
#     # # These are the labels
#     # labels_index ={'0,5':0, '1,0':1, '1,5':2, '2,0':3, '2,5':4, '3,0':5, '3,5':6, '4,0':7, '4,5':8, '5,0':9}
#     # df['Note'] = df['Note'].replace(labels_index)
#     return df



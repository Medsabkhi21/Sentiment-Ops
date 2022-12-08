import re
import string
import pandas as pd
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer  
STOPWORDS = stopwords.words("english")
import pickle
import numpy as np

def prediction(x_input):
    
    x= clean_data(x_input)
  
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    input_df = pd.DataFrame({x:x}, index=[0])
    print("input_df     ",input_df)
    x_processed = vectorizer.transform(input_df)
    print("x_processed",x_processed)
    model = pickle.load(open('classifier.pkl', 'rb'))

    pred = model.predict(x_processed)
    print(pred)
    return pred

def clean_data( x):
        r""" Main cleaning function. 
        Here we are:
        - lowering text
        - remove stopwords 
        - tags and hashtags 
        - single quotes
        - new lines
        - punctuation
        """
        # lower text 
        x = x.lower()
        # remove stopwords
        x = [word for word in x.split() if not word in STOPWORDS]
        x = ' '.join(x)
        # remove words after @
        x = re.sub("@\S+ ", "", x) 
        # remove single quote
        x = re.sub("\'", '', x) 
        # remove new line
        x = re.sub('\\n', '', x)   
        # remove puncts 
        x = x.translate(str.maketrans('', '', string.punctuation))

        return x 

import argparse

import pickle
import pandas as pd
import sklearn
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score



from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from  sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


import numpy as np
import pandas as pd 
import json
import matplotlib.pyplot as plt
import copy
from model import Model



parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data/train.jsonl", type=str, help="data path")
parser.add_argument("--model_path", default="models/model.model", type=str, help="Model path")

class Dataset:

    classes = ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY',
       'PARENTING', 'FOOD & DRINK', 'QUEER VOICES', 'HEALTHY LIVING',
       'BUSINESS', 'COMEDY', 'BLACK VOICES', 'SPORTS', 'PARENTS',
       'HOME & LIVING', 'WEDDINGS', 'IMPACT', 'WOMEN', 'CRIME', 'WORLD NEWS',
       'THE WORLDPOST', 'DIVORCE', 'MEDIA', 'RELIGION', 'WORLDPOST', 'TASTE',
       'WEIRD NEWS', 'GREEN', 'TECH', 'STYLE', 'SCIENCE', 'MONEY', 'FIFTY',
       'U.S. NEWS', 'ENVIRONMENT', 'GOOD NEWS', 'ARTS & CULTURE', 'ARTS',
       'CULTURE & ARTS', 'EDUCATION', 'LATINO VOICES', 'COLLEGE']
    
    def __init__(self, file_path ):
        self.data = self._load_data(file_path)

        
              
        if 'category' in self.data:
            self.y = [Dataset.classes.index(x) for x in self.data['category']]
        else:
            self.y = None
        
        # have to call Prepare data
        self.x = None
       

    def _load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        data = pd.DataFrame(data)
        #nltk.download('punkt')
        #data['headline'] = data['headline'].apply(lambda x: nltk.word_tokenize(x))
        #data['short_description'] = data['short_description'].apply(lambda x: nltk.word_tokenize(x))     
        return data
    
    def get_data_for_vectorizer(self):
        x =  self.data['headline'] + ' ' + self.data['short_description']
        return x

    def prepare_data(self, vectorizer):
        data = self.get_data_for_vectorizer()
        x = vectorizer.transform(data).toarray()

        #add week day 
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['weekday'] = self.data['date'].dt.dayofweek.astype(int)
        self.data['weekday'] = self.data['weekday'].astype(int)
        one_hot_encoder = OrdinalEncoder()
        weekday = one_hot_encoder.fit_transform(np.array(self.data['weekday']).reshape(-1, 1))
        #train_x = np.hstack((train_x, weekday)) #doesn't helped
        self.x = x
    
    def transform_data(self, vectorizer):
        print("words transform")
        
        
        


def test_model(model, train_x, train_y, name, grid = {}):
    print(f"testing: {name}")   
    grid_cv = GridSearchCV(model, grid, error_score='raise', scoring=make_scorer(accuracy_score))
    grid_cv.fit(train_x,train_y)
    print(f"accuracy: {grid_cv.best_score_}")
    print('-'*50,'\n')
    return grid_cv.best_estimator_


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    data = Dataset(args.data_path)
    print("words fit")
    vectorizer = TfidfVectorizer( analyzer='word',lowercase=True ,ngram_range=(1,1),max_features=1000).fit(data.get_data_for_vectorizer())
    data.prepare_data(vectorizer)

    print('accuracy when all is POLITICS: ', accuracy_score([Dataset.classes.index('POLITICS')] * len(data.y), data.y))
    print('\n')
    print(data.x[:10])
    test_model(GaussianNB(), data.x, data.y, name='Gausian')
    best_estimator =test_model(LogisticRegression(), data.x, data.y, name='Logistic Regression')
    
    Model.save((best_estimator,vectorizer),args.model_path)
    #train_x, vectorizer = prepareData(data,vectorizer = CountVectorizer(analyzer='char',lowercase=True, ngram_range=(1,4),max_features = 1000))
    #test_model(GaussianNB(), train_x, train_y, name='Gausian')
    #test_model(LogisticRegression(), train_x, train_y, name='Logistic Regression')
    
    

    



   



#import gensim 
import torch
import argparse
from torchtext.vocab import GloVe
import torch.nn

#import pickle
#import sklearn
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

from torchtext.data.utils import get_tokenizer


#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
#from  sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression



import os
import numpy as np
import pandas as pd 
import json
import matplotlib.pyplot as plt
#import copy
from model import Model



parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data/train.jsonl", type=str, help="data path")
parser.add_argument("--model_path", default="models/model.model", type=str, help="Model path")
parser.add_argument("--embedder_path", default="embedders/glove.6B.50d.txt", type=str, help="Model path")



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
        self.x = x
    
        
    def _tokenize(self, column, max_words):
        tokenizer = get_tokenizer('basic_english')
        tokens = self.data[column].apply(lambda x: tokenizer(x)[:max_words])
        return tokens
        
    def _column_to_embeddings(self, model, tokens, max_tokens):
        embedding_dim = model.vector_size
        
        embeddings = np.zeros((max_tokens,max_tokens, embedding_dim))
       
        for i, sample in enumerate(tokens):
            print(model.get_vecs_by_tokens(sample))
            for i_token, token in enumerate(sample):

                if token in model.vocab:
                    embeddings[i,i_token,:] = model[token]
                else:
                    embeddings[i,i_token,:] = np.zeros(embedding_dim)
                    print(f"for token: {token}, embeddings was not found")
        return embeddings


    def prepare_embeddings(self,columns, max_words, embedder_path):
        model = GloVe()
        #if os.path.exists(embedder_path):
        #    model  = gensim.models.KeyedVectors.load_word2vec_format(embedder_path, binary=True)
        #else:
        #    model  = gensim.downloader.load(DEFAULT_EMBEDDER)
        #
        input_data = []
        for column, max_words in zip(columns, max_words):
            tokens = self._tokenize(self, column, max_words)
            embeddings = self._column_to_embeddings(model, tokens)
            input_data.append(embeddings)
        self.x = torch.cat(input_data, dim=1)


def test_model(model, train_x, train_y, name, grid = {}):
    print(f"testing: {name}")   
    grid_cv = GridSearchCV(model, grid, error_score='raise', scoring=make_scorer(accuracy_score))
    grid_cv.fit(train_x,train_y)
    print(f"accuracy: {grid_cv.best_score_}")
    print('-'*50,'\n')
    return grid_cv.best_estimator_

def train_tf_idf(args):
    
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

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    data = Dataset(args.data_path)
    data.prepare_embeddings(('headline','short_description'), (44, 50), args.embedder_path)
    print(data.x[0])
    print(data.x[1])
    print(data.x[2])
    pass
    
    

    



   

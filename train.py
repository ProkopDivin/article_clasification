

#import gensim 
import torch
import argparse
import torch.nn
import torch.nn.functional as F


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
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

from NN_model import EmbModel


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data/train.jsonl", type=str, help="data path")
parser.add_argument("--model_path", default="models/model.model", type=str, help="Model path")
parser.add_argument("--embedder_path", default="embedders/glove.6B.50d.txt", type=str, help="embedders/glove.6B.DIMd.txt where: DIM in [50 , 100, 200, 300], and DIM == --emb_dimension")
parser.add_argument("--emb_dimension", default= 50 , type=int, help="dimensions 50 , 100, 200, 300")
parser.add_argument("--use_columns", default= ['headline','short_description'] , type=list, help="Model path")
parser.add_argument("--max_tokens", default= [44, 50] , type=list, help="Model path")

class BaseDataset:
    classes = ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY',
           'PARENTING', 'FOOD & DRINK', 'QUEER VOICES', 'HEALTHY LIVING',
           'BUSINESS', 'COMEDY', 'BLACK VOICES', 'SPORTS', 'PARENTS',
           'HOME & LIVING', 'WEDDINGS', 'IMPACT', 'WOMEN', 'CRIME', 'WORLD NEWS',
           'THE WORLDPOST', 'DIVORCE', 'MEDIA', 'RELIGION', 'WORLDPOST', 'TASTE',
           'WEIRD NEWS', 'GREEN', 'TECH', 'STYLE', 'SCIENCE', 'MONEY', 'FIFTY',
           'U.S. NEWS', 'ENVIRONMENT', 'GOOD NEWS', 'ARTS & CULTURE', 'ARTS',
           'CULTURE & ARTS', 'EDUCATION', 'LATINO VOICES', 'COLLEGE']
    
    def _load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        data = pd.DataFrame(data)
        return data
    
class TfIdfDataset(BaseDataset):
    
    
    def __init__(self, file_path ):
        self.data = self._load_data(file_path)
         
        if 'category' in self.data:
            self.y = [BaseDataset.classes.index(x) for x in self.data['category']]
        else:
            self.y = None
        
        # have to call Prepare data
        self.x = None
    
    def get_data_for_vectorizer(self):
        x =  self.data['headline'] + ' ' + self.data['short_description']
        return x

    def prepare_data(self, vectorizer):
        data = self.get_data_for_vectorizer()
        x = vectorizer.transform(data).toarray()
        self.x = x
    

class TorchDataset(BaseDataset,Dataset):

    
    def __init__(self, file_path ):
        super().__init__()
        
        self.data = self._load_data(file_path)
        one_hot_labels = torch.nn.functional.one_hot(torch.arange(len(BaseDataset.classes)))

        if 'category' in self.data:
            self.y = [one_hot_labels[BaseDataset.classes.index(x)] for x in self.data['category']]
        else:
            self.y = None
        
        # have to call Prepare data
        self.x = None
        # vocabulary for embeddings
        self.vocab = None
        self.embeddings = None
        self.emb_dim = 0
        self.tokenizer = get_tokenizer("basic_english")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        label, x = self.y[idx], self.x[idx]   
        txt_rep = x
        label, txt_rep = torch.tensor(label, dtype=torch.float32), torch.tensor(txt_rep, dtype=torch.long)
        return label.to(self.device), txt_rep.to(self.device)
      
    # Function to load GloVe embeddings
    def _load_embeddings(self, file_path):
        embeddings_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = vector
        return embeddings_dict

    def _create_tokens(self, data):
        for text in data:
            yield self.tokenizer(text.lower())
    
    def _build_vocab(self, text):
        
        vocab = build_vocab_from_iterator(self._create_tokens(text), specials=["<oov>", "<sos>"])
        vocab.set_default_index(vocab["<oov>"])
        self.vocab = vocab
    
    def _build_emb_matrix(self, trained_emb, emb_dim):
        embedding_matrix = np.random.normal(scale=0.6, size=(len(self.vocab), emb_dim))
        mean_embedding = np.mean(np.array(list(trained_emb.values())), axis=0)
        for word, idx in self.vocab.get_stoi().items():
            if word in trained_emb:
                embedding_matrix[idx] = trained_emb[word]
            else:
                embedding_matrix[idx] = mean_embedding
        self.embeddings = embedding_matrix
    
    def _text_to_indexes(self, text, vocab, tokenizer, max_tokens):
        tokens = tokenizer(text)
        indexes = [vocab[token] for token in tokens]
        indexes = torch.tensor(indexes)
        max_tokens -= len(indexes)
        return F.pad(indexes, (max_tokens, 0))

    def prepare_embeddings(self, emb_path, emb_dim, columns, max_words, vocab = None):
        if vocab is None:
            vocab_data = [self.data[name] for name in columns]
            self._build_vocab(pd.concat(vocab_data, axis=0).reset_index(drop=True))
        else:
            self.vocab = vocab
        trained_emb = self._load_embeddings(emb_path)
        self._build_emb_matrix(trained_emb, emb_dim)
        # want frst token of feature begin on the same index if using multiple text features in dataset 
        indexed_columns = [] 
        for name, max_tokens in zip(columns,max_words):
           indexed_data = torch.zeros([len(self.data[name]), max_tokens],dtype=torch.long)
           for idx, text in enumerate(self.data[name]):
               indexed_data[idx] = self._text_to_indexes(text, self.vocab, self.tokenizer, max_tokens) 
           indexed_columns.append(indexed_data)
        self.x = torch.cat(indexed_columns, dim=1)
        self.emb_dim = sum(max_words) * emb_dim


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
    train_dataset = TorchDataset(args.data_path)
    train_dataset.prepare_embeddings( args.embedder_path, args.emb_dimension, columns=args.use_columns, max_words=args.max_tokens)
    val_dataset = TorchDataset(args.data_path)
    val_dataset = TorchDataset("data/dev.jsonl")
    val_dataset.prepare_embeddings( args.embedder_path, args.emb_dimension, columns=args.use_columns, max_words=args.max_tokens, vocab = train_dataset.vocab)


    batch_size = 256
    dataloader_training = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_validation = DataLoader(val_dataset, batch_size=batch_size)


    vocab_size  = len(train_dataset.vocab)
    emb_dim = train_dataset.emb_dim
    hidden_size = 32
    num_layers = 1
    RNN_type = 'Simple RNN' #possible choices -> ['Simple RNN', 'LSTM', 'GRU']
    bidirectional = False
    lr = 1e-3
    
    model = EmbModel(vocab_size=vocab_size, emb_dim=emb_dim, hidden_size=hidden_size,
                      output_dim=len(BaseDataset.classes)).to(train_dataset.device)
    
 

    criterion = torch.nn.BCELoss()  # does not apply sigmoid
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

    print
    E = 500
    label, text = next(iter(dataloader_training))
    for itr in range(E):
        model.train()
        optimizer.zero_grad()
        logits = model(text)
        loss = criterion(logits, label)
        if itr % 100 == 0 or itr == E-1:
          print(f"epoch: {itr} -> Loss: {loss}")
        loss.backward()
        optimizer.step()


    
    

    



   

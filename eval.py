
import argparse
import lzma
import pickle
import pandas as pd
import sklearn
import numpy as np
from train import Dataset
from model import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data/dev.jsonl", type=str, help=" Data path")
parser.add_argument("--model_path", default="models/model.model", type=str, help="Model path")

class Evaluator:

    def __init__(self, model_path):
        with lzma.open(model_path, "rb") as model_file:
            self.model, self.data_preparator = pickle.load(model_file)
    
    def prepare_data(self, data):    
        data.prepare_data(self.data_preparator)
        return (data.x, data.y)
    

    def predict(self, x):
        return self.model.predict(x)
    

    def eval(self, data_path):
        data = Dataset(data_path)
        test_x, test_y = self.prepare_data(data)
        pred_y = self.predict(test_x)
        print(f"accuracy: {accuracy_score(test_y, pred_y)}")

        #print('cofusion matrix: ')
        #print(confusion_matrix(test_y, pred_y, Dataset.classes))
        #print('precision:')
        #print(precision_score(test_y, pred_y, Dataset.classes))
        #print('recall:')
        #print(recall_score(test_y, pred_y, Dataset.classes))
        #print()



        
if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    evaluator = Evaluator(args.model_path)
    evaluator.eval(args.data_path)


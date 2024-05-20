
import argparse
import pickle
import pandas as pd 
import torch 
from train import TorchDataset
from torch.utils.data import DataLoader
from train import BaseDataset
from NN_model import EmbModel
import sklearn.metrics



parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data/dev.jsonl", type=str, help=" Data path")
parser.add_argument("--model_path", default="models/model.model", type=str, help="Model path")
parser.add_argument("--emb_dimension", default= 50 , type=int, help="length of one embedding: 50 , 100, 200, 300")
parser.add_argument("--use_columns", default= ['headline','short_description'] , type=list, help="possible values: ['headline','short_description']")
parser.add_argument("--max_tokens", default= [44, 60] , type=list, help=" [44, 60] ... none of the headlines will be truncated, 0.85 % of short description will be trucated a litle bit ")

class Evaluator:

    def __init__(self, args):
        with open(args.model_path + "vocab.pkl", "rb") as vocab:
            #self.model, self.data_preparator = pickle.load(model_file)
            self.vocab =pickle.load(vocab)
        
        data = TorchDataset(args.data_path)
        data.prepare_embeddings(None, args.emb_dimension, args.use_columns, args.max_tokens, vocab=self.vocab)

        data1 = TorchDataset(args.data_path)
        data1.prepare_embeddings("embedders/glove.6B.50d.txt", args.emb_dimension, args.use_columns, args.max_tokens, vocab=self.vocab)
    

        vocab_size  = len(data.vocab)
        emb_dim = data.emb_dim
        tokens_insample = data.tokens
        hidden_layers = ()
        dropout_rate = 0.50
        
        self.model = EmbModel(vocab_size = vocab_size,
                          emb_dim = emb_dim, 
                          tokens_insample = tokens_insample,
                          hidden_layers = hidden_layers,
                          pretrained_emb = None,
                          learn_emb = True,
                          dropout_rate = dropout_rate,
                          device = data.device,
                          output_dim=len(BaseDataset.classes)).to(data.device)
        self.model.load_state_dict(torch.load(args.model_path))
        
        
        self.dataloader = DataLoader(data, batch_size=len(data.data), shuffle=True)
        

    def predict(self):
        with torch.no_grad():
            self.model.eval()  
            data_iter = iter(self.dataloader)
            batch_labels,batch_data,= next(data_iter)
            output = self.model(batch_data)
            prediction = EmbModel._logits_to_onehot(output)
        
        prediction = torch.argmax(prediction, dim=-1)
        batch_labels = torch.argmax(batch_labels, dim=-1)
        return (prediction,batch_labels)
    

    def print_con_matrix(self,true_y, pred_y):
        classes = BaseDataset.classes  
        con_matrix = sklearn.metrics.multilabel_confusion_matrix(true_y, pred_y, labels = [i for i in range(len(classes))] )
        for i, matrix in enumerate(con_matrix):
            print(f"Confusion matrix for class '{classes[i]}':")
            df = pd.DataFrame(matrix, index=["True Neg", "True Pos"], columns=["Pred Neg", "Pred Pos"])
            print(df)
            tn, fp, fn, tp = matrix.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"precision: {precision}")
            print(f"recall: {recall}")
            print()
        
    def eval(self): 
        pred_y, true_y= self.predict()
        self.print_con_matrix(true_y, pred_y)
        print("-" * 50)
        print(f"accuracy: {sklearn.metrics.accuracy_score(true_y, pred_y)} \n")
        
   


        
if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    evaluator = Evaluator(args)
    evaluator.eval()


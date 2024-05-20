
import argparse
import pickle
import torch 
from train import TorchDataset
from torch.utils.data import DataLoader
from train import BaseDataset
from NN_model import EmbModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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
        self.model.eval()
        
        self.dataloader = DataLoader(data, batch_size=len(data.data), shuffle=True)


    def predict(self):
        with torch.no_grad():  
            data_iter = iter(self.dataloader)
            batch_data, batch_labels = next(data_iter)
            output = self.model(batch_data)
            prediction = EmbModel._logits_to_onehot(output)

        
        return (prediction,batch_labels)

    def eval(self):
        
        pred_y, true_y= self.predict()
        pred_y = torch.argmax(pred_y, dim=-1)
        true_y = torch.argmax(true_y, dim=-1)

        print(f"accuracy: {accuracy_score(true_y, pred_y)}")
        print(pred_y)
        
        #print('cofusion matrix: ')
        #print(confusion_matrix(test_y, pred_y, Dataset.classes))
        #print('precision:')
        #print(precision_score(test_y, pred_y, Dataset.classes))
        #print('recall:')
        #print(recall_score(test_y, pred_y, Dataset.classes))
        #print()



        
if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    evaluator = Evaluator(args)
    evaluator.eval()


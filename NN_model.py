

import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from  sklearn.metrics import accuracy_score


class EmbModel(nn.Module):
    
    def __init__(self, vocab_size, emb_dim, tokens_insample, hidden_layers, output_dim, pretrained_emb, dropout_rate, device, learn_emb=False):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.num_of_layers = len(hidden_layers)
     
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb)
            self.embedding.weight.requires_grad = learn_emb
  
        self.hidden_layers = nn.ModuleList()
        input_size = emb_dim * tokens_insample
        self.dropout = nn.Dropout(dropout_rate)

        for i, size in enumerate(hidden_layers):
            if i < self.num_of_layers:
                self.hidden_layers.append(nn.Linear(input_size, size))
                input_size = size
         
        
        self.output = nn.Linear(input_size, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)   
        x = x.view(batch_size, -1) 
        x = self.dropout(x)

        for hidden in self.hidden_layers:
            x = F.relu(hidden(x))
            x = self.dropout(x)

        x = F.softmax(self.output(x), dim=1)
        return x
    
    def _logits_to_onehot(logits):
        one_hot = lambda x: torch.tensor([1 if i == torch.argmax(x) else 0 for i in range(len(x))])
        one_hot_tensor = torch.stack([one_hot(row) for row in logits])
        return one_hot_tensor
    
    def _count_correct(true_y, pred_y):
        count = 0
        for yt,yp in zip(true_y, pred_y):
            if torch.argmax(yt) == torch.argmax(yp): count += 1
        return count



    def train_and_val(model, optimizer, dataloader_training, dataloader_validation, epochs, criterion):
        loss_train, loss_val = [], []
        acc_train, acc_val = [], []

        for epoch in range(epochs):
            model.train()
            total_acc_train, total_count_train, n_train_batches, total_loss_train = 0, 0, 0, 0
           
            for idx, (label, text) in enumerate(dataloader_training):
                optimizer.zero_grad()
                logits = model(text)
                loss = criterion(logits, label)
                total_loss_train += loss
                loss.backward()
                optimizer.step()
                
          
                #labels_form_logits = lambda x: torch.argmax(x)
                #logits = torch.tensor(list(map(labels_form_logits, logits))).to(model.device)
                #total_acc_train += (logits == label).sum().item()
                total_acc_train += EmbModel._count_correct(label,EmbModel._logits_to_onehot(logits))
                total_count_train += label.size(0)
                n_train_batches += 1
        
            avg_loss_train = total_loss_train/n_train_batches
            loss_train.append(avg_loss_train.item())
            accuracy_train = total_acc_train/total_count_train
            acc_train.append(accuracy_train)
        
            total_acc_val, total_count_val, n_val_batches, total_loss_val = 0, 0, 0, 0
            with torch.no_grad():
                model.eval()
                for idx, (label, text) in enumerate(dataloader_validation):
                    logits = model(text)
                    loss = criterion(logits, label)
                    total_loss_val += loss
                    
                    #logits = torch.tensor(list(map(labels_form_logits, logits))).to(model.device)
                    #total_acc_val += (logits == label).sum().item()
                    total_acc_val += EmbModel._count_correct(label,EmbModel._logits_to_onehot(logits))
                    total_count_val += label.size(0)
                    n_val_batches += 1
            avg_loss_val = total_loss_val/n_val_batches
            loss_val.append(avg_loss_val.item())
            accuracy_val = total_acc_val/total_count_val
            acc_val.append(accuracy_val)
            if epoch % 1 == 0:
                print(f"epoch: {epoch+1} -> Accuracy: {100*accuracy_train:.2f}%, Loss: {avg_loss_train:.8f}",end=" ---------------- ")
                print(f"Val_Acc: {100*accuracy_val:.2f}%, Val_Loss: {avg_loss_val:.8f}")
        return loss_train, acc_train, loss_val, acc_val
    
    def plot_learning_acc_and_loss(loss_tr, acc_tr, loss_val, acc_val, epochs):

        plt.figure(figsize=(8, 10))
    
        plt.subplot(2, 1, 1)
        plt.grid()
        plt.plot(range(epochs), acc_tr, label='acc_training')
        plt.plot(range(epochs), acc_val, label='acc_validation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
    
        plt.subplot(2, 1, 2)
        plt.grid()
        plt.plot(range(epochs), loss_tr, label='loss_training')
        plt.plot(range(epochs), loss_val, label='loss_validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
    
        plt.show()
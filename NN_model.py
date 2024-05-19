

import torch.nn as nn
import torch
import torch.nn.functional as F




class EmbModel(nn.Module):
    
    def __init__(self, vocab_size, emb_dim, tokens_insample, hidden_layers, output_dim, pretrained_emb, dropout_rate, learn_emb=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.num_of_layers = len(hidden_layers)
     
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb)
            self.embedding.weight.requires_grad = learn_emb
  
        self.hidden_layers = nn.ModuleList()
        input_size = emb_dim * tokens_insample
        self.dropout = nn.Dropout(dropout_rate)

        for i, size in enumerate(hidden_layers):
            if i < self.num_of_layers - 1:
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
    
    def train(model, optimizer, dataloader_train, dataloader_val, epochs, criterion):
        loss_train, loss_val = [], []
        acc_train, acc_val = [], []
        for epoch in range(epochs):
          model.train()
          total_acc_train, total_count_train, n_train_batches, total_loss_train = 0, 0, 0, 0
          for idx, (label, text) in enumerate(dataloader_train):
            optimizer.zero_grad()
            logits = model(text)
            loss = criterion(logits, label)
            total_loss_train += loss
            loss.backward()
            optimizer.step()
      
            labels_form_logits = lambda x: 0. if x < 0.5 else 1.
            logits = torch.tensor(list(map(labels_form_logits, logits))).to(model.device)
            total_acc_train += (logits == label).sum().item()
            total_count_train += label.size(0)
            n_train_batches += 1
      
          avg_loss_train = total_loss_train/n_train_batches
          loss_train.append(avg_loss_train.item())
          accuracy_train = total_acc_train/total_count_train
          acc_train.append(accuracy_train)
      
          total_acc_val, total_count_val, n_val_batches, total_loss_val = 0, 0, 0, 0
          with torch.no_grad():
              model.eval()
              for idx, (label, text) in enumerate(dataloader_val):
                  logits = model(text)
                  loss = criterion(logits, label)
                  total_loss_val += loss
                  logits = torch.tensor(list(map(labels_form_logits, logits))).to(model.device)
                  total_acc_val += (logits == label).sum().item()
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


import torch.nn as nn
import torch
import torch.nn.functional as F




class EmbModel(nn.Module):
    
    def __init__(self, vocab_size, emb_dim, hidden_size, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        self.hidden = nn.Linear(emb_dim, hidden_size)
        self.output = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        hidden_out = F.relu(self.hidden(embedded))
        output_out = F.softmax(self.output(hidden_out), dim=1)
        return output_out
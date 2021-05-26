
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk import word_tokenize

class sentiNN(nn.Module):
    """ 
    Neural Network for sentiment analysis. GRU model with binary classification linear layer on top.
    WITHOUT Features.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, use_features:list=None):
        super(sentiNN, self).__init__()
        
        # Variables / parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.use_features = use_features
        
        # Layers text
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

        # Layers features
        if self.use_features:
            self.feat_linear = nn.Linear(len(self.use_features), len(self.use_features))
            self.fc3 = nn.Linear(self.hidden_size * self.sequence_length * 2 + len(use_features), out_features=2)

        else:
            self.fc3 = nn.Linear(self.hidden_size * self.sequence_length * 2, out_features=2)

        # Criterion
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):        
        # Initialize GRU hidden state
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Index text and features
        x_text = x[:, :self.sequence_length, :]
        x_feat = x[:, self.sequence_length:, :]

        print("Shapes... Text:", x_text.shape, "Features:", x_feat.shape)

        # Forward text
        out, _ = self.gru(x_text)
        out = out.reshape(out.shape[0], -1)

        # Forward features
        if self.use_features:
            feat_out = self.feat_linear(x_feat)
            out = nn.torch.cat((out, feat_out), 0)

        out = self.fc3(out)
        
        return out
    


if __name__ == '__main__':
    main()
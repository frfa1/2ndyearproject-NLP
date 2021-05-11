
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk import word_tokenize

class sentiNN(nn.Module):
    """ 
    Neural Network for sentiment analysis. GRU model with binary classification FFNN on top.
    """
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(sentiNN, self).__init__()
        
        # Variables / parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layers
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        #self.fc1 = nn.Linear(self.hidden_size * sequence_length, 100)
        #self.fc2 = nn.Linear(100, out_features=2)
        self.fc3 = nn.Linear(self.hidden_size * sequence_length * 2, out_features=2)
        
    def forward(self, x):        
        # Initialize GRU hidden state
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward Prop
        out, _ = self.gru(x) #, h0)
        out = out.reshape(out.shape[0], -1)        
        #out = F.relu(self.fc1(out))
        #out = self.fc2(out)
        out = self.fc3(out)
        
        return out


if __name__ == '__main__':
    main()
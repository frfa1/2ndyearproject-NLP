
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk import word_tokenize
from preprocessing import create_emb_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class sentiNN(nn.Module):
    """ 
    Neural Network for sentiment analysis. GRU model with binary classification linear layer on top.
    WITHOUT Features.
    """
    
    def __init__(self, hidden_size1, hidden_size2, num_layers1, num_layers2, sequence_length, vocab_size, emb_dim, num_features=None): # input_size (Old) weight_matrix
        super(sentiNN, self).__init__()
        
        # Variables / parameters
        #self.input_size = input_size # Old
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2

        self.sequence_length = sequence_length
        self.num_features = num_features
        self.vocab_size = vocab_size
        #self.weight_matrix = weight_matrix
        
        # Embedding layer (modified from Christian)
        #self.embedding, num_embeddings, embedding_dim = create_emb_layer(self.weight_matrix) # pretrained embs
        self.embedding = nn.Embedding(self.vocab_size+1, emb_dim) # random embs   
        
        # Layers text
        #self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True) # Old
        #self.lstm = nn.LSTM(128, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True) # test lstm
        self.gru1 = nn.GRU(emb_dim, self.hidden_size1, self.num_layers1, batch_first=True, bidirectional=True, dropout=0.2) # With emb layer
        self.gru2 = nn.GRU(self.hidden_size1 * 2, self.hidden_size2, self.num_layers2, batch_first=True, bidirectional=True, dropout=0.2)

        # Layers features
        if self.num_features:
            self.feat_linear = nn.Linear(self.num_features, self.num_features)
            self.fc3 = nn.Linear(self.hidden_size2 * self.sequence_length * 2 + self.num_features, out_features=2)

        else:
            self.fc3 = nn.Linear(self.hidden_size2 * self.sequence_length * 2, out_features=2)

        # Criterion
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):        
        # Initialize GRU hidden state
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Index text and features
        x_text = x[:, :self.sequence_length].to(torch.long)
        x_feat = x[:, self.sequence_length:]

        # Embedding
        embedded_text = self.embedding(x_text)

        # Forward text
        #out, _ = self.lstm(embedded_text) # test lstm
        out, _ = self.gru1(embedded_text)
        out, _ = self.gru2(out)
        out = out.reshape(out.shape[0], -1)

        # Forward features
        if self.num_features:
            feat_out = self.feat_linear(x_feat)
            out = torch.cat((out, feat_out), 1)
        out = self.fc3(out)

        return out

def main():
    pass
    


if __name__ == '__main__':
    main()
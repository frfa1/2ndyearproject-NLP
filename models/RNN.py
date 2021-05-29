import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import get_vocab, binary_y, new_preprocessing
from loader import load_train, load_dev, load_movies

import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data(train,dev,test,max_sequence_length,dump=False):
    print('Initializing preprocessing...')

    vocab, word_idx, idx_word = get_vocab(train['reviewText']) # Get vocab etc from train

    train_text = new_preprocessing(train['reviewText'], vocab, word_idx, idx_word, max_length=max_sequence_length)
    train_feats = train.drop(["reviewText", "sentiment"], axis=1)
    train_conc = np.concatenate((train_text, train_feats.values), axis=1)
    train_x = torch.tensor(train_conc)
    train_y = binary_y(train["sentiment"])
    all_train = TensorDataset(train_x, train_y)

    dev_text = new_preprocessing(dev['reviewText'], vocab, word_idx, idx_word, max_length=max_sequence_length)
    dev_feats = dev.drop(["reviewText", "sentiment"], axis=1)
    dev_conc = np.concatenate((dev_text, dev_feats.values), axis=1)
    dev_x = torch.tensor(dev_conc)
    dev_y = binary_y(dev["sentiment"])
    all_dev = TensorDataset(dev_x, dev_y)

    test_text = new_preprocessing(test['reviewText'], vocab,word_idx, idx_word, max_length=max_sequence_length)
    test_feats = test.drop(["reviewText","sentiment"], axis=1)
    test_conc = np.concatenate((test_text, test_feats.values), axis=1)
    test_x = torch.tensor(test_conc)
    test_y = binary_y(test['sentiment'])
    all_test = TensorDataset(test_x,test_y)

    

    # Batching the data
    batch_size = 50
    train_batches = DataLoader(all_train, batch_size=batch_size)
    dev_batches = DataLoader(all_dev, batch_size=batch_size)

    

    if dump:
        print('\tMaking pickles...')
        with open('pickles/train_batches.pickle', 'wb') as f:
            pickle.dump(train_batches, f, pickle.HIGHEST_PROTOCOL)
        with open('pickles/dev_batches.pickle', 'wb') as f:
            pickle.dump(dev_batches, f, pickle.HIGHEST_PROTOCOL)
        with open('pickles/vocab.pickle', 'wb') as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
        with open('pickles/train_conc_shape.pickle', 'wb') as f:
            pickle.dump(train_conc.shape, f, pickle.HIGHEST_PROTOCOL)
        with open('pickles/train_conc_shape.pickle', 'wb') as f:
            pickle.dump(train_conc.shape, f, pickle.HIGHEST_PROTOCOL)
        with open('pickles/test_X.pickle', 'wb') as f:
            pickle.dump(test_x, f, pickle.HIGHEST_PROTOCOL)
        with open('pickles/test_y.pickle', 'wb') as f:
            pickle.dump(test_y, f, pickle.HIGHEST_PROTOCOL)

    print("Preprocessing completed!")

    return train_batches, dev_batches, vocab, train_conc.shape, test_x,test_y




class RNN(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, num_layers1, num_layers2, sequence_length, vocab_size, emb_dim, num_features=None): # input_size (Old) weight_matrix
        super(RNN, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2

        self.sequence_length = sequence_length
        self.num_features = num_features
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size+1, emb_dim) # random embs   

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

def runNN(train,dev,test,hidden_size1, hidden_size2, num_layers1, num_layers2, sequence_length, emb_dim,n_features=None):
    train_batches, dev_batches, vocab, data_shape, test_X,test_y = get_data(train,dev,test,sequence_length)
    net = RNN(hidden_size1, hidden_size2, num_layers1, num_layers2, sequence_length, len(vocab), emb_dim, num_features=n_features).float()


def main():
    train = load_train()
    dev = load_dev()
    test = load_movies()

    sequence_length = 60 # pickles currently at 60
    hidden_size1 = 100
    num_layers1 = 2
    hidden_size2 = 80
    num_layers2 = 2
    emb_dim = 400
    num_features = 0 # can be between 0 and 13

    learning_rate = 0.001
    momentum = 0.9
    num_epoch = 1

    train_batches, dev_batches, vocab, data_shape, test_X,test_y = get_data(train,dev,test,sequence_length,dump=True)
    #runNN(train,dev,test,hidden_size1,hidden_size2,num_layers1,num_layers2,sequence_length,emb_dim,num_features)
    

if __name__ == '__main__':
    main()
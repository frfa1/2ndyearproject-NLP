import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score,classification_report

from preprocessing import get_vocab, binary_y, new_preprocessing
from loader import load_train, load_dev, load_movies, load_train_handcrafted, load_dev_handcrafted, load_hard_handcrafted, load_movies_handcrafted

import pickle
import joblib

device = torch.device('cuda0' if torch.cuda.is_available() else 'cpu')


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


def load_pickled():
    with open('pickles/train_batches.pickle', 'rb') as f:
        train_batches = pickle.load(f)
    with open('pickles/dev_batches.pickle', 'rb') as f:
        dev_batches = pickle.load(f) 
    with open('pickles/vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    with open('pickles/train_conc_shape.pickle', 'rb') as f:
        data_shape = pickle.load(f)
    with open('pickles/test_X.pickle', 'rb') as f:
        test_X = pickle.load(f)
    with open('pickles/test_y.pickle', 'rb') as f:
        test_y = pickle.load(f)
    return train_batches,dev_batches,vocab,data_shape,test_X,test_y


class RNN(nn.Module):
    def __init__(self, train_batches,dev_batches,hidden_size1, hidden_size2, num_layers1, num_layers2, sequence_length, vocab_size, emb_dim, num_features=None): # input_size (Old) weight_matrix
        super(RNN, self).__init__()

        self.train_batches = train_batches
        self.dev_batches = dev_batches

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2

        self.sequence_length = sequence_length
        self.num_features = num_features
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size+1, emb_dim) # random embs   

        self.gru1 = nn.GRU(emb_dim, self.hidden_size1, self.num_layers1, batch_first=True, bidirectional=True, dropout=0.2) # With emb layer
        #self.gru2 = nn.GRU(self.hidden_size1 * 2, self.hidden_size2, self.num_layers2, batch_first=True, bidirectional=True, dropout=0.2)

        # Layers features
        if self.num_features:
            #self.feat_linear = nn.Linear(self.num_features, self.num_features)
            self.fc3 = nn.Linear(self.hidden_size1 * self.sequence_length * 2 + self.num_features, out_features=2)

        else:
            self.fc3 = nn.Linear(self.hidden_size1 * self.sequence_length * 2, out_features=2)

        # Criterion
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):        
        # Initialize GRU hidden state
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Index text and features
        x_text = x[:, :self.sequence_length].to(torch.long).to(device)
        x_feat = x[:, self.sequence_length:]

        # Embedding
        embedded_text = self.embedding(x_text)

        # Forward text
        out, _ = self.gru1(embedded_text)
        #out, _ = self.gru2(out)
        out = out.reshape(out.shape[0], -1)

        # Forward features
        if self.num_features:
            #feat_out = self.feat_linear(x_feat)
            out = torch.cat((out, x_feat), 1)
        out = self.fc3(out)

        return out

    def learn(self, learning_rate, momentum, num_epoch,save_model):
        # Loss function
        criterion = self.loss
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        epoch_score = []

        # Actual training
        for epoch in range(num_epoch):  # loop over the dataset multiple times

            train_losses = [] # List of losses on train set
            val_losses = [] # List of losses on dev set
            val_accuracies = [] # List of accuracies on dev set

            running_loss = 0.0
            for i, data in enumerate(self.train_batches, 0): # Iterates through each batch
                inputs, labels = data # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad() # zero the parameter gradients
                outputs = self(inputs.float()) # forward + backward + optimize
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() # append loss

                # print statistics
                m = 10 # print every m mini-batches
                if i % m == 0 and i != 0:    # print every 200 mini-batches
                    train_losses.append(running_loss / m) # Append average loss to train_losses

                    val_loss, val_acc = self._validate()
                    val_losses.append(val_loss) # Append loss on whole validation set of this iteration
                    val_accuracies.append(val_acc) # Same but for val accuracy
                    self.train() # Go back to training state

                    print('[%d, %5d] Training loss: %.3f | Validation loss: %.3f | Validation accuracy: %.1f' 
                        %(epoch + 1, i + 1, running_loss / m, val_loss, val_acc)) # Just some user interface

                    running_loss = 0.0

            # Average loss and accuracy for the epoch:
            avg_train_loss = 0
            avg_val_loss = 0
            avg_val_acc = 0
            for j in range(len(train_losses)):
                avg_train_loss += train_losses[j]
                avg_val_loss += val_losses[j]
                avg_val_acc += val_accuracies[j]
            avg_train_loss = avg_train_loss / len(train_losses)
            avg_val_loss = avg_val_loss / len(val_losses)
            avg_val_acc = avg_val_acc / len(val_accuracies)
            epoch_score.append([epoch+1, avg_train_loss, avg_val_loss, avg_val_acc])

        epoch_score_df = pd.DataFrame(epoch_score, columns=["epoch", "train_loss", "val_loss", "val_accuracy"])

        if save_model:
            joblib.dump(self,'50_40_50_4.joblib')
        return epoch_score_df


    def _validate(self):       
        criterion = self.loss # Get the loss function from the model class               
        correct = 0                                               
        total, total2 = 0, 0                                               
        running_loss = 0.0                                        
        self.eval() # Go to evaluation state                                
        with torch.no_grad():                                     
            for i, data in enumerate(self.dev_batches):                     
                inputs, labels = data 
                inputs, labels = inputs.to(device), labels.to(device)

                # Get the val loss                                        
                outputs = self(inputs.float())                     
                loss = criterion(outputs, labels)                    
                running_loss += loss.item()  
                total += 1  
                
                # Get the val accuracy
                total2 += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)  # Get the max value of each row, i.e. predicted                                  
                correct += (predicted == labels).sum().item()       
        mean_val_accuracy = 100 * correct / total2           
        mean_val_loss = running_loss / total

        return mean_val_loss, mean_val_accuracy 

    def predict(self,features,labels=None):
        features = features.to(device)

        if labels != None:
            labels = labels.to(device)

        self.eval()
        with torch.no_grad():
            test_predictions = self(features.float())
            _, predicted = torch.max(test_predictions,1)

        if labels != None:
            print(classification_report(labels,predicted))
            return predicted


def runNN(
        train_batches,
        dev_batches,
        vocab: dict,
        data_shape,
        test_X,
        test_y,
        hidden_size1: int, 
        hidden_size2: int, 
        num_layers1: int, 
        num_layers2: int, 
        sequence_length: int, 
        emb_dim: int,
        learning_rate: float,
        momentum: int,
        num_epochs: int,
        n_features=None,
        dump_trained=False,
        use_trained=False
    ):
    if not use_trained:
        print('Initialising NN...')
        net = RNN(train_batches,dev_batches,hidden_size1, hidden_size2, num_layers1, num_layers2, sequence_length, len(vocab), emb_dim, num_features=n_features).float()
        print(net.learn(learning_rate,momentum,num_epochs,dump_trained))
    if use_trained:
        print('Loading trained model...')
        net = joblib.load('50_40_50_3.joblib')

    print('Predicting...')
    y_pred = net.predict(test_X,labels=test_y)





def main():
    train = load_train_handcrafted()
    dev = load_dev_handcrafted()
    test = load_movies_handcrafted()

    sequence_length = 60 # pickles currently at 60, if you change, then set make_data to True
    make_data = True

    if make_data:
        train_batches, dev_batches, vocab, data_shape, test_X,test_y = get_data(train,dev,test,sequence_length,dump=False)
    else:
        print('Loading pickles...')
        train_batches, dev_batches, vocab, data_shape, test_X,test_y = load_pickled()
    
    # the below variables can be changed as you like
    hidden_size1 = 100
    num_layers1 = 2
    hidden_size2 = 40
    num_layers2 = 2
    emb_dim = 50 # 50, 100 , 300
    num_features = 7

    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 5

    # this is the call to the wrapper of the RNN model. It can train on the data loaded from above or it can load an already saved
    # model so that you can skip the training process and go straight to predictions.

    runNN(
        train_batches, 
        dev_batches, 
        vocab, 
        data_shape, 
        test_X,
        test_y,
        hidden_size1,
        hidden_size2,
        num_layers1,
        num_layers2,
        sequence_length,
        emb_dim,
        learning_rate,
        momentum,
        num_epochs,
        num_features,
        dump_trained=False,
        use_trained=False # setting this to true requires that you have already trained a model and dumped it in the pickles folder. To do so set dump_trained to True and run the script
    )
    

if __name__ == '__main__':
    main()
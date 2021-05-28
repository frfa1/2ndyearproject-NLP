import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk import word_tokenize
import joblib
import pandas as pd

from preprocessing import get_embs, preprocessing, binary_y, create_weight_matrix, preprocess_to_idx, get_vocab, new_preprocessing
from sentiNN import sentiNN

from loader import load_train, load_dev, load_test, load_train_handcrafted, load_dev_handcrafted
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(sequence_length):

    # Get pretrained embeddings
    #embs = get_embs("glove_6b")
 
    ### New ###
    train = load_train_handcrafted()
    dev = load_dev_handcrafted()
    vocab, word_idx, idx_word = get_vocab(train['reviewText']) # Get vocab etc from train

    train_text = new_preprocessing(train['reviewText'], vocab, word_idx, idx_word, max_length=sequence_length)
    train_feats = train.drop(["reviewText", "sentiment"], axis=1)
    train_conc = np.concatenate((train_text, train_feats.values), axis=1)
    train_x = torch.tensor(train_conc)
    train_y = binary_y(train["sentiment"])
    all_train = TensorDataset(train_x, train_y)

    dev_text = new_preprocessing(dev['reviewText'], vocab, word_idx, idx_word, max_length=sequence_length)
    dev_feats = dev.drop(["reviewText", "sentiment"], axis=1)
    dev_conc = np.concatenate((dev_text, dev_feats.values), axis=1)
    dev_x = torch.tensor(dev_conc)
    dev_y = binary_y(dev["sentiment"])
    all_dev = TensorDataset(dev_x, dev_y)

    print("Datasets loading done")

    # Batching the data
    batch_size = 50
    train_batches = DataLoader(all_train, batch_size=batch_size)
    dev_batches = DataLoader(all_dev, batch_size=batch_size)

    return train_batches, dev_batches, vocab, train_conc.shape


def training(model, train_batches, dev_batches, learning_rate, momentum, num_epoch):
    # Loss function
    criterion = model.loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    epoch_score = []

    # Actual training
    for epoch in range(num_epoch):  # loop over the dataset multiple times

        train_losses = [] # List of losses on train set
        val_losses = [] # List of losses on dev set
        val_accuracies = [] # List of accuracies on dev set

        running_loss = 0.0
        for i, data in enumerate(train_batches, 0): # Iterates through each batch
            inputs, labels = data # get the inputs; data is a list of [inputs, labels]
            optimizer.zero_grad() # zero the parameter gradients
            outputs = model(inputs.float()) # forward + backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # append loss

            # print statistics
            m = 500 # print every m mini-batches
            if i % m == 0 and i != 0:    # print every 200 mini-batches
                train_losses.append(running_loss / m) # Append average loss to train_losses

                val_loss, val_acc = validate(dev_batches, model)
                val_losses.append(val_loss) # Append loss on whole validation set of this iteration
                val_accuracies.append(val_acc) # Same but for val accuracy
                model.train() # Go back to training state

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

    return model, epoch_score_df


def validate(dev_batches, model):       
    criterion = model.loss # Get the loss function from the model class               
    correct = 0                                               
    total, total2 = 0, 0                                               
    running_loss = 0.0                                        
    model.eval() # Go to evaluation state                                
    with torch.no_grad():                                     
        for i, data in enumerate(dev_batches):                     
            inputs, labels = data 
            #inputs, labels = inputs.float(), labels.float()  # Added. Maybe change to "device" later                   
            #inputs = inputs.to(device)                        
            #labels = labels.to(device)    

            # Get the val loss                                        
            outputs = model(inputs.float())                           
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


#### Call training once ####
#model = sentiNN(hidden_size1, hidden_size2, num_layers1, num_layers2, sequence_length, len(vocab), emb_dim, num_features=num_features).float()
#n_model, epoch_score = training(model, train_batches, dev_batches, learning_rate, momentum, num_epoch)


#### Ablation study ####

def ablation_study():

    print("INITIALIZING ABLATION STUDY")

    ablation_scores = [0 for featz in range(num_features)]

    for i in range(num_features):
        temp_num_features = num_features - i

        print("Running with", temp_num_features, "features...")
        
        if temp_num_features != 0:
            model = sentiNN(hidden_size1, hidden_size2, num_layers1, num_layers2, sequence_length, len(vocab), emb_dim, num_features=temp_num_features).float()
        else:
            model = sentiNN(hidden_size1, hidden_size2, num_layers1, num_layers2, sequence_length, len(vocab), emb_dim).float()
        n_model, epoch_score = training(model, train_batches, dev_batches, learning_rate, momentum, num_epoch)

        ablation_scores[temp_num_features] = (n_model, epoch_score)
    
    return ablation_scores

#### End Ablation study ####


"""

#### Grid search ####
# Things to search for:
learning_rates = [0.001, 0.0001]
hidden_sizes = [50, 100, 300]
# Note: Epochs are also searched, but each step is stored in training

# Keeping scores:
grid_scores = {}
grid_scores["lr"] = 0
grid_scores["hs"] = 0
grid_scores["epoch"] = 0
grid_scores["best_model"] = 0
grid_scores["epoch_score"] = 0
best_loss = 1000000

for i in learning_rates:
    for j in hidden_sizes:

        print("Learning rate:", i, "| Hidden size:", j)

        model = sentiNN(input_size, j, num_layers, sequence_length).float()
        n_model, epoch_score = training(model, train_batches, dev_batches, i, momentum, num_epoch)

        for idx, k in enumerate(epoch_score):
            if k[1] < best_loss:
                grid_scores["lr"] = i
                grid_scores["hs"] = j
                grid_scores["epoch"] = idx
                grid_scores["best_model"] = n_model
                grid_scores["epoch_score"] = epoch_score
          
#### End Grid search ####
"""
#joblib.dump(grid_scores["best_model"], "trained_models/nn_baseline.joblib")


def main():
    sequence_length = 60
    train_batches, dev_batches, vocab, data_shape = get_data(sequence_length)

    # Define network - Rewrite to grid search
    #input_size = data_shape[1] # Old
    hidden_size1 = 100
    num_layers1 = 2
    hidden_size2 = 80
    num_layers2 = 2
    emb_dim = 400
    num_features = data_shape[1] - sequence_length

    # Training
    learning_rate = 0.001
    momentum = 0.9
    num_epoch = 1

    ablation_scores = ablation_study()

    return ablation_scores

if __name__ == '__main__':
    main()
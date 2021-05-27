import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk import word_tokenize
import joblib
import pandas as pd

from preprocessing import get_embs, preprocessing, binary_y, create_weight_matrix, preprocess_to_idx
from sentiNN import sentiNN

from loader import load_train, load_dev, load_test, load_train_handcrafted, load_dev_handcrafted
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(sequence_length):

    # Get pretrained embeddings
    embs = get_embs("glove_6b")
 
    ### New ###
    train = load_train_handcrafted()
    train_text = preprocess_to_idx(train['reviewText'], max_length=sequence_length)
    train_feats = train.drop(["reviewText", "sentiment"], axis=1)

    conc = np.concatenate((train_text, train_feats.values), axis=1)
    temp = []
    for i in conc:
        temp.append(i)
    torch.tensor(temp)
    print("Success")

    train_x = torch.tensor(np.concatenate((train_text, train_feats.values), axis=1))
    train_y = binary_y(train["sentiment"])
    all_train = TensorDataset(train_x, train_y)

    dev = load_dev_handcrafted()
    dev_text = preprocess_to_idx(dev['reviewText'], max_length=sequence_length)
    dev_feats = dev.drop(["reviewText", "sentiment"], axis=1)
    dev_x = torch.tensor(np.concatenate((dev_text, dev_feats.values), axis=1))
    dev_y = binary_y(dev["sentiment"])
    all_dev = TensorDataset(dev_x, dev_y)

    print("Datasets loading done")

    # Batching the data
    batch_size = 50
    train_batches = DataLoader(all_train, batch_size=batch_size)
    dev_batches = DataLoader(all_dev, batch_size=batch_size)

    return train_batches, dev_batches, train_x.shape, embs


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
            m = 100 # print every m mini-batches
            if i % m == 0 and i != 1:    # print every 200 mini-batches
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
        epoch_score.append([avg_train_loss, avg_val_loss, avg_val_acc])

    return model, epoch_score


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


sequence_length = 50
train_batches, dev_batches, data_shape, embs = get_data(sequence_length)

# Get initialized weight matrix
weight_matrix = create_weight_matrix(embs)

# Define network - Rewrite to grid search
# input_size = data_shape[2] # Old
hidden_size = 300
num_layers = 2

# Training
learning_rate = 0.001
momentum = 0.9
num_epoch = 5

print(data_shape)

#### Call training once ####
#model = sentiNN(hidden_size, num_layers, sequence_length, weight_matrix, ).float()
#(self, hidden_size, num_layers, sequence_length, weight_matrix, use_features:list=None)
#n_model, epoch_score = training(model, train_batches, dev_batches, learning_rate, momentum, num_epoch)
#print("Printing epoch scores:")
#print(epoch_score)

#### Grid search ####
# Things to search for:
learning_rates = [0.001, 0.0001]
hidden_sizes = [50, 100, 300]
# Note: Epochs are also searched, but each step is stored in training

"""# Keeping scores:
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
          
#### End Grid search ####"""

#joblib.dump(grid_scores["best_model"], "trained_models/nn_baseline.joblib")


def main():
    pass

if __name__ == '__main__':
    main()
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk import word_tokenize

from preprocessing import get_embs, preprocessing, binary_y
from sentiNN import sentiNN

from loader import load_train, load_dev, load_test
from torch.utils.data import DataLoader, TensorDataset


def get_data():
    # Get pretrained embeddings
    embs = get_embs()

    # Loading train, dev and test data
    train = load_train()
    dev = load_dev()
    test = load_test()

    train_x = preprocessing(train["reviewText"], embs, max_length=60)
    train_y = binary_y(train["sentiment"])
    all_train = TensorDataset(train_x, train_y)

    dev_x = preprocessing(dev["reviewText"], embs, max_length=60)
    dev_y = binary_y(dev["sentiment"])
    all_dev = TensorDataset(dev_x, dev_y)

    # Batching the data
    batch_size = 50
    train_batches = DataLoader(all_train, batch_size=batch_size)
    dev_batches = DataLoader(all_dev, batch_size=batch_size)

    return train_batches, dev_batches, embs, train, dev


def training(model, train_batches, dev_batches, learning_rate, momentum, num_epoch):
    # Loss function
    criterion = model.loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = [] # List of losses on train set
    val_losses = [] # List of losses on dev set

    # Actual training
    for epoch in range(num_epoch):  # loop over the dataset multiple times

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
            if i % 100 == 0:    # print every 200 mini-batches
                train_losses.append(running_loss / (i + 1)) # Append average loss to train_losses

                val_loss, _ = validate(dev_batches, model)
                val_losses.append(val_loss) # Append loss on whole validation set of this iteration

                print('[%d, %5d] Training loss: %.3f | Validation loss: %.3f' %(epoch + 1, i + 1, running_loss / (i + 1), val_loss)) # Just some user interface

                running_loss = 0.0

    return model, train_losses, val_losses


def validate(dev_batches, model):       
    criterion = model.loss # Get the loss function from the model class               
    correct = 0                                               
    total = 0                                                 
    running_loss = 0.0                                        
    model.eval()                                              
    with torch.no_grad():                                     
        for i, data in enumerate(dev_batches):                     
            inputs, labels = data 
            #inputs, labels = inputs.float(), labels.float()  # Added. Maybe change to "device" later                   
            #inputs = inputs.to(device)                        
            #labels = labels.to(device)   
            #print(inputs, labels) # Just testing...                                             
            outputs = model(inputs.float())                           
            loss = criterion(outputs, labels)                 
            _, predicted = torch.max(outputs.data, 1)  # Get the max value of each row, i.e. predicted       
            total += labels.size(0)                           
            correct += (predicted == labels).sum().item()     
            running_loss = running_loss + loss.item()         
    mean_val_accuracy = (100 * correct / total)               
    mean_val_loss = ( running_loss )  

    return mean_val_loss, mean_val_accuracy     
    #mean_val_accuracy = accuracy(outputs,labels)             
    #print('Validation Accuracy: %d %%' % (mean_val_accuracy)) 
    #print('Validation Loss:'  ,mean_val_loss )   


# Define network - Rewrite to grid search
input_size = train_x.shape[2]
hidden_size = 300
num_layers = 2
sequence_length = train_x.shape[1]

# Training
learning_rate = 0.001
momentum = 0.9
num_epoch = 1

# Call train function
#model = sentiNN(input_size, hidden_size, num_layers, sequence_length).float()
#n_model, train_losses, val_losses = training(model, train_batches, dev_batches, learning_rate, momentum, num_epoch)

def main():
    pass

if __name__ == '__main__':
    main()
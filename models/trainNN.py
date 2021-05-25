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


def get_data(sequence_length):
    # Get pretrained embeddings
    embs = get_embs()

    # Loading train, dev and test data
    train = load_train()
    dev = load_dev()
    #test = load_test()

    print("Datasets loading done")

    train_x = preprocessing(train["reviewText"], embs, max_length=sequence_length)
    train_y = binary_y(train["sentiment"])
    all_train = TensorDataset(train_x, train_y)

    dev_x = preprocessing(dev["reviewText"], embs, max_length=sequence_length)
    dev_y = binary_y(dev["sentiment"])
    all_dev = TensorDataset(dev_x, dev_y)

    # Batching the data
    batch_size = 50
    train_batches = DataLoader(all_train, batch_size=batch_size)
    dev_batches = DataLoader(all_dev, batch_size=batch_size)

    #print(train_x.shape[2], train_x.shape[1], print(train_x.shape))
    #print(train_batches.shape[2], train_batches.shape[1], print(train_batches.shape))

    return train_batches, dev_batches, train_x.shape


def training(model, train_batches, dev_batches, learning_rate, momentum, num_epoch):
    # Loss function
    criterion = model.loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    epoch_score = []

    # Actual training
    for epoch in range(num_epoch):  # loop over the dataset multiple times

        train_losses = [] # List of losses on train set
        val_losses = [] # List of losses on dev set

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

                val_loss = validate(dev_batches, model)
                val_losses.append(val_loss) # Append loss on whole validation set of this iteration
                model.train() # Go back to training state

                print('[%d, %5d] Training loss: %.3f | Validation loss: %.3f' %(epoch + 1, i + 1, running_loss / (i + 1), val_loss)) # Just some user interface

                running_loss = 0.0

        # Average loss for the epoch:
        avg_train_loss = 0
        avg_val_loss = 0
        for j in range(len(train_losses)):
            avg_train_loss += train_losses[j]
            avg_val_loss += val_losses[j]
        avg_train_loss, avg_val_loss = avg_train_loss/len(train_losses), avg_val_loss/len(val_losses)
        epoch_score.append([avg_train_loss, avg_val_loss])

    return model, epoch_score


def validate(dev_batches, model):       
    criterion = model.loss # Get the loss function from the model class               
    #correct = 0                                               
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
            running_loss += loss.item()  
            total += 1  
            
            #total += labels.size(0)
            #_, predicted = torch.max(outputs.data, 1)  # Get the max value of each row, i.e. predicted                                  
            #correct += (predicted == labels).sum().item()       
    #mean_val_accuracy = (100 * correct / total)               
    mean_val_loss = running_loss / total

    return mean_val_loss #, mean_val_accuracy     
    #mean_val_accuracy = accuracy(outputs,labels)             
    #print('Validation Accuracy: %d %%' % (mean_val_accuracy)) 
    #print('Validation Loss:'  ,mean_val_loss )   


sequence_length = 40
train_batches, dev_batches, data_shape = get_data(sequence_length)

# Define network - Rewrite to grid search
input_size = data_shape[2]
hidden_size = 300
num_layers = 2

# Training
learning_rate = 0.001
momentum = 0.9
num_epoch = 10

# Call train function
model = sentiNN(input_size, hidden_size, num_layers, sequence_length).float()
n_model, epoch_score = training(model, train_batches, dev_batches, learning_rate, momentum, num_epoch)

print("Printing epoch scores:")
print(epoch_score)

# Grid search
"""learning_rates = [0.001, 0.0001]
hidden_sizes = [50, 100, 300]
for i in learning_rates:
    for j in hidden_sizes:"""



def main():
    pass

if __name__ == '__main__':
    main()
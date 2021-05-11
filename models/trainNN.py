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

# Define network
input_size = train_x.shape[2]
hidden_size = 300
num_layers = 2
sequence_length = train_x.shape[1]

net = sentiNN(input_size, hidden_size, num_layers, sequence_length).float()

# Training
learning_rate = 0.001
momentum = 0.9
batch_size = 50
num_epoch = 30

# Batching the data
train_batches = DataLoader(all_train, batch_size=batch_size)
dev_batches = DataLoader(all_dev, batch_size=batch_size)

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# Actual training
loss_list = []

for epoch in range(num_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_batches, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            loss_list.append(running_loss)
            running_loss = 0.0

print('Finished Training')

if __name__ == '__main__':
    main()
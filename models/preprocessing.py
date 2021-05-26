#%%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk import word_tokenize

def get_embs(emb="glove_6b"):
    """
    Load embeddings
    """
    
    if emb == "googlenews":
        # Google News
        import gensim.models
        googEmbs = gensim.models.KeyedVectors.load_word2vec_format('../embeddings/googlenews.bin', binary=True)
        return googEmbs

    if emb == "glove_6b":
        # Glove 6B
        glove_dict = {}
        with open("../embeddings/glove.6B.50d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector

        print('loading finished')
        return glove_dict

# Preprocess function

def preprocessing(sentences, embs, max_length=None):
    """
    Inputs sequence of strings (predictor). Outputs data as word embeddings with fixed length and tensor format.
    
    max_length defines maximum length of sentences. Too short sentences will be padded, and too long sentences will be cut.
    If not defined, all sentences will be the length of the longest.
    """
    
    cleaned_sentence = []
    sent_length = 0
    
    for sentence in sentences:
        sentence = word_tokenize(sentence) # Preprocessing step: Tokenize
        emb_sent = []
        
        for word in sentence: # Convert each word into embedding or zero vector
            
            try:
                emb_word = embs[word.lower()]
            except KeyError:
                emb_word = np.zeros((50,)) 
            emb_sent.append(emb_word)
            
        if len(emb_sent) > sent_length:
            sent_length = len(emb_sent)
        cleaned_sentence.append(emb_sent)
        
    # Padding to longest sentence length -- Or max length variable if defined
    if not max_length:
        max_length = sent_length
        
    # Padding to longest sentence length
    for idx, cleaned_sent in enumerate(cleaned_sentence):
        if len(cleaned_sent) < max_length:
            for i in range(max_length - len(cleaned_sent)):
                cleaned_sent.append(np.zeros((50,)))
        if len(cleaned_sent) > max_length:
            cleaned_sentence[idx] = cleaned_sentence[idx][:max_length]
            
    return cleaned_sentence
    #return torch.tensor(cleaned_sentence) # a tensor of data. Each index is an instance


def preprocess_to_idx(sentences, embs, max_length=None):

    word_idx = dict((word, i) for i, word in enumerate(list(embs))) #create indices from embeddings
    
    sent_length = 0
    train_data_idx = list()    
    for line in sentences:
        clean_line = word_tokenize(line)                     #tokenize line
        line_clean = [word.lower() for word in clean_line]   #concat list to string and make lower
        line_indices = list(map(word_idx.get, line_clean))   #map words to indices
        line_no_none = list(filter(None, line_indices))      #remove None values
        if len(line_no_none) > sent_length:
            sent_length = len(line_no_none)
        train_data_idx.append(line_no_none)

    # Padding to longest sentence length -- Or max length variable if defined
    if not max_length:
        max_length = sent_length
        
    # Padding to longest sentence length
    for idx, cleaned_sent in enumerate(train_data_idx):
        if len(cleaned_sent) < max_length:
            for i in range(max_length - len(cleaned_sent)):
                cleaned_sent.append(0)
        if len(cleaned_sent) > max_length:
            train_data_idx[idx] = train_data_idx[idx][:max_length]

    return torch.tensor(train_data_idx)


def binary_y(y):
    """
    Converts sequence of string labels to binary 1 for positive and 0 for negative tensor.
    """
    
    for i in range(len(y)):
        if y[i] == "positive":
            y[i] = 1
        if y[i] == "negative":
            y[i] = 0
            
    return torch.tensor(y)


if __name__ == '__main__':
    main()
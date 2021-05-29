#%%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nltk import word_tokenize
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        with open("../embeddings/glove.6B.50d.txt", 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
            glove_dict["<PAD>"] = np.zeros((50,))
            glove_dict["<UNK>"] = np.random.normal(scale=0.6, size=(50, ))

        print('loading finished')
        return glove_dict

def get_vocab(sentences):
    print("Making vocab...")
    vocab = []
    word_idx = {}
    idx_word = {}
    indices = 1
    for sentence in sentences:
        sentence = word_tokenize(sentence)
        for word in sentence:
            w = word.lower()
            if w not in vocab:
                vocab.append(w)
                word_idx[w] = indices
                idx_word[indices] = w
                indices += 1

    ## Adding UNK and PAD ##
    vocab.append("<PAD>")
    word_idx["<PAD>"] = len(vocab)
    idx_word[len(vocab)] = "<PAD>"

    vocab.append("<UNK>")
    word_idx["<UNK>"] = len(vocab)
    idx_word[len(vocab)] = "<UNK>"

    print("Vocab made!")

    return vocab, word_idx, idx_word

# Preprocess function
def new_preprocessing(sentences, vocab=None, word_idx=None, idx_word=None, max_length=100):
    print("Preprocessing texts...")

    cleaned_sentence = []

    for sentence in sentences:
        sentence = word_tokenize(sentence)
        sentence = sentence[:max_length]
        idx_sent = []
        for word in sentence:
            w = word.lower()
            try:
                w_idx = word_idx[w]
            except:
                w_idx = word_idx["<UNK>"]
            idx_sent.append(w_idx)
        cleaned_sentence.append(idx_sent)
        
    # Padding to max sentence length
    for idx, cleaned_sent in enumerate(cleaned_sentence):
        if len(cleaned_sent) < max_length:
            for i in range(max_length - len(cleaned_sent)):
                cleaned_sent.append(word_idx["<PAD>"])
            
    print("Done preprocessing texts!")
    return cleaned_sentence

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


def preprocess_to_idx(sentences, max_length=None):
    
    with open("words.txt", 'r', encoding='utf8') as f:
        words = []
        for line in f:
            words.append(line.strip())

    word_idx = dict((word, i) for i, word in enumerate(words)) #create indices from embeddings
        
    #glove = {w: embs[word_idx[w]] for w in embs.keys()}
    
    sent_length = 0
    train_data_idx = []
    for line in sentences:
        clean_line = word_tokenize(line)                     #tokenize line
        line_clean = [word.lower() for word in clean_line]   #concat list to string and make lower
        #line_indices = list(map(word_idx.get, line_clean))   #map words to indices
        line_indices = []
        for word in line_clean:
            if word in word_idx:
                line_indices.append(word_idx[word])
            else:
                line_indices.append(len(word_idx))
        #line_indices = [word_idx[word] for word in line_clean if word in word_idx else len(word_idx)]
        #line_no_none = list(filter(None, line_indices))      #remove None values
        if len(line_indices) > sent_length:
            sent_length = len(line_indices)
        train_data_idx.append(line_indices)

    # Padding to longest sentence length -- Or max length variable if defined
    if not max_length:
        max_length = sent_length
        
    # Padding to longest sentence length
    for idx, cleaned_sent in enumerate(train_data_idx):
        if len(cleaned_sent) < max_length:
            for i in range(max_length - len(cleaned_sent)):
                cleaned_sent.append(400000)
        if len(cleaned_sent) > max_length:
            train_data_idx[idx] = train_data_idx[idx][:max_length]

    print(len(train_data_idx[0]))
    for sent in train_data_idx:
        for word in sent:
            try:
                if type(word) == int:
                    continue
                else:
                    print(type(word))
                    print(word)
            except:
                print(word)
                continue

    return train_data_idx


def build_words(embs):
    vocab = []
    for word, vector in embs.items():
        vocab.append(word.lower())
            
    with open("words.txt", "w", encoding='utf8') as f:
        for word in vocab:
            f.write(word + "\n")

def build_vocab(text):
    vocab = []
    for sentence in text:
        sentence = word_tokenize(sentence)
        for word in sentence:
            if word.lower() not in vocab:
                vocab.append(word.lower())
    
    with open("vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")
#embs = get_embs()
#build_words(embs)

#build_vocab(train['reviewText'])

def create_weight_matrix(embs):
    '''
    Creates weight matrix, where each index is a vector
    If word does not exists, it creates a random vector (np.random.normal(scale=0.6, size=(50, ))
    '''
    with open("vocab.txt", 'r', encoding='utf8') as f:
        vocab = []
        for line in f:
            vocab.append(line.strip())
            
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0

    for i, word in enumerate(vocab):
        try:
            weights_matrix[i] = embs[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(50, ))
    
    return weights_matrix

def create_emb_layer(weights_matrix, non_trainable=False):
    '''
    Creates embeddings for self.embedding in sentiNN
    '''
    
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
    
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


'''
def create_embedding_matrix(word_index, embedding_dict, dimension):
    embedding_matrix = np.zeros((len(word_index)+1, dimension))

    for word,index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix

def process_embs(text, embs, dimension=50):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(split=" ")
    tokenizer.fit_on_texts(text)
    text_token = tokenizer.texts_to_sequences(text)

    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict=embs, dimension=50)
    
    return text_token, embedding_matrix
'''

def binary_y(y):
    """
    Converts sequence of string labels to binary 1 for positive and 0 for negative tensor.
    """
    
    for i in range(len(y)):
        if y[i] == "positive":
            #y[i] = 1
            y.loc[i] = 1
        if y[i] == "negative":
            #y[i] = 0
            y.loc[i] = 0
            
    return torch.tensor(y)

def main():
    pass


if __name__ == '__main__':
    main()
#%%
import re
import gensim
from loader import load_train, load_dev, load_test
from nltk import word_tokenize

def preprocess(sentences):
    """
    Inputs sequence of strings (predictor). Outputs data as word embeddings with fixed length and tensor format.
    """
    
    cleaned_sentences = []
    sent_length = 0
    print('Preprocessing...')
    for sentence in sentences:
        sentence = word_tokenize(sentence) # Preprocessing step: Tokenize
        
        cleaned_sentences.append(sentence)
    print('Done')
    return cleaned_sentences



# Load data
train = load_train()
dev = load_dev()
test = load_test()

# Clean training data
train_x = preprocess(train["reviewText"])

# Create word embedidngs using word2vec
model = gensim.models.Word2Vec(train_x, 
                               min_count=1,           # Ignore words that appear less than this
                               vector_size=100,       # Dimensionality of word embeddings
                               workers=3,             # Number of processors (parallelisation)
                               window=5,              # Context window for words during training
                               epochs=50)             # Number of epochs training over corpus

# We only want to save the word vectors and not the complete model
word_vectors = model.wv
word_vectors.save("word2vec.wordvectors") #save word vectors

# ------- To include into Pytorch ------- #

#load word vectors, memory-mapping (mmap) = read-only
#word_vectors = gensim.models.KeyedVectors.load("word2vec.wordvectors", mmap='r')

#import torch
#import torch.nn as nn
#weights = torch.FloatTensor(word_vectors)
#embedding = nn.Embedding.from_pretrained(weights)
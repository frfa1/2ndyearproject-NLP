#%%
import re
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases, ENGLISH_CONNECTOR_WORDS
from loader import load_train, load_dev, load_test
from nltk import word_tokenize

def preprocess(sentences):
    """
    Inputs sequence of strings (predictor). Outputs data as word embeddings with fixed length and tensor format.
    """
    
    cleaned_sentences = []
    sent_length = 0
    
    for sentence in sentences:
        sentence = word_tokenize(sentence) # Preprocessing step: Tokenize
        
        cleaned_sentences.append(sentence)

    return cleaned_sentences

# Load data
train = load_train()
dev = load_dev()
test = load_test()

# Clean training data
train_x = preprocess(train["reviewText"])

# Create word embedidngs using word2vec
model = Word2Vec(train_x, 
                 min_count=1,   # Ignore words that appear less than this
                 vector_size=100,       # Dimensionality of word embeddings
                 workers=3,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 epochs=50)       # Number of epochs training over corpus

# Save the model
#model.save("word2vec.model")

# The model can then be loaded using
#model = Word2Vec.load("word2vec.model")
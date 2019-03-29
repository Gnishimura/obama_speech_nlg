import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.Error)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Prepare the NLTK stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) #deacc=True removes punc

def tokenize_data(df):
    """Input a dataframe of speech data with column 'speeches'. Return
    a list of tokens"""

    data = df.speeches.values.tolist()
    return data_words = list(sent_to_words(data))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def clean_data(data_words):
    """Input a token list to remove stopwords, make bigrams, and lemmatize."""

    bigram = gensim.models.Phrases(pre_pres_words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # Remove stop words
    data_nostops = remove_stopwords(data_words)

    # Form bigrams
    data_bigrams = make_bigrams(data_nostops)

    # Initialize spacy 'en' model, keep the tagger component for efficiency
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    ppw_lemmatized = lemmatization(ppw_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    return data_lemmatized


def prep_LDA_inputs(doc):
    """Input a document in list form. Output a dictionary, corpus, and tdf for the LDA model."""

    # Create Dictionary
    id2word = corpora.Dictionary(doc)

    # Create Corpus
    texts = doc

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return id2word, texts, corpus



# Create Mallet Model
# mallet_path = 'mallet-2.0.8/bin/mallet'
# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

# # Compute Coherence Score
# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=ppw_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# pprint('\nCoherence Score: ', coherence_ldamallet)

mallet_path = 'mallet-2.0.8/bin/mallet'

def compute_coherence_values(dictionary, corpus, texts, limit, start=4, step=2):
    """Computes c_v coherence for various numbers of topics.
    
    Parameters:
    ------------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    
    Returns:
    ------------
    model_list : List of LDA topic models
    coherence_values: Coherence values corresponding to the LDA model with respective number of topics
    """
    
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        mallet_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(mallet_model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        
    return model_list, coherence_values

# Visualize Topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
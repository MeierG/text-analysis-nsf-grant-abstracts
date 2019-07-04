 # show warnings only the first time code is run

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import gensim
from gensim.models.ldamodel import LdaModel
#from operator import itemgetter, attrgetter

import pyLDAvis
import pyLDAvis.gensim

from operator import itemgetter, attrgetter

import spacy
nlp = spacy.load('en_core_web_sm')

from nltk.tokenize import word_tokenize, sent_tokenize
# importt modules for text analysis
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import nltk
from nltk.corpus import names
# try a naive bayes model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import gensim

stopwords_spacy = spacy.lang.en.stop_words.STOP_WORDS
stopwords = set(nltk.corpus.stopwords.words('english')+list(stopwords_spacy))

def get_tokens(original_text):
    """return list of tokens in abstract
    after removing punctuation and stopwords"""
    tokens = nlp(original_text, disable = ['ner', 'parser'])
    lemmas = [txt.lemma_ for txt in tokens if not rm_punc(txt)]
    stopwords_spacy = spacy.lang.en.stop_words.STOP_WORDS
    stopwords = set(nltk.corpus.stopwords.words('english')+list(stopwords_spacy))
    nlp_abstract = [lemma for lemma in lemmas if lemma not in stopwords]
    #return ' '.join(nlp_abstract)
    return nlp_abstract

import spacy
nlp = spacy.load('en_core_web_sm')

stopwords_spacy = spacy.lang.en.stop_words.STOP_WORDS
stopwords = set(nltk.corpus.stopwords.words('english')+list(stopwords_spacy))

# remove punctuation
def rm_punc(txt):
    return txt.is_punct or txt.is_space # added the token.is_space

# add function to remove breaks
def rm_xlm_tags(text):
    """return text without xml tags"""
    text = text.str.replace('(<br/>)', "")
    text = text.str.replace('(<br>)', "")
    text = text.str.replace('(<a).*(>).*(</a>)', '')
    text = text.str.replace('(&amp)', '')
    text = text.str.replace('(&gt)', '')
    text = text.str.replace('(&lt)', '')
    text = text.str.replace('(\xa0)', ' ')
    text = text.str.replace('<', '')
    return text

def adverb_count(txt):
    """return adverb count"""
    tags = nlp(txt)
    adv = [token.pos_ for token in tags]
    return adv.count('ADV')

def get_tokens(original_text):
    """return list of tokens in abstract
    after removing punctuation and stopwords"""
    tokens = nlp(original_text, disable = ['ner', 'parser'])
    lemmas = [txt.lemma_ for txt in tokens if not rm_punc(txt)]
    stopwords_spacy = spacy.lang.en.stop_words.STOP_WORDS
    stopwords = set(nltk.corpus.stopwords.words('english')+list(stopwords_spacy))
    nlp_abstract = [lemma for lemma in lemmas if lemma not in stopwords]
    #return ' '.join(nlp_abstract)
    return nlp_abstract

def topic_modeling(df):
    df['AbstractNarration'] = rm_xlm_tags(df['AbstractNarration'])
    df['adverb_count'] = df['AbstractNarration'].apply(adverb_count)
    df['nlp_abstract'] = df['AbstractNarration'].apply(get_tokens)
    return df

def get_theme(txt, min_topic_freq=0.05, dictionary='dictionary'):
    """return the most likely topic based on text"""

    new_doc = get_tokens(txt)
    new_doc_bow = dictionary.doc2bow(new_doc)
    main_theme = sorted(lda.get_document_topics(new_doc_bow), key=itemgetter(1), reverse=True)[0]
    #theme_col = [txt for txt in abstract_col.apply(main_theme)]
    #return zip(*theme_col)

    #return main_theme

# get theme and probability in separate columns
def get_theme_and_prob(abstract_col):
    """return the probability of the topic"""

    theme_col = [txt for txt in get_theme(abstract_col)]
    return zip(*theme_col)
    #return theme_col

def find_topics(topic_number = 10):
    """return the topic"""
    topics = lda.print_topics(num_words=1)
    return topics

def ldaModel(df, original_text, get_topics= 10):
    #warnings.filterwarnings('ignore')
    import gensim
    tokens = nlp(df[original_text], disable = ['ner', 'parser'])
    lemmas = [txt.lemma_ for txt in tokens if not rm_punc(txt)]
    stopwords_spacy = spacy.lang.en.stop_words.STOP_WORDS
    stopwords = set(nltk.corpus.stopwords.words('english')+list(stopwords_spacy))
    nlp_abstract = [lemma for lemma in lemmas if lemma not in stopwords]

    dictionary = Dictionary(df[original_text].apply(get_tokens))
    dictionary_from_nlpAbstract = Dictionary(nlp_abstract)
    dictionary_from_nlpAbstract.save('gensim_dict_fromNLPAbstract.gensim')
    corpus = [dictionary.doc2bow(text) for text in nlp_abstract]
    get_topics = 10
    np.random.seed(44)
    lda = gensim.models.ldamodel.LdaModel(corpus, num_topics = get_topics, id2word=dictionary, passes=20)
    lda.save('../models/ldamodel.gensim')
    topics = lda.print_topics(num_words=5)
    return lda


def explore_topic(topic_number, model, topn=5):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
    print (u'{:20} {}'.format(u'term', u'frequency') + u'\n')

    for term, frequency in lda.show_topic(topic_number, topn):
        print (u'{:20} {:.3f}'.format(term, round(frequency, 3)))

def LDAmulticoreModel(df, num_topics=10):

    import warnings
    def fxn():
        warnings.warn('deprecated', DeprecationWarning)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

    dictionary = Dictionary(df['AbstractNarration'].apply(get_tokens))
    dictionary_from_nlpAbstract = Dictionary(df['nlp_abstract'])
    dictionary_from_nlpAbstract.save('gensim_dict_fromNLPAbstract.gensim')
    corpus = [dictionary.doc2bow(text) for text in df['nlp_abstract']]
    # multicore model
    np.random.seed(44)
    lda_multicore = LdaMulticore(corpus, num_topics, id2word=dictionary,workers=4)
    lda_multicore.save('../models/lda_multicoremodel.gensim')
    #print('Topics from LDA Multicore model', lda_multicore.print_topics())
    return lda_multicore


def lda_topic_new_doc(original_text, min_topic_freq=0.05):
    """
    accept the original text of a review and (1) parse it with spaCy,
    (2) apply text pre-proccessing steps, (3) create a bag-of-words
    representation, (4) create an LDA representation, and
    (5) print a sorted list of the top topics in the LDA representation
    """

    # parse the review text with spaCy
    tokens = nlp(original_text, disable = ['ner', 'parser'])
    lemmas = [txt.lemma_ for txt in tokens if not rm_punc(txt)]
    stopwords_spacy = spacy.lang.en.stop_words.STOP_WORDS
    stopwords = set(nltk.corpus.stopwords.words('english')+list(stopwords_spacy))
    nlp_abstract = [lemma for lemma in lemmas if lemma not in stopwords]

    # create a bag-of-words representation
    abstract_bow = dictionary.doc2bow(nlp_abstract)

    # create an LDA representation
    abstract_lda = lda[abstract_bow]

    # sort with the most highly related topics first
    #review_lda = sorted(review_lda, key=lambda (topic_number, freq): -freq)
    main_theme = sorted(lda.get_document_topics(abstract_bow), key=itemgetter(1), reverse=True)[0]

    print(main_theme)
    print('words associated with this topic:\n',lda.print_topic(main_theme[0]))

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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


# convert the dates to date format to analyze funding trends over time
def convert_dates(df, col_date=None):
    """return pandas series in datetime format

    Keyword arguments:
    df -- dataframe
    col_date -- pandas series with date data

    """
    df[col_date] = pd.to_datetime(df[col_date])#, format='%m/%d/%Y')
    df[col_date] = df[col_date].dt.strftime('%m-%d-%Y')
    return df

def cleanDataPhaseI(dfpath):

    with open(dfpath) as f:
        snf_data= pd.read_csv(f)
        print('\nThere are', '{:,}'.format(snf_data.shape[0]), 'grant abstracts in the dataset\n')

    snf = snf_data.copy()

    # set index to the award ID
    snf.set_index('AwardID', inplace=True)

    # remove the records with missing abstracts, less than 10 words and 0 values for the award amount
    snf.dropna(axis = 0, subset = ['AbstractNarration'], inplace=True)
    snf = snf.loc[(snf['AbstractNarration'].str.len()) >10]
    snf = snf.loc[snf['AwardAmount']>500]

    # rename the long name column to a the more desccriptive name
    snf.rename(columns={'LongName': 'NSF_org'}, inplace=True)

    # remove attributtes with over 50% missing values
    snf.drop(columns=['AwardTotalIntnAmount', 'Name.1'], inplace=True)

    # apply function to date columns
    snf = convert_dates(snf, 'AwardEffectiveDate')
    snf = convert_dates(snf, 'AwardExpirationDate')

    # extract year to plot changes over time
    snf['year'] = pd.to_datetime(snf['AwardEffectiveDate']).dt.year
    snf['year']= snf['year'].astype('category')

    # remove award categories
    rm = ['BOA/Task Order', 'Contract Interagency Agreement',
          'Intergovernmental Personnel Award', 'Interagency Agreement']
    snf = snf[~snf['Value'].isin(rm)]

    # remove the one billion dollar award that spans over 11 years
    snf = snf[snf['AwardAmount'] != max(snf['AwardAmount'])]

    # rename directorates as several directorates are listed twice with slightly different names
    snf['NSF_org'].replace({'Direct For Computer & Info Scie & Enginr': 'Directorate for Computer & Information Science & Engineering',
                           'Directorate For Geosciences': 'Directorate for Geosciencess',
                           'Direct For Education and Human Resources': 'Directorate for Education & Human Resources',
                           'Directorate For Engineering': 'Directorate for Engineering',
                           'Direct For Mathematical & Physical Scien': 'Directorate for Mathematical & Physical Sciences',
                           'Direct For Social, Behav & Economic Scie': 'Directorate for Social, Behavioral & Economic Sciences',
                           'Direct For Biological Sciences':'Directorate For Biological Sciences'}, inplace = True)

    print('\nAfter initial cleaning, there are: ', '{:,}'.format(snf.shape[0]), 'grant abstracts in the dataset\n')

    return snf_data, snf

# create a function to get the duration in days
def grant_duration(effective, expiration):
    """return grant duration in days

    Keyword arguments:
    effective -- grant effective date
    expiration -- grant expiration date"""

    diff = (pd.to_datetime(expiration) - pd.to_datetime(effective)).dt.days
    #return diff
    return diff

# get daily amount
def amount_per_duration(df, amount, duration):
    """return daily amount

    Keyword arguments:
    df -- dataset to use
    amount -- amount awarded
    duration -- duration in days
    """
    result = df[amount]/(df[duration])
    #return result
    return result

def cleanDataPhaseII(df):
    # apply grant duration function to each row
    df['grant_duration'] = grant_duration(effective=df['AwardEffectiveDate'], expiration=df['AwardExpirationDate'])

    # set duration of awards with same ffective and expiration date to 1 day.
    df.loc[df['grant_duration'] ==0, 'grant_duration'] = 1

    # apply amount per duration to each row
    df['amount_awarded_per_day'] = amount_per_duration(df, amount='AwardAmount', duration='grant_duration')

    df = df[(df['amount_awarded_per_day'] > 0) & (df['amount_awarded_per_day']<100000)]

    return df

def flesch_score(txt):
    """return the flesch reading ease score"""

    score = textstat.flesch_reading_ease(txt)
    return score

def clean_text(text):
    """Returns column values in lower case
     a copy of the string with the leading
     and trailing characters removed."""

    text = text.replace("(<br/>)", "")
    text = text.replace('(<a).*(>).*(</a>)', '')
    text = text.str.replace('[^\w\s]','')
    text = text.str.replace('(\xa0)', ' ')
    #text = text.replace('[^\w\s]','')
    text = text.str.lower()
    return text

# count the number of sentences in the abstract before removing punctuation
def sentence_count(text):
    """return the number of sentences in abstract"""

    sc = [len(sent_tokenize(txt)) for txt in text]
    return sc


# using all the functions to get the grammar
def grammar_info(df, col):
    """return three separate attributes with
    clean abstract, flesh score and sentence count"""

    df['clean_abstract'] = clean_text(df[col])
    df['flesch_score'] = df[col].apply(flesch_score)
    df['sentence_count'] = sentence_count(df[col])
    return df

# proportion of unique words of the total number of words
def lexical_diversity(unique_wc, total_wc):
    """return the proportion of unique words of
    the total number of words used"""

    ld = unique_wc / total_wc
    return ld

def avg_words_per_sent(sent_count, word_count):
    """return the average number of words per sentence"""
    result = word_count/sent_count
    return result

def avg_word_len(clean_text):
    """use the clean text abstract with no punctuation
    to return aferage word lenght"""

    words = clean_text.split()
    word_len = [len(word)for word in words]

    return int(sum(word_len)/len(words))

def word_count(clean_text):
    """return number of words excluding punctuation"""
    return [len(word_tokenize(txt)) for txt in clean_text]

def unique_words_count(clean_text):
    """return number of unique words in the text
    excluding punctuation"""
    unique_wc = [len(set(word_tokenize(txt))) for txt in clean_text]
    return unique_wc

def semmantics(txt, clean_txt):
    """concatenates three new attributes to dataframe
    with word count, average word lenght and unique words"""

    txt['word_count'] = word_count(txt[clean_txt])
    txt['avg_word_len'] = txt[clean_txt].apply(avg_word_len)
    txt['unique_words'] = unique_words_count(txt[clean_txt])
    #txt['lexical_diversity'] = lexical_diversity(txt['unique_words'], txt['word_count'])
    return txt

def get_grammar(df):
    df = grammar_info(df, 'AbstractNarration')
    df = semmantics(df, 'clean_abstract')
    df['avg_words_per_sent'] = avg_words_per_sent(df['sentence_count'], df['word_count'])
    df = df.loc[df['word_count'] >10]
    df['lexical_diversity'] = lexical_diversity(df['unique_words'], df['word_count'])
    return df



def gender_identification(f_name):
    """return gender based on last letter of first name
        and lenght of name"""
    return {'last_letter': f_name[-1], 'lenght':len(f_name)}


def gender_classifier(first_name):
    #if True: nltk.download('names')

    # separate male and femal names
    male_names = [n for n in names.words('male.txt')]
    female_names = [n for n in names.words('female.txt')]

    #create a list of tuples with the name and the gender
    labeled_names = ([(name.lower(), 'male') for name in male_names] + [(name.lower(), 'female') for name in female_names])

    # randomly shuffle the names
    np.random.seed(44)
    random.shuffle(labeled_names)

    X,y = list(zip(*labeled_names))

    # create a tuple
    gender_sets = [(gender_identification(f_name), gender) for (f_name, gender) in labeled_names]

    np.random.seed(44)

    #split the list into even parts
    train_set, test_set = gender_sets[:int(len(gender_sets)*.7)], gender_sets[int(len(gender_sets)*.7):]

    # build a classifier
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    #print('Accuracy on unseen data: ', nltk.classify.accuracy(classifier, test_set))
    pred_accuracy= nltk.classify.accuracy(classifier, test_set)
    pred= classifier.classify(gender_identification(first_name))

    # try on a new name and accuracy of the model)
    #print('\nThe model estimates that ', first_name, 'is a', pred, 'name')
    return pred

def gender_sklearn(first_name):
    tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=5, norm='l2',
                        encoding='latin-1',
                        ngram_range=(1, 2), stop_words='english')
        # separate male and femal names
    male_names = [n for n in names.words('male.txt')]
    female_names = [n for n in names.words('female.txt')]

        #create a list of tuples with the name and the gender
    labeled_names = ([(name.lower(), 'male') for name in male_names] + [(name.lower(), 'female') for name in female_names])

        # randomly shuffle the names
    np.random.seed(44)
    random.shuffle(labeled_names)

    X,y = list(zip(*labeled_names))

    features = tfidf.fit_transform(X).toarray()
    labels = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)


    pred = clf.predict(count_vect.transform(y_test))
    print("Accuracy on unseen data:",metrics.accuracy_score(pred, y_test))
    np.mean(pred == y_test)
    pred_gender = clf.predict(count_vect.transform([first_name]))
    print('The model estimates that', first_name, 'is a', pred_gender, 'name')

def get_signer_name(text):
    """return the gender of program officer based
    on first name"""
    txt= text.lower().split()[0]
    name = gender_identification(txt)
    #gender = classifier.classify(name)
    gender = gender_classifier(txt)
    return gender

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

def get_theme(txt, min_topic_freq=0.05):
    """return the most likely topic based on text"""

    new_doc = get_tokens(txt)
    new_doc_bow = dictionary.doc2bow(new_doc)
    main_theme = sorted(lda.get_document_topics(new_doc_bow), key=itemgetter(1), reverse=True)[0]
    return main_theme

# get theme and probability in separate columns
def get_theme_and_prob(abstract_col):
    """return the probability of the topic"""

    theme_col = [txt for txt in abstract_col.apply(get_theme)]
    return zip(*theme_col)
    #return theme_col

def find_topics(topic_number = 10):
    """return the topic"""
    topics = lda.print_topics(num_words=1)
    return topics

def explore_topic(topic_number, topn=5):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
    print (u'{:20} {}'.format(u'term', u'frequency') + u'\n')

    for term, frequency in lda.show_topic(topic_number, topn):
        print (u'{:20} {:.3f}'.format(term, round(frequency, 3)))

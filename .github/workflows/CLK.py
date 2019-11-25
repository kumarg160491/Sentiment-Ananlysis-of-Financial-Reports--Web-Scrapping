#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:05:59 2019

@author: kumar
"""
###########################
'''Importing Libraries'''
###########################

import pandas as pd
import numpy as np
#!pip install nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps=PorterStemmer()

##########################
'''Loading Data'''
##########################

data=pd.read_excel('/home/kumar/Downloads/cik_list.xlsx')
data.head()

###################################
'''reading URL from column'''
##################################
from urllib.request import urlopen
my_files=[]
for link in data['SECFNAME_URL']:  
    with urlopen(link) as response:
        myfile = response.read()
        print(myfile)
        my_files.append(myfile)
####################################

print(my_files)

#my_files = (my_files.decode('utf-8', 'ignore') )

#my_files = ' '.join([str(my_files) for my_files in my_files])
type(my_files)

#################################
'''Converting data to string'''
#################################
my_files=[x.decode('utf-8') for x in my_files]


#################################
'''Converted data to string'''
#################################
type(my_files)

#################################
'''Verifying data'''
#################################

my_files[0]

import requests
#url="https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-000413.txt"
r=requests.get(my_files)
text=r.my_files
#################################
'''Loading Punkt from nltk'''
#################################
nltk.download('punkt')

#################################
'''Importing Word Lemmatizer'''
#################################
from nltk.stem import WordNetLemmatizer
import re
wnl=WordNetLemmatizer()

#################################
'''Tokenize data'''
#################################
sentence=nltk.sent_tokenize(my_files)

####################################
'''Cleaning data using Stopwords'''
####################################

corpus=[]
for i in range(len(sentence)):
    review= re.sub("[^a-zA-Z]"," ", sentence[i])
    review=review.lower()  
    review=review.split()
    review=[wnl.lemmatize(word) for word in review if not word in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)

############################
'''counting of words'''
###########################
   
def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

print( word_count(corpus[0]))


#################################
'''Total word count'''
#################################
for new_corpus in corpus:
    new_corpus=print( word_count(new_corpus))

#################################
'''Bag of words'''
#################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
rfr=RandomForestClassifier(n_estimators=200,criterion='entropy')
cv=CountVectorizer(ngram_range=(2,2))
cv.fit_transform(corpus).toarray()


import seaborn as sns
import matplotlib.pyplot as plt


plt.hist(data['FYRMO'])

from sklearn.preprocessing import MinMaxScaler
ss=MinMaxScaler(feature_range=(-1,1),copy=True)
ss.fit_transform(data['FYRMO'])

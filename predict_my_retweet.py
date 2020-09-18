# -*- coding: utf-8 -*-

# Basic packages
import pandas as pd 
import numpy as np
import re
import os
import collections
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import regex
import math
import random
import json

# Packages for data preparation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Packages for modeling
from keras import models
from keras import optimizers
from keras.models import Model
from keras import layers
from keras.layers import Dense, Embedding, LSTM, Input, SimpleRNN, TimeDistributed, Concatenate, BatchNormalization, LeakyReLU
from keras import regularizers
from keras import backend as K
from keras import regularizers

import nltk
nltk.download('stopwords')

path = os.path.dirname(os.path.realpath(__file__))

# take input
tweet_text = input("Enter tweet :") 
friends_count = int(input("Enter friends_count :"))
followers_count = int(input("Enter followers_count :"))
account_age = int(input("Enter account_age in MONTHS :"))
total_tweet_count = int(input("Enter total_tweet_count :"))
favourited_tweet_count = int(input("Enter favourited_tweet_count :"))
user_info_input = [friends_count, followers_count, account_age, total_tweet_count, favourited_tweet_count]

"""Tweet text pre-processing"""

FLAGS = re.MULTILINE | re.DOTALL | regex.VERSION1

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + regex.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()

NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
VAL_SIZE = 1000  # Size of the validation set
NB_START_EPOCHS = 2000  # Number of epochs we usually start to train with
BATCH_SIZE = 64  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 26  # Maximum number of words in a sequence
GLOVE_DIM = 100  # Number of dimensions of the GloVe word embeddings
LSTM_OUT = 512  # output dimension of language model lstm
NORMALIZE_TO = 100  # normalize the value of features between 0 to NORMALIZE_TO
RETWEETS_NORM_TO = 10
HOURS = 72  # number of hours the dataset was recorded for
RANDOM_NUM = random.randint(0,100)

def remove_stopwords(input_text):
    '''
    Function to remove English stopwords from a Pandas Series.
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 
    
def remove_mentions(input_text):
    '''
    Function to remove mentions, preceded by @, in a Pandas Series
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    return re.sub(r'@\w+', '', input_text)

df = pd.read_csv(path+'/user_info_with_age.csv') 
df = df[['tweet_id', 'text', 'friends_count', 'followers_count', 'account_age', 'total_tweet_count', 'favourited_tweet_count']]      # X, y
df.text = df.text.apply(tokenize).apply(remove_stopwords).apply(remove_mentions)

# find maximum
dg = pd.read_csv(path+'/retweet_count_new_unnormalized.csv')
dg = dg[[str(HOURS)]]
max_retweet_count = int(dg.max())

# user account features
dh = pd.read_csv(path+'/user_info_with_age.csv') 
user_featuers = ['friends_count', 'followers_count', 'account_age', 'total_tweet_count', 'favourited_tweet_count']

features = []
max_feature_values = []
user_feature_count = len(user_featuers)
for i in range(user_feature_count-1):
    for j in range(i+1,user_feature_count):
        feature = dh[user_featuers[i]]*dh[user_featuers[j]]
        max_feature_value = feature.max()
        max_feature_values.append(max_feature_value)
        feature = (feature/max_feature_value)*NORMALIZE_TO
        features.append(feature)

features_count = len(features)
features = pd.concat(features, axis=1)

X_train, X_test, u_train, u_test = train_test_split(df.text, features, test_size=0.1, random_state=RANDOM_NUM)

seq_lengths = X_train.apply(lambda x: len(x.split(' ')))
tweet_stats = seq_lengths.describe()
MAX_LEN = int(tweet_stats['max'])

tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=" ")
tk.fit_on_texts(X_train)      # creates a internal dictionary


"""Creating a Dictionary"""
print('creating Dictionary...')
glove_file = path+'/glove.twitter.27B.100d.txt'
# glove_file = 'customWE_100d.txt'
emb_dict = {}
glove = open(glove_file)
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector
glove.close()

f = open(path+'/keywords.txt')
keywords = []
for line in f:
    word = line.split('  ')[0]
    word = tokenize(word)
    keywords.append(word)

custom_glove_file = path+'/custom_WE.txt'
custom_glove = open(custom_glove_file)
for line in custom_glove:
    values = line.split()
    word = values[0]
    if word in keywords:
        vector = np.asarray(values[1:], dtype='float32')
        emb_dict[word] = vector
custom_glove.close()

emb_matrix = np.zeros((NB_WORDS, GLOVE_DIM))
for w, i in tk.word_index.items():
    # The word_index contains a token for all words of the training data so we need to limit that
    if i < NB_WORDS:
        vect = emb_dict.get(w)
        # Check if the word from the training data occurs in the GloVe word embeddings
        # Otherwise the vector is kept with only zeros
        if vect is not None:
            emb_matrix[i] = vect
    else:
        break

"""Building Model"""

# This is for training
encoder_inputs = Input(shape=(MAX_LEN, ), name='input_1')
embedding = Embedding(NB_WORDS, GLOVE_DIM, name='embedding_1')
embedding_inputs = embedding(encoder_inputs)
encoder = LSTM(LSTM_OUT, dropout_U = 0.3, dropout_W = 0.3, name='lstm_1', kernel_regularizer=regularizers.l2(0.05))
lstm_output = encoder(embedding_inputs)
user_info_inputs = Input(shape=(features_count,), name='input_2')
dense1 = Dense(units=128, name='dense_1', kernel_regularizer=regularizers.l2(0.05))     # Whc
full_info = Concatenate(name='concatenate_1')([lstm_output, user_info_inputs])
encoder_output_dense1 = dense1(full_info)

decoder_inputs = Input(shape=(None, 1), name='input_3')

# dynamic RNN
decoder_rnn = SimpleRNN(128, return_sequences=True, return_state=True, name='rnn_1', kernel_regularizer=regularizers.l2(0.05))
time_distributed = TimeDistributed(Dense(1, activation='relu'), name='time_distributed_1')

# decoder_outputs 
# We are passing encoder_output as the hidden state of dynamic RNN
decoder_outputs, _ = decoder_rnn(decoder_inputs, initial_state=encoder_output_dense1)
decoder_outputs = time_distributed(decoder_outputs)
final_model = Model([encoder_inputs, user_info_inputs, decoder_inputs], decoder_outputs)

# weights of encoder is already there in decoder, hence we dont need to call it saperatly.
final_model.load_weights(path+'/saved_models/final_model.h5', by_name=True)

encoder_model = Model([encoder_inputs, user_info_inputs], encoder_output_dense1)
decoder_state_input = Input(shape=(128,),name='input_4')
decoder_outputs, decoder_state = decoder_rnn(decoder_inputs, initial_state=decoder_state_input)
decoder_outputs = time_distributed(decoder_outputs)
decoder_model = Model([decoder_inputs, decoder_state_input], [decoder_outputs] + [decoder_state])

def decode_sequence(input_seq,user_info_input):
    input_seq = tokenize(remove_stopwords(remove_mentions(input_seq)))
    input_seq = tk.texts_to_sequences(pd.Series(input_seq))
    input_seq = pad_sequences(input_seq, maxlen=MAX_LEN)
 
    feature_count = 0 
    features = []   
    for i in range(user_feature_count-1):
        for j in range(i+1,user_feature_count):
            feature = user_info_input[i]*user_info_input[j]
            feature = (feature/max_feature_values[feature_count])*NORMALIZE_TO
            features.append(feature)
            feature_count += 1

    features = np.array([features])
    state_value = encoder_model.predict([input_seq, features])
    
    target = 0
    target_list = []
    for t in range(1,HOURS+1):
        targets = np.array([[[target]]])
        targets, state_value = decoder_model.predict([targets,state_value])
        target = targets[0][0][0]
        target = (target*max_retweet_count)/RETWEETS_NORM_TO   # unnormalize number of retweets
        target_list.append(target)
        target = max(math.ceil(target)-1, math.floor(target))
        target = (target/max_retweet_count)*RETWEETS_NORM_TO   # normalize number of retweets

    return target_list  

# tweet_text = 'Anyone with coronavirus symptoms can book an appointment at a regional testing site.   John discusses how he suppor… https://t.co/o2CDWInYPB'
# user_info_input = [818,690640,133,13594,514]

predicted_y = decode_sequence(tweet_text, user_info_input)
print('predicted retweets: ',predicted_y)

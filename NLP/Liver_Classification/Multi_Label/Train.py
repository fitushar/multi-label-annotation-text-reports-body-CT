from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
np.random.seed(42)
import datetime
'''
Author: Fakrul Islam Tushar (Source Code is from Krystina, Duke Summer Intern)
        email: ft42@duke.edu,f.i.tushar.eee@gmail.com
        Date: 8/24/2020,Durham,NC.
        Implementation: Tensorflow 2.0

-Edits: 8/24/2020--Have Completed the Chnages the Code Structure.

'''

import tensorflow as tf
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

##-----tospecify the gpu usages----###3
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

import gensim
from gensim.models.keyedvectors import KeyedVectors
import re
import io
import pandas as pd
import string
import random
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.corpus import stopwords
from tensorflow.keras.optimizers import Adam
from config import*
from model import Attention_BiLSTM_WithPTEM
from model import Attention_BiLSTM

#--------------Reading---Data--FUNCTION----
#----Punctuation list---I just Removed this two [',', '.']
puncts = ['"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', 'â€¢',  '~', '@', 'Â£',
     'Â·', '_', '{', '}', 'Â©', '^', 'Â®', '`',  '<', 'â†’', 'Â°', 'â‚¬', 'â„¢', 'â€º',  'â™¥', 'â†', 'Ã—', 'Â§', 'â€³', 'â€²', 'Ã‚', 'â–ˆ', 'Â½', 'Ã ', 'â€¦',
     'â€œ', 'â˜…', 'â€', 'â€“', 'â—', 'Ã¢', 'â–º', 'âˆ’', 'Â¢', 'Â²', 'Â¬', 'â–‘', 'Â¶', 'â†‘', 'Â±', 'Â¿', 'â–¾', 'â•', 'Â¦', 'â•‘', 'â€•', 'Â¥', 'â–“', 'â€”', 'â€¹', 'â”€',
     'â–’', 'ï¼š', 'Â¼', 'âŠ•', 'â–¼', 'â–ª', 'â€ ', 'â– ', 'â€™', 'â–€', 'Â¨', 'â–„', 'â™«', 'â˜†', 'Ã©', 'Â¯', 'â™¦', 'Â¤', 'â–²', 'Ã¨', 'Â¸', 'Â¾', 'Ãƒ', 'â‹…', 'â€˜', 'âˆž',
     'âˆ™', 'ï¼‰', 'â†“', 'ã€', 'â”‚', 'ï¼ˆ', 'Â»', 'ï¼Œ', 'â™ª', 'â•©', 'â•š', 'Â³', 'ãƒ»', 'â•¦', 'â•£', 'â•”', 'â•—', 'â–¬', 'â¤', 'Ã¯', 'Ã˜', 'Â¹', 'â‰¤', 'â€¡', 'âˆš', ]

#---Function to read Data
def Get_ReportText_and_Labels(csv,nb_classes):
    load_data = pd.read_csv(csv)
    reports = list(load_data[REPORT_TEXT_COLUMN_NAME]) #Train_Data
    if nb_classes == 1:
        reports_labels  =load_data[LABELS_COLUMB_BINARY_LABEL_NAME].values
        np.where(reports_labels==0, 5, reports_labels)
        np.where(reports_labels==1, 0, reports_labels)
        np.where(reports_labels==5, 1, reports_labels)
    elif nb_classes > 1:
        reports_labels  =load_data[LABELS_COLUMN_NAMES].values #--Getting Labels
    return reports,reports_labels

##--clean Punctuation
def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, " ")
    return x

#---Cleaning the numbers
def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        x = re.sub('[0-9]{1}', '#', x)
    return x

def Text_Cleaning(text_data):
    # Some preprocesssing that will be common to all the text classification methods you will see.
    for report in range(0,len(text_data)):
        text_data[report] = clean_text(text_data[report])
        text_data[report] = clean_numbers(text_data[report])
    return text_data



def Training():

    ##logfile
    logdir = os.path.join(LOG_NAME, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    ##csv_logger
    csv_logger = tf.keras.callbacks.CSVLogger(TRAINING_CSV)
    ##Model-checkpoings
    path=TRAINING_SAVE_MODEL_PATH
    model_path=os.path.join(path, MODEL_SAVING_NAME)
    Model_callback= tf.keras.callbacks.ModelCheckpoint(filepath=model_path,save_best_only=False,save_weights_only=True,
                                                       monitor=ModelCheckpoint_MOTITOR,verbose=1)

    #----Loading Traing and Validation Data-----
    X_train, y_train = Get_ReportText_and_Labels(TRAIN_CSV,NUMBER_OF_CLASSES)
    X_val,  y_val  = Get_ReportText_and_Labels(VAL_CSV,NUMBER_OF_CLASSES)

    # Cleaning Numbers and Punctuation
    X_train=Text_Cleaning(X_train)
    X_val=Text_Cleaning(X_val)

    ##---Making Tokenizer from Training Data
    #Tokenizing text
    MAX_NB_WORDS =80000
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X_train)
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    #--Apply Tokenizer to Train And Validation
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)


    # A dictionary mapping words to an integer index
    word_index = tokenizer.word_index

    # The first indices are reserved
    word_index["<PAD>"] = 0

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # Apply Padding to X
    X_train_pad = pad_sequences(X_train_seq, MAX_WORDS)
    X_val_pad = pad_sequences(X_val_seq, MAX_WORDS)
    # Print shapes
    print("Shape of X: {}".format(X_train_pad.shape))

    if USING_PRE_TRAINED_EMBADDING == True:
        print('Using Pretraining Embaddinf---{}'.format(PRE_TRAINING_EMBADDING))
        vocab_size = len(tokenizer.word_index) + 1
        print("loading word2vec model")
        word2vec_model = KeyedVectors.load_word2vec_format(PRE_TRAINING_EMBADDING, binary=True)
        def getVector(str):
            if str in word2vec_model:
                return word2vec_model[str]
            else:
                return None;
        def isInModel(str):
            return str in word2vec_model

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, 200))
        for word, i in tokenizer.word_index.items():
            embedding_vector = getVector(word)
            if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

    #-----FinalData---Loading---
    X_train_cv = X_train_pad
    y_train_cv = y_train
    X_valid_cv = X_val_pad
    y_valid_cv = y_val


    #----Buiding the model
    if RESUME_TRAINING==1:
        if USING_PRE_TRAINED_EMBADDING == True:
            nlp_model=Attention_BiLSTM_WithPTEM(nb_classes=NUMBER_OF_CLASSES,max_words=MAX_WORDS,word_index=word_index,embedding_matrix=embedding_matrix,
                                                embedding_dim=EMBADDING_DIMENTION,pretraining_embadding=USING_PRE_TRAINED_EMBADDING)
        else:
            nlp_model=Attention_BiLSTM(nb_classes=NUMBER_OF_CLASSES,max_words=MAX_WORDS,word_index=word_index,
                                       embedding_dim=EMBADDING_DIMENTION,pretraining_embadding=USING_PRE_TRAINED_EMBADDING)


        nlp_model.load_weights(RESUME_TRAIING_MODEL)
        initial_epoch_of_training=TRAINING_INITIAL_EPOCH
        print('Resume-Training From-Epoch{}-Loading-Model-from_{}'.format(initial_epoch_of_training,RESUME_TRAIING_MODEL))
        nlp_model.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[TRAIN_CLASSIFY_METRICS,tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
        nlp_model.summary()

    else:
        initial_epoch_of_training=0
        if USING_PRE_TRAINED_EMBADDING == True:
            nlp_model=Attention_BiLSTM_WithPTEM(nb_classes=NUMBER_OF_CLASSES,max_words=MAX_WORDS,word_index=word_index,embedding_matrix=embedding_matrix,
                                                embedding_dim=EMBADDING_DIMENTION,pretraining_embadding=USING_PRE_TRAINED_EMBADDING)
        else:
            nlp_model=Attention_BiLSTM(nb_classes=NUMBER_OF_CLASSES,max_words=MAX_WORDS,word_index=word_index,
                                       embedding_dim=EMBADDING_DIMENTION,pretraining_embadding=USING_PRE_TRAINED_EMBADDING)

        nlp_model.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[TRAIN_CLASSIFY_METRICS,tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
        nlp_model.summary()


    nlp_model.fit(X_train_cv,y_train_cv,
               epochs=TRAING_EPOCH,
               batch_size=BATCH_SIZE,
               shuffle=SHUFFLE,
               initial_epoch=initial_epoch_of_training,
               validation_data = (X_valid_cv, y_valid_cv),
               callbacks=[tensorboard_callback,csv_logger,Model_callback])

    e = nlp_model.layers[1]
    weights = e.get_weights()[0]
    print(weights.shape) #shape:(vocab_size, embedding_dim)

    ############################Saving the Embading for visualization##########################
    out_v = io.open('vecsct.tsv', 'w', encoding='utf-8')
    out_m = io.open('metact.tsv', 'w', encoding='utf-8')
    for word_num in range(0, len(word_index)):
      word = reverse_word_index[word_num]
      embeddings = weights[word_num]
      out_m.write(word + "\n")
      out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()

if __name__ == '__main__':
   Training()

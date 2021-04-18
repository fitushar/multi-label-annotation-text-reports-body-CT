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
from deploy_config import*
from model import Attention_BiLSTM_WithPTEM
from model import Attention_BiLSTM

#--------------Reading---Data--FUNCTION----
#----Punctuation list---I just Removed this two [',', '.']
puncts = PUNCTUATION

#---Function to read Data
def Get_ReportText_and_Labels_Predict(csv,nb_classes):
    load_data = pd.read_csv(csv)
    reports = list(load_data[REPORT_TEXT_COLUMN_NAME]) #Train_Data
    subject_id= list(load_data[SUBJECT_ID_COLUMN_NAME])
    if nb_classes == 1:
        reports_labels  =load_data[LABELS_COLUMB_BINARY_LABEL_NAME].values
        np.where(reports_labels==0, 5, reports_labels)
        np.where(reports_labels==1, 0, reports_labels)
        np.where(reports_labels==5, 1, reports_labels)
    elif nb_classes > 1:
        reports_labels  =load_data[LABELS_COLUMN_NAMES].values #--Getting Labels
    return reports,reports_labels,subject_id

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



def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def attention2color(attention_score,max_at,min_at):

    #attention_score=(attention_score-min_at)/(max_at-min_at)
    #print(attention_score)
    r = 255-int(attention_score*255)
    print(r)
    color = rgb_to_hex((255, r, r))
    return str(color)



def Predict_nlp():

    #----Loading Traing and Validation Data-----
    X_val1,  y_val, val_subject  = Get_ReportText_and_Labels_Predict(TEST_CSV,NUMBER_OF_CLASSES)
    X_val=Text_Cleaning(X_val1)

    # loading
    with open(TOKENIZER_PICKLE, 'rb') as handle:
        tokenizer = pickle.load(handle)
    #--Apply Tokenizer to Train And Validation
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    word_index = tokenizer.word_index
    word_index["<PAD>"] = 0
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    X_val_pad = pad_sequences(X_val_seq, MAX_WORDS)
    print("Shape of X: {}".format(X_val_pad.shape))

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

    #----Buiding the model

    if USING_PRE_TRAINED_EMBADDING == True:
        nlp_model=Attention_BiLSTM_WithPTEM(nb_classes=NUMBER_OF_CLASSES,max_words=MAX_WORDS,word_index=word_index,embedding_matrix=embedding_matrix,
                                                embedding_dim=EMBADDING_DIMENTION,pretraining_embadding=USING_PRE_TRAINED_EMBADDING)
    else:
        nlp_model=Attention_BiLSTM(nb_classes=NUMBER_OF_CLASSES,max_words=MAX_WORDS,word_index=word_index,
                                       embedding_dim=EMBADDING_DIMENTION,pretraining_embadding=USING_PRE_TRAINED_EMBADDING)


    nlp_model.load_weights(MODEL_WIGHT)
    print('Loading Model Weight from:{}'.format(MODEL_WIGHT))
    nlp_model.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[TRAIN_CLASSIFY_METRICS,tf.keras.metrics.Accuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    nlp_model.summary()

    model_with_attentions = tf.keras.Model(inputs=nlp_model.input,outputs=[nlp_model.output,nlp_model.get_layer('attention_vec').output])

    Report_name=[]
    label=[]
    prediction_list=[]
    pred0=[]
    pred1=[]
    pred2=[]
    pred3=[]
    pred4=[]
    lbl0=[]
    lbl1=[]
    lbl2=[]
    lbl3=[]
    lbl4=[]



    for i in range(len(X_val_pad)):
    #for i in range(0,1000):

        rio_repo=X_val_pad[i]
        rio_repo = rio_repo.reshape((1,MAX_WORDS))
        lbl=y_val[i]
        tokenized_sample = text_to_word_sequence(X_val1[i])
        #predict_model=nlp_model.predict(rio_repo)

        #--|Attention|---
        predict_model, attentions = model_with_attentions.predict(rio_repo)
        max_attention_score=max(max(np.sqrt(attentions)))
        min_attention_score=min(min(np.sqrt(attentions)))
        # Get word attentions using attenion vector
        token_attention_dic = {}
        toks=[]
        atts=[]
        max_score = 0.0
        min_score = 0.0
        for token, attention_score in zip(tokenized_sample, attentions[0][-len(tokenized_sample):]):
            toks.append(token)
            atts.append(np.sqrt(attention_score))
        html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
        #for token, attention in token_attention_dic.items():
        for iterator in range(len(toks)):
            token=toks[iterator]
            attention = atts[iterator]
            html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention,max_attention_score,min_attention_score),token)

        html_text += "</p>"
        # Display text enriched with attention scores
        Html_file= open(SAVING_HTML_FILE+"report_test_"+str(val_subject[i])+"_.html","w")
        Html_file.write((html_text))
        Html_file.close()
        print('{}---Subject_id={}---{}-lbl-{}'.format(i,val_subject[i],predict_model[0],lbl))

        Report_name.append(val_subject[i])
        label.append(lbl)
        prediction_list.append(predict_model[0])
        pred0.append(predict_model[0][0])
        pred1.append(predict_model[0][1])
        pred2.append(predict_model[0][2])
        pred3.append(predict_model[0][3])
        pred4.append(predict_model[0][4])
        lbl0.append(lbl[0])
        lbl1.append(lbl[1])
        lbl2.append(lbl[2])
        lbl3.append(lbl[3])
        lbl4.append(lbl[4])

    image_data=pd.DataFrame(list(zip(Report_name,label,prediction_list,pred0,pred1,pred2,pred3,pred4,lbl0,lbl1,lbl2,lbl3,lbl4)),columns=['id','lbl','predict','pred0','pred1','pred2','pred3','pred4','lbl0','lbl1','lbl2','lbl3','lbl4'])
    image_data.to_csv(SAVING_CSV,encoding='utf-8', index=False)


if __name__ == '__main__':
   Predict_nlp()

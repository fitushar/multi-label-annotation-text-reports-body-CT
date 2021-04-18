'''
Author: Fakrul Islam Tushar (Source Code is from Krystina, Duke Summer Intern)
email: ft42@duke.edu,f.i.tushar.eee@gmail.com
Date: 8/24/2020,Durham,NC.
Implementation: Tensorflow 2.0
'''
import random
import tensorflow as tf
import numpy as np
np.random.seed(42)

def Attention_BiLSTM_WithPTEM(nb_classes,max_words,word_index,embedding_matrix,embedding_dim=200,pretraining_embadding=True):

    # The dimension of word embeddings & Input
    embedding_dim = embedding_dim
    sequence_input = tf.keras.Input(shape=(max_words,), dtype='int32')

    # Word embedding layer
    if (pretraining_embadding==True):
        embedded_inputs =tf.keras.layers.Embedding(len(word_index) + 1,embedding_dim,weights=[embedding_matrix], trainable=True,input_length=max_words)(sequence_input)
    else:
        embedded_inputs =tf.keras.layers.Embedding(len(word_index) + 1,embedding_dim,input_length=max_words)(sequence_input)

    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True))(embedded_inputs)
    lstm_outs = tf.keras.layers.Dropout(0.2)(lstm_outs)

    # Attention Mechanism - Generate attention vectors
    input_dim = int(lstm_outs.shape[2])
    permuted_inputs  = tf.keras.layers.Permute((2, 1))(lstm_outs)
    attention_vector = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(lstm_outs)
    attention_vector = tf.keras.layers.Reshape((max_words,))(attention_vector)
    attention_vector = tf.keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = tf.keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

    ###----Flatten and classification
    fc = tf.keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
    if nb_classes == 1:
        output = tf.keras.layers.Dense(1, activation='sigmoid')(fc)
    elif nb_classes > 1:
        output = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(fc)

    # Finally building model
    model = tf.keras.Model(inputs=[sequence_input], outputs=output)
    return model

def Attention_BiLSTM(nb_classes,max_words,word_index,embedding_dim=200,pretraining_embadding=False):

    # The dimension of word embeddings & Input
    embedding_dim = embedding_dim
    sequence_input = tf.keras.Input(shape=(max_words,), dtype='int32')

    # Word embedding layer
    if (pretraining_embadding==True):
        embedded_inputs =tf.keras.layers.Embedding(len(word_index) + 1,embedding_dim,weights=[embedding_matrix], trainable=True,input_length=max_words)(sequence_input)
    else:
        embedded_inputs =tf.keras.layers.Embedding(len(word_index) + 1,embedding_dim,input_length=max_words)(sequence_input)

    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True))(embedded_inputs)
    lstm_outs = tf.keras.layers.Dropout(0.2)(lstm_outs)

    # Attention Mechanism - Generate attention vectors
    input_dim = int(lstm_outs.shape[2])
    permuted_inputs  = tf.keras.layers.Permute((2, 1))(lstm_outs)
    attention_vector = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(lstm_outs)
    attention_vector = tf.keras.layers.Reshape((max_words,))(attention_vector)
    attention_vector = tf.keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = tf.keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

    ###----Flatten and classification
    fc = tf.keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
    if nb_classes == 1:
        output = tf.keras.layers.Dense(1, activation='sigmoid')(fc)
    elif nb_classes > 1:
        output = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(fc)

    # Finally building model
    model = tf.keras.Model(inputs=[sequence_input], outputs=output)
    return model

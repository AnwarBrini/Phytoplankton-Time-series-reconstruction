# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:26:06 2019
@author: anwar
"""
from keras.layers import Dense, LeakyReLU, Dropout, LSTM, Bidirectional, TimeDistributed
from keras.models import Sequential
from keras.regularizers import l1
from keras.optimizers import Adam

def mlp_4layers(input_dim=13):
	# create model
    model = Sequential()
    model.add(Dense(143, input_dim=input_dim))
    model.add(LeakyReLU())
   
    model.add(Dense(66, input_dim=143))
    model.add(LeakyReLU())

    model.add(Dense(33, input_dim=66))
    model.add(LeakyReLU())
    model.add(Dropout(0.15))
    
    model.add(Dense(1, input_dim=33, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse', 'mae',])
    return model


def mlp_1000_500(input_dim=13):
    
    model = Sequential()
   
    model.add(Dense(1000, input_dim=input_dim))
    model.add(LeakyReLU())
    
    model.add(Dense(1000, input_dim=1000))
    model.add(LeakyReLU())

    model.add(Dense(500, input_dim=1000))
    model.add(LeakyReLU())

    model.add(Dense(500, input_dim=500))
    model.add(LeakyReLU())
    model.add(Dropout(0.15))
    
    model.add(Dense(120, input_dim=500))
    model.add(LeakyReLU())
    model.add(Dropout(0.15))
    
    model.add(Dense(120, input_dim=120))
    model.add(LeakyReLU())
    model.add(Dropout(0.15))
    
    model.add(Dense(1, input_dim=120))
	# Compile model
	# Compile model
    model.compile(loss='mse', optimizer="Adam",metrics=['mse', 'mae',])
    return model

def lstm_layers():
    model = Sequential()
    model.add(LSTM(500, input_shape=(1,13),return_sequences=True))
    model.add(LeakyReLU())
    model.add(LSTM(500,return_sequences=True))
    model.add(LeakyReLU())
    model.add(LSTM(230,return_sequences=True))
    model.add(LeakyReLU())
    model.add(LSTM(120,return_sequences=True))
    model.add(LeakyReLU())
    model.add(Dropout(0.15))
    model.add(LSTM(120))
    model.add(LeakyReLU())
    model.add(Dropout(0.15))
    model.add(Dense(1))
	# Compile model
    model.compile(loss='mse', optimizer="Adam",metrics=['mse', 'mae',])
    return model


def lstm_bid():
    model = Sequential()
    model.add(Bidirectional(LSTM(500, input_shape=(1,13),return_sequences=True)))
    model.add(LeakyReLU())
    model.add(Bidirectional(LSTM(500,dropout=0.25,return_sequences=True, recurrent_dropout=0.1)))
    model.add(LeakyReLU())
    model.add(Bidirectional(LSTM(120,dropout=0.25,return_sequences=True, recurrent_dropout=0.1)))
    model.add(LeakyReLU())
    model.add(Bidirectional(LSTM(120,dropout=0.25, recurrent_dropout=0.1)))
    model.add(LeakyReLU())
    model.add(TimeDistributed(Dense(1)))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer="Adam",metrics=['mse', 'mae',])
    return model


def autoencod(input_dim=13):
    
    model = Sequential()
    model.add(Dense(7,input_dim=input_dim,activation="relu",activity_regularizer=l1(0.001)))
    model.add(Dense(4,input_dim=7,activation="relu",activity_regularizer=l1(0.001)))
    model.add(Dense(4,input_dim=4,activation="relu",activity_regularizer=l1(0.001)))
    model.add(Dense(7,input_dim=4,activation="relu",activity_regularizer=l1(0.001)))
    model.add(Dense(13,input_dim=7,activity_regularizer=l1(0.001)))

    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse', 'mae',])
    
    return model


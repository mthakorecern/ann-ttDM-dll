#!/usr/bin/env python

from __future__ import print_function

import time
start_time = time.time()
import json
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
from sklearn.utils import class_weight

from matplotlib import pyplot
#import ROOT
import numpy as np
import pandas as pd
import uproot as uprt
import seaborn as sns
import h5py
import csv

from my_functions import compare_train_test_binary
from my_functions import selection_criteria
from my_functions import plot_input_features
from my_functions import AUC_ROC

from sample_files import signal_paths, background_paths

from keras.callbacks import LambdaCallback

class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print(self.model.layers[0].get_weights())

  def on_train_batch_end(self, batch, logs=None):
    print(self.model.layers[0].get_weights())

def prepare_data():

    # Load data from Parquet files
    #samples = ['ttbarH125tobbbar_2L_allVars_2018.root', 'ttbarsignalplustau_fromDilepton_2018_20.9.21.root']
    #samples = [signal_paths["DMPseudo_top_tChan_Mchi1_Mphi100"],background_paths["TTToSemiLeptonic"]]
    
    Signal_df = {}
    Background_df={}
    Selection_s = {}
    Selection_b = {}
    Selection_s_temp = {}
    Selection_b_temp = {}

    Selection_inputs = ["nVetoElectrons", "nLooseMuons", "njets", "METcorrected_pt", "nbjets"]

    ML_inputs = ["M_Tb", "minDeltaPhi12", "deltaPhij1", "deltaPhij2", "deltaPhij3", "deltaPhib1", "MET_pt", "MET_phi", "njets", "nbjets", "nfjets"]

    signal_samples = ["DMScalar_top_tChan_Mchi1_Mphi100"]
    background_samples = ["TTToSemiLeptonic","TTTo2L2Nu"]

    print('Preparing data')
    for s in signal_samples:  # loop over samples
        print(s)
        file = uprt.open(signal_paths[s])
        tree = file["Events"]
        Signal_df[s] = tree.arrays(ML_inputs, library="pd")
        Selection_s[s] = tree.arrays(Selection_inputs, library="pd")
        Selection_s_temp[s] = np.vectorize(selection_criteria)(
            Selection_s[s].METcorrected_pt,
            Selection_s[s].nVetoElectrons,
            Selection_s[s].nLooseMuons,
            Selection_s[s].njets,
            Selection_s[s].nbjets)
        Signal_df[s] = Signal_df[s][Selection_s_temp[s]==True]
        Signal_df[s] = Signal_df[s].iloc[0:8500]
            #print(DataFrames[])
            #DataFrames[0].append(df[j])
    #print(Signal_df)

    for t in background_samples:  # loop over samples
        print(t)
        file = uprt.open(background_paths[t])
        tree = file["Events"]
        Background_df[t] = tree.arrays(ML_inputs, library="pd")
        Selection_b[t] = tree.arrays(Selection_inputs, library="pd")
        Selection_b_temp[t] = np.vectorize(selection_criteria)(
            Selection_b[t].METcorrected_pt,
            Selection_b[t].nVetoElectrons,
            Selection_b[t].nLooseMuons,
            Selection_b[t].njets,
            Selection_b[t].nbjets)
        Background_df[t] = Background_df[t][Selection_b_temp[t]==True]
        Background_df[t] = Background_df[t].iloc[0:4250]
    print(type(Background_df))
    


    '''
    file = uprt.open(background_paths["TTToSemiLeptonic"])
    tree = file["Events"]
    DataFrames[1] = tree.arrays(ML_inputs, library="pd")
    Selection[1] = tree.arrays(Selection_inputs, library="pd")

    selection_temp[1] = np.vectorize(selection_criteria)(
    Selection[1].METcorrected_pt,
    Selection[1].nVetoElectrons,
    Selection[1].nLooseMuons,
    Selection[1].njets,
    Selection[1].nbjets)

    DataFrames[1] = DataFrames[1][selection_temp[1]==True]
    #print(DataFrames[1])
    DataFrames[1] = DataFrames[1].iloc[0:8500]
    '''


    all_MC_s = []
    all_y_s = []
    all_MC_b = []
    all_y_b = []

    
    for u in signal_samples:  # loop over the different samples
        #print(s)
        all_MC_s.append(Signal_df[u][ML_inputs])  # append the MC dataframe to the list containing all MC features
        all_y_s.append(np.ones(Signal_df[u].shape[0]))  # signal events are labelled with 1

    for v in background_samples:  # loop over the different samples
        #print(s)
        all_MC_b.append(Background_df[v][ML_inputs])  # append the MC dataframe to the list containing all MC features
        all_y_b.append(np.full(Background_df[v].shape[0], 0))
    
    all_MC = all_MC_s + all_MC_b  # define empty list that will contain all features for the MC
    all_y = all_y_s + all_y_b  # define empty list that will contain labels whether an event in signal or background
    #print(all_y)
    X = np.concatenate(all_MC)  # concatenate the list of MC dataframes into a single 2D array of features, called X\
    #print("X is as follows:")
    #print(X)
    #print(X.size)

    y = np.concatenate(all_y)  # concatenate the list of labels into a single 1D array of labels, called y
    #print("y is as follows:")
    #print(y)
    #print(y.size)


    # make train and test sets
    print('Preparing train and test data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=492)  # set the random seed for reproducibility

    scaler = StandardScaler()  # initialise StandardScaler
    scaler.fit(X_train)  # Fit only to the training data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)

    for x, var in enumerate(ML_inputs):
        plot_input_features(X, y, x, var)

    # Save .csv with normalizations
    mean = scaler.mean_
    std = scaler.scale_
    with open('variable_norm.csv', mode='w') as norm_file:
        headerList = ['', 'mu', 'std']
        norm_writer = csv.DictWriter(norm_file, delimiter=',', fieldnames=headerList)
        norm_writer.writeheader()
        for x, var in enumerate(ML_inputs):
            print(var, mean[x], std[x])
            norm_writer.writerow({'': var, 'mu': mean[x], 'std': std[x]})

    X_valid_scaled, X_train_nn_scaled = X_train_scaled[:1000], X_train_scaled[1000:]  # first 1000 events for validation
    y_valid, y_train_nn = y_train[:1000], y_train[1000:]  # first 1000 events for validation

    print('Input feature correlation')
    print(Background_df['TTToSemiLeptonic'].corr())  # Pearson
    fig = pyplot.figure(figsize=(20, 16))
    corrMatrix = Background_df['TTToSemiLeptonic'].corr()
    ax = pyplot.gca()
    pyplot.text(0.5, 1.05, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=28)
    sns.heatmap(corrMatrix, annot=True, cmap=pyplot.cm.Blues)
    pyplot.savefig('TTToSemiLeptonic_correlation.png')

    print(Background_df['TTTo2L2Nu'].corr())  # Pearson
    fig = pyplot.figure(figsize=(20, 16))
    corrMatrix = Background_df['TTTo2L2Nu'].corr()
    ax = pyplot.gca()
    pyplot.text(0.5, 1.05, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=28)
    sns.heatmap(corrMatrix, annot=True, cmap=pyplot.cm.Blues)
    pyplot.savefig('TTTo2L2Nu_correlation.png')


    return X_train_nn_scaled, y_train_nn, X_test_scaled, y_test, X_valid_scaled, y_valid

def nn_model():

    # create model
    model = Sequential()
    model.add(Dense(249, input_dim=11,kernel_regularizer=regularizers.l1_l2(l1=1e-5,l2=7*1e-4)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.09))
    model.add(Dense(24,kernel_regularizer=regularizers.l1_l2(l1=1e-5,l2=7*1e-4)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.09))
    model.add(Dense(1, activation='sigmoid'))

    return model

def train_model(X_train, y_train, X_test, y_test, X_val, y_val):

    # fetch cnn model
    model = nn_model()

    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)

    # Compile model
    model.compile(loss='binary_crossentropy',optimizer='adagrad',metrics=['accuracy'])

    weight_print = MyCustomCallback()
    # Fit model
    history = model.fit(X_train, y_train, epochs=200, batch_size=1000, validation_data=(X_val, y_val), verbose=1)#,callbacks = [weight_print])

    # plot ROC
    decisions_tf = model.predict(X_test)
    fpr_tf, tpr_tf, thresholds_tf = roc_curve(y_test, decisions_tf)
    auc_ = auc(fpr_tf,tpr_tf)
    fauc_ = "{:.2f}".format(auc_)
    figRoc = pyplot.figure(figsize=(15, 15))
    ax = pyplot.gca()    
    pyplot.text(0.5, 1.05, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold", horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=28)
    pyplot.plot(tpr_tf, 1-fpr_tf, linestyle='--',linewidth=8,color='blue', label='ttHbb vs tt - AUC:'+str(fauc_))
    #pyplot.plot([1, 0], [1, 0], linestyle='dotted', color='grey', label='Luck') # plot diagonal line
    pyplot.xlabel('Signal efficiency',fontsize=28)
    pyplot.ylabel('Background Rejection',fontsize=28)
    pyplot.legend(loc='best',fontsize=28)
    pyplot.grid()
    pyplot.savefig('ROC.png')

    # plot train-test comparisons
    compare_train_test_binary(model,X_train,y_train,X_test,y_test,'tDM vs ttbar')

    # evaluate model
    loss, acc = model.evaluate(X_test, y_test, verbose=1,batch_size=1000)
    print('Test loss: {:.4f}'.format(loss))
    print('Test accuracy: {:.4f}'.format(acc))

    # confusion matrix
    y_pred = model.predict(X_test)
    Y_test = y_test.reshape(len(y_test),1)
    Y_pred = y_pred
    Y_pred[y_pred<0.5] = 0
    Y_pred[y_pred>0.5] = 1 
    mat = confusion_matrix(Y_test, Y_pred)
    classes = [0,1]
    con_mat_norm = np.around(mat.astype('float') / mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    # plot confusion matrix
    fig1 = pyplot.figure(figsize=(15, 15))
    ax = pyplot.gca()    
    pyplot.text(0.5, 1.05, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold", horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=28)
    sns.heatmap(con_mat_df, annot=True, cmap=pyplot.cm.Blues)
    #pyplot.tight_layout()
    
    pyplot.ylabel('True Class',fontsize=28)
    pyplot.xlabel('Predicted Class',fontsize=28)
    pyplot.savefig('confusion_matrix.png')

    # save trained model
    model.save('model.h5')

    return acc, history


def plot_model(history):
    # plot entropy loss
    pyplot.subplot(2, 1, 1)
    ax = pyplot.gca()    
    pyplot.text(0.5, 1.15, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold", horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=28)
    pyplot.title('Entropy Loss',fontsize=28)
    pyplot.plot(history[1].history['loss'], color='blue', label='train')
    pyplot.plot(history[1].history['val_loss'], color='red', label='test')

    # plot accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.title('Accuracy',fontsize=28)
    pyplot.plot(history[1].history['accuracy'], color='blue', label='train')
    pyplot.plot(history[1].history['val_accuracy'], color='red', label='test')
    pyplot.xlabel('Epoch',fontsize=20)
    pyplot.savefig('loss_accuraccy.png')


def main():

    # 1 load train dataset
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

    # 2 train model
    history = train_model(X_train, y_train, X_test, y_test, X_val, y_val)

    # 3 plot model
    plot_model(history)

    print(('\033[1m'+'> Time Elapsed = {:.3f} sec'+'\033[0m').format((time.time()-start_time)))

if __name__ == "__main__":
    main()

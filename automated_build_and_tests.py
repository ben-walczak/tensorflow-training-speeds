# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:15:59 2019

@author: Benjamin
"""

import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

import tensorflow as tf

def create_random_data(size_n = 500, n_labels = 2):
    df = pd.DataFrame(np.random.randint \
                      (0,100,size=(size_n, 4)), columns=list('ABCD'))
    df['E'] = np.random.randint(0,n_labels,size=(size_n, 1))
    return df

def create_train_test_splits(df, column_to_predict):
    train_dataset = df.sample(frac=0.8,random_state=0)
    test_dataset = df.drop(train_dataset.index)
    train_labels = train_dataset.pop(column_to_predict)
    test_labels = test_dataset.pop(column_to_predict)

    return train_dataset, test_dataset, train_labels, test_labels

def build_train_model(n_samples, epochs, n_labels, n_threads, \
                      iterations, column_to_predict, \
                      suppress_prints = True):

    df = create_random_data(n_samples, n_labels)
    train_dataset, test_dataset, train_labels, test_labels = \
                        create_train_test_splits(df,column_to_predict)
    train_dataset = train_dataset.astype(float)
    
    x = tf.placeholder(dtype = tf.float32, shape = [None, train_dataset.shape[1]])
    y = tf.placeholder(dtype = tf.int32, shape = [None])
    
    # set up loss functions, optimizer, and accuracy metrics
    logits = tf.contrib.layers.fully_connected(train_dataset, n_labels, tf.nn.relu)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits \
                              (labels = y, logits = logits))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001) \
                            .minimize(loss)
    correct_pred = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    train_time = {'time':[],'epochs':[],'thread count':[],
                  'label count':[], 'sample count':[]}
    
    for i in range(iterations):
        
        # sets up session
        tf.set_random_seed(1)
        sess = tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=n_threads))
        sess.run(tf.global_variables_initializer())
    
    
        # train and time model
        start = time.time()
        for i in range(epochs):
            if not suppress_prints: print('Epoch: ', i)
            _, accuracy_val = sess.run([train_op, accuracy], \
                    feed_dict={x: train_dataset, y: train_labels})
            if not suppress_prints: print('Accuracy: ', accuracy_val)
        end = time.time()
        total_time = end - start
        
        # record training times
        train_time['time'].append(total_time)
        train_time['epochs'].append(epochs)
        train_time['thread count'].append(n_threads)
        train_time['label count'].append(n_labels)
        train_time['sample count'].append(n_samples)
        
    return train_time

def train_model_per_thread(epochs, n_samples):
    n_labels = 2
    iterations = 20
    column_to_predict = 'E'
    thread_max = 4
    
    thread_d = {}
    
    for n_threads in range(1, thread_max+1):
        time_d = build_train_model(n_samples, epochs, n_labels, n_threads, \
                                  iterations, column_to_predict)
        cols = ['time', 'epochs', 'thread count', 'label count', 'sample count']
        time_df = pd.DataFrame(data = time_d, columns=cols)
        thread_d[str(n_threads)+' thread(s)'] = time_df
        
    thread_df = pd.DataFrame()
    
    for i in range(1, thread_max+1):
        thread_df = thread_df.append(thread_d[str(i)+' thread(s)']['time'], ignore_index=True)
    
    thread_df = thread_df.transpose()
    thread_df.columns = ['1', '2', '3', '4']
    
    thread_df.boxplot()
    plt.xlabel('Number of threads')
    plt.ylabel('Time to train model')
    plt.savefig('boxplots_of_'+str(epochs)+ \
            '_epochs_'+str(n_samples)+'_samples.png')
    
    thread_df.plot.kde()
    plt.xlabel('Time to train model')
    plt.title('Time Distribution of '+str(epochs)+ \
                ' Epochs and '+str(n_samples)+' Samples')
    plt.savefig('time_distribution_'+str(epochs)+ \
                '_epochs_'+str(n_samples)+'_samples.png')
    
    thread_df.to_csv('time_df_'+str(epochs)+ \
            '_epochs_'+str(n_samples)+'_samples.csv')
    
epochs_list = [50, 100]
n_samples_list = [100, 1000, 10000]

for epochs in epochs_list:
    for n_samples in n_samples_list:
        train_model_per_thread(epochs, n_samples)
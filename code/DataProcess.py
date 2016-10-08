#!/usr/bin/python
#coding=utf8

import numpy as np
import scipy as sy
import os, sys, logging
#import keras models
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Embedding, Input
from keras.preprocessing.text import Tokenizer

class ZeroProDetection:
    TESTPORTION = 0.05
    def __init__(self, dataFileName):
        # set the logger format
        logFormat = logging.Formatter("%(asctime)s :%(name)s :%(levelname)s :%(message)s" )        
        # set a stream handler to output to the terminal
        tLogHandle = logging.StreamHandler()
        tLogHandle.setFormatter(logFormat)
        tLogHandle.setLevel(logging.INFO)        
        # set a logger for the class to show the progress
        self.logger = logging.getLogger("ZeroProDetection")        
        self.logger.setLevel(logging.INFO)        
        self.logger.addHandler(tLogHandle)        
        
        self.dataFileName = dataFileName
        
    def textPreprocess(self):
        # Most codes of this function are learning from Francois
        self.logger.info('Fetch the dataset ...')
        textList = []
        labelList = []
        fpDataFile = open(self.dataFileName)
        for line in fpDataFile:
            splitLine = line.split('\t')
            textList.append(splitLine[0])
            labelList.append(splitLine[1])
        fpDataFile.close()
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(textList)
        self.word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(textList)
        # ramdon the sentence matrix        
        indices = np.arange(sequences.shape[0])
        np.random.shuffle(indices)
        sequences = sequences[indices]
        labelList = labelList[indices]
        # split the data into training set and test set
        nbTest = int(self.TESTPORTION * sequences.shape[0]) 
        
        self.trainData = sequences[ : -nbTest]
        self.trainLabels = labelList[ : -nbTest]
        self.testData = sequences[ -nbTest :]
        self.testLabels = labelList[ -nbTest : ]
        
    def loadWordVecs(self, vectFile, vectorDim):
        # Most codes of this function are learning from Francois
        embeddings_index = {}
        fpVecFile = open(vecFile)
        for line in fpVecFile:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        fpVecFile.close()
        embedding_matrix = np.zeros((len(self.word_index) + 1, vectorDim))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector        
        return embedding_matrix

    def networConstuct(self, wordEmbeddingDim, wordEmbeddingMatrix = []):
        sequencesInput = Input(shape=(wordEmbeddingDim)) 
        if len(wordEmbeddingMatrix) > 10:
            embeddingLayer = Embedding(len(self.word_index) + 1, wordEmbeddingDim, weights = [wordEmbeddingMatrix], trainable = False)
        else:
            embeddingLayer = Embedding(len(self.word_index) + 1, wordEmbeddingDim, activation = 'relu')



if __name__ == '__main__':
    #if len(sys.argv) < 2:
    #    exit(-1)
    dataFileName = './dataset/ProsessedText.txt'
    processor = ZeroProDetection(dataFileName)
    processor.textPreprocess()

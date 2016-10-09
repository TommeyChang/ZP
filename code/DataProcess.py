#!/usr/bin/python
#coding=utf8

import numpy as np
import scipy as sy
import os, sys, logging
#import keras models
from keras.models import Sequential, Model

from keras.layers import Embedding, Input, Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

class ZeroProDetection:
    TESTPORTION = 0.05
    MAX_SEQUENCE_LENGTH = 50
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
        # Read the file from the datafile
        textList = []
        rawLabelList = []
        fpDataFile = open(self.dataFileName)
        for line in fpDataFile:
            splitLine = line.split('\t')
            if len(splitLine[1][:-1]) > 0:                
                textList.append(splitLine[0])
                labels = map(int,splitLine[1][:-1].split(','))
                rawLabelList.append(labels)
        fpDataFile.close()
        
        # use Tokenizer to process the word
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(textList)
        self.word_index = tokenizer.word_index
        #
        rawSequences = tokenizer.texts_to_sequences(textList)
        sequences = pad_sequences(rawSequences, self.MAX_SEQUENCE_LENGTH)
        labelList = pad_sequences(rawLabelList, self.MAX_SEQUENCE_LENGTH)
       
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
        fpVecFile = open(vectFile)
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

    def modelExternalEmbedding(self, wordEmbeddingDim, wordEmbeddingMatrix):
        # Set a empty model
        mainInput = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        embeddingOutput = Embedding(len(self.word_index) + 1, wordEmbeddingDim, weights = [wordEmbeddingMatrix], trainable = False)(mainInput)
        bLSTMOutput = Bidirectional(LSTM(128, return_sequences = True))(embeddingOutput)
        finalOutput = LSTM(1, activation = 'hard_sigmoid', return_sequences = True)(bLSTMOutput)
        model = Model(mainInput, finalOutput)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        return model
    
    def modelTraining(self, model, epoches, batchSize):
        model.fit(self.trainData, self.trainLabels, validation_data = (self.testData, self.testLabels), nb_epoch = epoches)       

if __name__ == '__main__':
    #if len(sys.argv) < 2:
    #    exit(-1)
    dataFileName = './datasetsequence/ProsessedText.txt'
    dataFileName = './dataset/ProsessedText.txt'
    processor = ZeroProDetection(dataFileName)
    processor.textPreprocess()
    print processor.testData.shape
    #wordEmbeddingMatrix = processor.loadWordVecs('./datasetsequence/dataset.vec', 300)
    #processorModel = processor.modelExternalEmbedding(300, wordEmbeddingMatrix)
    #processor.modelTraining(processorModel, 100, 128)
    

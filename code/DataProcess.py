#!/usr/bin/python
#coding=utf8

import numpy as np
import scipy as sy
import os, sys, logging
#import keras models
from keras.models import Sequential, Model, model_from_json

from keras.layers import Embedding, Input, merge, TimeDistributed
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

class ZeroProDetection:
    TESTPORTION = 0.01
    MAX_SEQUENCE_LENGTH = 30
    PADDING_FLAG = 1
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
        totalLines = 0
        for line in fpDataFile:
            splitLine = line.split('\t')
            if len(splitLine[1][:-1]) > 0:                
                textList.append(splitLine[0])
                labels = map(int,splitLine[1][:-1].split(','))                                   
                rawLabelList.append(labels)
                totalLines += 1
        fpDataFile.close()
        self.logger.info('Finish fetching dataset, %d records.' % totalLines)
        
        
        # use Tokenizer to process the word
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(textList)
        self.word_index = tokenizer.word_index
        self.logger.info('There are %d words in the corpus.' % len(self.word_index))
        rawSequences = tokenizer.texts_to_sequences(textList)
        
        sequences = pad_sequences(rawSequences, self.MAX_SEQUENCE_LENGTH, padding = 'post')
        labelList = pad_sequences(rawLabelList, self.MAX_SEQUENCE_LENGTH, padding = 'post')
        
       
        # ramdon the sentence matrix
        
        indices = np.arange(sequences.shape[0])
        np.random.shuffle(indices)
        
        sequences = sequences[indices]
        labelList = labelList[indices]
        # split the data into training set and test set
        
        nbTest = int(self.TESTPORTION * sequences.shape[0])
        
        self.trainData = sequences[ : -nbTest]
        self.testData = sequences[ -nbTest :]
        
        #Tensorflow
        self.trainLabels = np.array(labelList[ : -nbTest]).reshape(labelList[ : -nbTest].shape[0], self.MAX_SEQUENCE_LENGTH, 1)            
        self.testLabels = np.array(labelList[ -nbTest : ]).reshape(labelList[ -nbTest : ].shape[0], self.MAX_SEQUENCE_LENGTH, 1)
        
            

    def loadWordVecs(self, vectFile, vectorDim):
        # Most codes of this function are learning from Francois
        self.logger.info('Sarting to load the word vectors...')
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
        self.logger.info('Finish loading the word vectors.')
        return embedding_matrix

    def modelExternalEmbedding(self, wordEmbeddingDim, wordEmbeddingMatrix, dimLSTM, dimDense):
        self.logger.info('Start to construct the model...')
        # Set a empty model
        mainInput = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        embeddingOutput = Embedding(len(self.word_index) + 1, wordEmbeddingDim, weights = [wordEmbeddingMatrix], trainable = False, mask_zero = True)(mainInput)
        forwardLSTM = LSTM(dimLSTM, return_sequences = True, consume_less = 'cpu', activation = 'tanh')(embeddingOutput)
        backwardLSTM = LSTM(dimLSTM, return_sequences = True, consume_less = 'cpu', activation = 'tanh')(embeddingOutput)
        mergeOutput = merge([forwardLSTM, backwardLSTM], mode = 'concat', concat_axis = -1)
        dpOutput = Dropout(0.05)(mergeOutput)
        denseOutput = TimeDistributed( Dense( dimDense, activation = 'tanh'))(dpOutput)
        finalOutput = TimeDistributed( Dense( 1, activation = 'sigmoid'))(denseOutput)
        model = Model(mainInput, finalOutput)
        model.compile( loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
        self.logger.info('Finish constructing the model')
        return model
    
    def modelTraining(self, model, epoches, batchSize, valSplit, modelConfigSavePath = './'):
        self.logger.info('Start training...')      
        model.fit(self.trainData, self.trainLabels, nb_epoch = epoches, batch_size = batchSize, validation_split = valSplit) 
        fpModelConfig = open(modelConfigSavePath + 'model.json', 'w')
        fpModelConfig.write(model.to_json())       

if __name__ == '__main__':
    #if len(sys.argv) < 2:
    #    exit(-1)
    dataFileName = './datasetclean/ProsessedText.txt'
    #dataFileName = './dataset/ProsessedText.txt'
    processor = ZeroProDetection(dataFileName)
    processor.textPreprocess()
    wordEmbeddingMatrix = processor.loadWordVecs('./datasetclean/dataset.vec', 300)
    processorModel = processor.modelExternalEmbedding(300, wordEmbeddingMatrix, 256, 512)
    processor.modelTraining(processorModel, 10, 32, 0.05)
    

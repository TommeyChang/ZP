#!/usr/bin/python
#coding=utf8

import numpy as np
import scipy as sy
import os, sys, logging
#import keras models
from keras.models import Model

from keras.layers import Embedding, Input, merge, TimeDistributed
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils 

from keras.utils.visualize_util import plot



class ZeroProDetection:
    TESTPORTION = 0.0004
    MAX_SEQUENCE_LENGTH = 35
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

    def modelConstruct(self, wordEmbeddingDim, dimLSTM, dimDense,  wordEmbeddingMatrix = []):
        self.logger.info('Start to construct the model...')
        # Set a empty model
        mainInput = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        if len(wordEmbeddingMatrix) > 0:
            embeddingOutput = Embedding(len(self.word_index) + 1, wordEmbeddingDim, weights = [wordEmbeddingMatrix], trainable = False, mask_zero = True)(mainInput)
        else:
            embeddingOutput = Embedding(len(self.word_index) + 1, wordEmbeddingDim, mask_zero = True)(mainInput)
        forwardLSTM = LSTM(dimLSTM, return_sequences = True, consume_less = 'cpu', activation = 'tanh')(embeddingOutput)
        backwardLSTM = LSTM(dimLSTM, return_sequences = True, consume_less = 'cpu', activation = 'tanh', go_backwards = True)(embeddingOutput)
        mergeOutput = merge([forwardLSTM, backwardLSTM], mode = 'concat', concat_axis = -1)
        dpOutput = Dropout(0.05)(mergeOutput)
        
        denseOutput = TimeDistributed( Dense ( dimDense, activation = 'tanh') )( dpOutput )
        denseSecOutput = TimeDistributed( Dense (dimDense, activation = 'tanh') ) (denseOutput)
        finalOutput = TimeDistributed( Dense ( 1, activation = 'sigmoid') )( denseSecOutput )
        model = Model(mainInput, finalOutput)
        model.compile( loss='binary_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
        plot(model, to_file = 'model.png') 
        self.logger.info('Finish constructing the model.')
        return model    
    
    
    def modelTraining(self, model, epoches, batchSize, valSplit):
        self.logger.info('Start training the model, batch size: %d, epoches: %d, validation split %f' % (batchSize, epoches, valSplit))      
        model.fit(self.trainData, self.trainLabels, nb_epoch = epoches, batch_size = batchSize, validation_split = valSplit) 
        #save the model
        try:
            os.mkdir('model')
        except OSError:
            pass
        fpModelConfig = open('./model/model.json', 'w')
        fpModelConfig.write(model.to_json())
        fpModelConfig.close
        #save the weights
        model.save_weights('./model/model.h5')
        self.logger.info('Finish training the model, model files saved in ./model/model.json, model weights saved in ./model/model.h5.')
        return model

if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit(-1)
    
    dataFileName = sys.argv[1] + 'ProcessedText.txt'
   
    inputDim = int(sys.argv[2])
    lstmDim = int(sys.argv[3])
    denseDim = int(sys.argv[4])
    epochNum = int(sys.argv[5])
    batchSize = int(sys.argv[6])
    valSplit = float(sys.argv[7])
    
    processor = ZeroProDetection(dataFileName)
    processor.textPreprocess()
    if sys.argv[8] == 'e':
        vecFileName = sys.argv[1] + 'dataset.vec'
        wordEmbeddingMatrix = processor.loadWordVecs(vecFileName, inputDim)
        untrainModel = processor.modelConstruct(inputDim, lstmDim, denseDim, wordEmbeddingMatrix)
    else:
        untrainModel = processor.modelConstruct(inputDim, lstmDim, denseDim)
    trainedModel = processor.modelTraining(untrainModel, epochNum, batchSize, valSplit) 
    
    predictLabels = trainedModel.predict(processor.testData, batch_size = 32)
    for predictLabel, realLabel in zip(predictLabels, processor.testLabels):
        differeLabel = np.zeros(predictLabel.shape)
        differePos = ( predictLabel != realLabel )
        differeLabel[differePos] = 1
        
        print realLabel.reshape(1,processor.MAX_SEQUENCE_LENGTH)
        print predictLabel.reshape(1,processor.MAX_SEQUENCE_LENGTH)
        print differeLabel.reshape(1,processor.MAX_SEQUENCE_LENGTH)
   
   
   
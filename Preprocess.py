from keras.models import Sequential
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer

class ZeroPro:
    def __init__(self):
        pass

    def dataPreprocess(self, dataFileName):
        fpDataset = open(dataFileName)
        text = []
        lbaels = []
        for dataLine in fpDataset:
            dataLineSplit = dataLine.split()
            text.append(dataLineSplt[0])
            layers.append(dataLineSplit[1])
        fpDataset.closed()
        tokenizer = Tokenizera(filters='')
        tokenizer.fit_on_texts(text)
        sequence = tokenizer.texts_to_sequences(text)



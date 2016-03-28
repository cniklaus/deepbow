from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation
from numpy import array as arr
import random

def parseFields(line):
    return line.rstrip().lower().split("\t")

def parseline(line, labels):
    fields = parseFields(line)
    
    assert len(fields) == len(labels), "length of labels ({0}) and fields ({1}) differ!".format(str(labels), str(fields))
    
    return { l: (f.split(" ") if " " in f else f) for l, f in zip(labels, fields) }

def loaddata(filename):
    with open(filename, 'r') as fh:
        result = list()
        
        labels = parseFields(fh.readline())
        
        for line in fh:
            linedata = parseline(line, labels)
            
            result.append(linedata)
        
        return result
	  
def getVocabulary(data, key):
    return sorted({word for sample in data for word in sample[key]})

def getVocs(data):
    origVoc = getVocabulary(data, 'original')
    simpVoc = getVocabulary(data, 'simplified')
    
    return (origVoc, simpVoc)

def getBowVec(sample, vocabulary):
    return [(1 if word in sample else 0) for word in vocabulary]

def sampleInFeats(sample, (origVoc, simpVoc)):
    return getBowVec(sample['original'], origVoc) + getBowVec(sample['simplified'], simpVoc)

def sampleOutFeats(sample, keys=['g', 'm', 's', 'overall']):
    return [val for key in keys for val in getBowVec(sample[key], ['bad', 'ok', 'good'])]

def datasetInFeats(data, vocs):
    return arr([sampleInFeats(sample, vocs) for sample in data])

def datasetOutFeats(data, keys=['g', 'm', 's', 'overall']):
    return arr([sampleOutFeats(sample, keys) for sample in data])

def datasetFeats(data, vocs, keys=['g', 'm', 's', 'overall']):
    return (datasetInFeats(data, vocs), datasetOutFeats(data, keys))

def prepareModel(inSize, hidSize, outSize):
    model = Sequential()
    
    model.add(Dense(output_dim=hidSize, input_dim=inSize, init="glorot_uniform", activation="sigmoid"))
    model.add(Dense(output_dim=outSize, init="glorot_uniform", activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    
    return model

def trainModel((trainX, trainY), hiddenSize = 3, devFeats = False):
    model = prepareModel(len(trainX[0]), hiddenSize, len(trainY[0]))
    
    model.fit(trainX, trainY, nb_epoch=1024, batch_size=400, show_accuracy=True, validation_split=0.1, verbose=1)
    
    return model

def testModel(model, (testX, testY)):
    return model.evaluate(testX, testY)

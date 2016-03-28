import sys
from lib import *

data = loaddata(sys.argv[1])

vs = getVocs(trainSet)
trainFeats = datasetFeats(trainSet, vs)

model = trainModel(trainFeats)

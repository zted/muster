"""
This file contains some useful functions for reading/writing to file
and some folder manipulations
"""

import os
from nltk.corpus import wordnet as wn

def writeToFile(aFile,metricString,offsetID,word,numString):
    aFile.write('-{0}- -{1}- -{2}- {3}\n'.format(metricString, str(offsetID), word, numString))
    return

def removeAllSubfiles(someFile):
    allFiles = os.listdir(someFile)
    for f in allFiles:
        os.remove(someFile+'/'+f)
    return

def getUniqueOffsetIDs(inFile):
    '''
    :param inFile:file that contains words we want offset IDs for
    :return: a set containing integers which are offset IDs
    '''
    uniqueNums = set([])
    with open(inFile) as f:
        for line in f:
            w = line.split("\n")[0]
            synsets = wn.synsets(w)
            s = synsets[0].offset()
            # we take the most relevant synset for now
            uniqueNums.add(s.offset())
    return uniqueNums

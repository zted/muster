"""
This file contains some useful functions for reading/writing to file
and some folder manipulations
"""

import os

def writeToFile(aFile,metricString,offsetID,word,numString):
    aFile.write('-{0}- -{1}- -{2}- {3}\n'.format(metricString, str(offsetID), word, numString))
    return

def removeAllSubfiles(someFile):
    allFiles = os.listdir(someFile)
    for f in allFiles:
        os.remove(someFile+'/'+f)
    return
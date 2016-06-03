"""
This file contains some useful functions for reading/writing to file
and some folder manipulations
"""

import os

from nltk.corpus import wordnet as wn


def writeToFile(aFile, metricString, offsetID, word, numString):
    aFile.write('-{0}- -{1}- -{2}- {3}\n'.format(metricString, str(offsetID), word, numString))
    return


def removeAllSubfiles(someDir):
    # removes all subfiles from a directory
    allFiles = os.listdir(someDir)
    for f in allFiles:
        os.remove(someDir + '/' + f)
    return


def wordsToOffsetID(inFile):
    """
        :param inFile:file that contains words we want offset IDs for
        :return: a set containing integers which are offset IDs
        """
    uniqueNums = set([])
    f = open(inFile, 'r')
    for line in f:
        w = line.split("\n")[0]
        synsets = wn.synsets(w)
        s = synsets[0].offset()
        # we take the most relevant synset for now
        uniqueNums.add(s)
    return uniqueNums


def getOffsetIDs(inFile):
    """
        :param inFile: file containing offset IDs we want
        :return: a set containing integers which are offset IDs
        """
    uniqueNums = set([])
    f = open(inFile, 'r')
    for line in f:
        n = line.split("\n")[0]
        uniqueNums.add(int(n))
    return uniqueNums

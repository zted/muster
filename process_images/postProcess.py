
import os
import numpy as np
from nltk.corpus import wordnet as wn

WORDSFILE = '/home/tedz/Desktop/research/code/data/words_to_get.txt'
RESULTSDIR = '/home/tedz/Desktop/results_1stRun/results'

def resultsToCSV(inDirectory,outFile):
    '''
    goes through the resultsFiles, extracts the vectors associated
    with the mean, and writes it to a csv file that can then be
    processed by matlab
    :param inDirectory: contains result files with synsets
    and their feature representations
    :return: set of synsets
    '''
    oFile = open(outFile,'w')
    allFiles = os.listdir(inDirectory)
    for f in allFiles:
        fullPath = inDirectory + '/' + f
        ff = open(fullPath,'r')
        for line in ff:
            myStrs = line[0:30].split('-')[0:4]
            if myStrs[1] != 'mean':
                continue
            myVec = line.split('[')[1]
            myVec = myVec.rstrip(']\n')
            oFile.write(myVec+'\n')
    oFile.close()
    return

VECS_LONG = '/home/tedz/Desktop/results_1stRun/onlyMean.csv'
resultsToCSV(RESULTSDIR, VECS_LONG)

# at this point you should find the eigenvectors if using linear PCA
# dimension, put it into a file and set file path to the variable below
EIGENVECTORFILE = '/home/tedz/eigVecs.txt'


def reduceDimensions(inDirectory, outFile, eigenVectors):
    '''
    goes through the resultsFiles, extracts the vectors associated
    with the mean, reduces the dimensions according to the size of
    eigenVectors, and writes the reduced vector corresponding with
    its offset ID to a new file
    :param inDirectory: directory containing result files
    :param outFile: file to write the new reduced vectors to
    :param eigenVectors: needs to be type numpy.matrix
    :return:
    '''
    oFile = open(outFile,'w')
    allFiles = os.listdir(inDirectory)
    for f in allFiles:
        fullPath = inDirectory + '/' + f
        ff = open(fullPath,'r')
        for line in ff:
            myStrs = line[0:30].split('-')[0:4]
            if myStrs[1] != 'mean':
                continue
            myVec = line.split('[')[1]
            myVec = myVec.rstrip(']\n')
            offset = myStrs[3]
            vecArray = np.array(map(float,myVec.split(',')))
            reducedArray = vecArray * eigenVectors
            tempStr = '={0}= {1}\n'.format(offset,str(reducedArray.A1.tolist()))
            # ^A1 converts the 1 x n matrix to an array
            oFile.write(tempStr)
    oFile.close()
    return

f = open (EIGENVECTORFILE, 'r')
eigVecs = np.matrix([ map(float,line.split(',')) for line in f ])
# ^loads matrix in file into an numpy matrix
f.close()

REDUCEDVEC_ID = '/home/tedz/Desktop/rv.txt'
reduceDimensions(RESULTSDIR, REDUCEDVEC_ID, eigVecs)


def uniqueOffsetsFromResults(inDirectory):
    '''
    finds all the unique offset IDs from a directory with
    result files in them
    :param inDirectory: contains result files with synsets
    and their feature representations
    :return: set of synsets
    '''
    allFiles = os.listdir(inDirectory)
    uniqueNums = set([])
    for f in allFiles:
        fullPath = inDirectory + '/' + f
        ff = open(fullPath,'r')
        for line in ff:
            myStrs = line[0:30].split('-')[0:4]
            if myStrs[1] != 'mean':
                continue
            uniqueNums.add(int(myStrs[3]))
    return uniqueNums


def getWords(inFile):
    '''
    :param inFile: file containing all the words
    :return: list containing all the words
    '''
    uniqueWords = []
    f = open(inFile,'r')
    for line in f:
        w = line.split("\n")[0]
        uniqueWords.append(w)
    return uniqueWords


def wordIDPairs(allWords,allOffsets):
    '''
    :param allWords: words we care about
    :param allOffsets: offsets that we've processed
    :return: list of tuples, contains (word, offset),
    were offset is the offsetID of the highest synset
    for a given word that we have a representation of
    '''
    tuplesList = []
    for w in allWords:
        synsets = wn.synsets(w)
        for s in synsets:
            if s.offset() in allOffsets:
                tempTuple = (w,s.offset())
                tuplesList.append(tempTuple)
                break
    return tuplesList

offsets = uniqueOffsetsFromResults(RESULTSDIR)
words = getWords(WORDSFILE)
tupList = wordIDPairs(words,offsets)
# with this list of tuples, we should be able to
# match the words with their respective synsets

def generateWordVecPairs(reducedVecFile,outFile,tupleList):
    '''
    :param reducedVecFile: file with vectors of reduced dimensions
    corresponding to an offset ID
    :param outFile:
    :param tupleList: tuples with word, Offset ID
    :return:
    '''
    tpCopy = tupleList[:]
    # make a copy instead of modifying reference
    inFile = open(reducedVecFile,'r')
    out = open(outFile,'w')
    for line in inFile:
        myStrs = line.split('=')
        offID = int(myStrs[1])
        removeIndices = []
        for i in range(len(tpCopy)):
            if tpCopy[i][1] == offID:
                removeIndices.append(i)
                word = tpCopy[i][0]
                outStr = '={0}= {1}'.format(word,myStrs[-1])
                out.write(outStr)
        for ind in removeIndices:
            del tpCopy[ind]
    inFile.close()
    out.close()
    return

REDUCEDVEC_WORDS = '/home/tedz/Desktop/rv_words.txt'
generateWordVecPairs(REDUCEDVEC_ID,REDUCEDVEC_WORDS,tupList)
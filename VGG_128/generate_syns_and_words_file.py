# -*- coding: utf-8 -*-
"""
Created on Sun May 15 18:17:38 2016

@author: Guillem
"""


###########################################
# converting words_to_get into synset Ids #
###########################################

from nltk.corpus import wordnet as wn


def numToSynsetId(number):
    num = str(number)
    if 8-len(num) > 0:
        zeros = str(0)*(8 - len(num))
        synId = 'n'+ zeros + num
    else: 
        synId = 'n'+  num         
    return synId


def wordsToOffsetID(inFile):
    '''
    :param inFile:file that contains words we want offset IDs for
    :return: a set containing integers which are offset IDs
    '''
    #uniqueNums = set([])
    #uniqueWords = set([])
    uniqueNums = []
    uniqueWords = []
    f = open(inFile,'r')
    #new = open("C:/Guillem(work)/KU_Leuven/Code/VGG_128/synsets_to_get.txt", "w")
    for line in f:
        w = line.split("\n")[0]
        synsets = wn.synsets(w)
        s = synsets[0].offset()
        s1 = numToSynsetId(s)
        # we take the most relevant synset for now
        #uniqueNums.add(s)
        uniqueNums.append(s1)
        uniqueWords.append(w)  
        #uniqueNums.add(s1)
        #uniqueWords.add(w)  
    return (uniqueWords, uniqueNums)


fileOpen= 'C:/Guillem(work)/KU_Leuven/Code/VGG_128/words_to_get.txt'
words, synsets = wordsToOffsetID(fileOpen)


#now write the file with both, words and synset_id (This is manually building a CSV)
text_file = open('C:/Guillem(work)/KU_Leuven/Code/VGG_128/syns_and_words.csv', "w")
#text_file = open('C:/Guillem(work)/KU_Leuven/Code/VGG_128/syns_and_words.txt', "w")
for i in range(len(words)):
    write_line = synsets[i] + "," + words[i] + "\n"
    text_file.write(write_line)
text_file.close()


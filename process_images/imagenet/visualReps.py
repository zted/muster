"""
The purpose of this file is to generate visual representations for concept words into an output file
A trained Alexnet is used to evaluate the images, and several measures are calculated for each concept
word such as entropy, dispersion, mean, standard deviation.
"""

import os
import sys
import caffe
import numpy as np
import time
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import logging

## Setting some path variables:
HOMEDIR = os.environ['HOME']
CAFFE_ROOT = HOMEDIR + '/caffe'
OUTPUT_DIRECTORY = HOMEDIR
sys.path.extend([HOMEDIR + '/packages'])

from nltk.corpus import wordnet as wn
from progressbar import Bar,ETA,FileTransferSpeed,Percentage,ProgressBar

PROCESSING_DIRECTORY = sys.argv[1]
# check processing directory is a indeed a directory
dirFlag = os.path.isdir(PROCESSING_DIRECTORY)
if not dirFlag:
    raise SystemError('Entered: {0} is not a valid directory to process'.format(PROCESSING_DIRECTORY))
# ^directory needs to contain one folder per synset, like n0123456
# and in each folder a batch of pictures associated with the synset

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = CAFFE_ROOT + '/models/bvlc_alexnet/deploy.prototxt'
PRETRAINED = CAFFE_ROOT + '/models/bvlc_alexnet/bvlc_alexnet.caffemodel'

# Prepare the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(OUTPUT_DIRECTORY+'/Log_toyset.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s --- %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info("LETS GET STARTED")

senseIdToSynset = {s.offset(): s for s in wn.all_synsets()}

caffe.set_mode_cpu()
# caffe.set_mode_gpu()

# chop off the last layer of alexnet, we don't actually need the classification
extraction_layer = 'fc7'

# load alexnet's NN model
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(CAFFE_ROOT + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# create a dictionary that maps synset offset IDs to synset objects
senseIdToSynset = {s.offset(): s for s in wn.all_synsets()}

def processOneClass(thisDir,minPics = 4, maxPics = 1000):
    """
    Processes all images in one directory
    :param thisDir: directory where all the images of a class are stored
    :return: visual representations vectors for each image in a list
    """
    allVecs = []
    allImgs = os.listdir(thisDir)
    numImgs = len(allImgs)
    count = 0
    if numImgs < minPics:
        # not enough images to get a representation
        return allVecs

    for imgName in os.listdir(thisDir):
        if count > maxPics:
            # we've seen enough images from this class
            break
        count += 1
        imgPath = thisDir + '/' + imgName
        img = caffe.io.load_image(imgPath)
        net.predict([img])
        feature_vec = net.blobs[extraction_layer].data[0].copy()
        # the copy() is needed, otherwise feature_vec will store a pointer
        allVecs.append(feature_vec[:])
    return allVecs

def entropy(x, k=3, base=2):
    """
    Adapted from Greg Ver Steeg's NPEET toolkit - more info http://www.isi.edu/~gregv/npeet.html
    The classic K-L k-nearest neighbor continuous entropy estimator
    :param x: a list of numbers, e.g. x = [1.3,3.7,5.1,2.4]
    :param k: lower bound on how many elements must be in x
    :param base: base to work in
    :return:
    """
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    x = [[elem] for elem in x]
    d = len(x[0])
    N = len(x)
    intens = 1e-10 #small noise to break degeneracy, see doc.
    x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
    tree = ss.cKDTree(x)
    nn = [tree.query(point, k+1, p=float('inf'))[0][k] for point in x]
    const = digamma(N)-digamma(k) + d*log(2)
    return (const + d*np.mean(map(log,nn)))/log(base)

def computationsPerDimension(vecs):
    dimensions = len(vecs[0])
    mean = np.array([0.0] * dimensions)
    std = mean.copy()
    ent = mean.copy()
    mp = mean.copy()
    # ^mp stands for maxpool, we take the maximums of each vector
    for i in range(dimensions):
        allNums = np.array([j[i] for j in vecs])
        mean[i] = allNums.mean()
        std[i] = allNums.std()
        ent[i] = entropy(allNums)
        mp[i] = allNums.max()
    return mean, std, ent, mp

def calculateDispersion(vecs):
    """
    :param vecs: list of vector representations of a concept word
    :return: the dispersion of said concept word. a scalar
    """
    numVecs = len(vecs)
    accum = 0.0
    for i in range(numVecs-1):
        for j in range(i+1,numVecs):
            vi = vecs[i]; vj = vecs[j]
            dp = np.dot(vi,vj)
            denom = np.linalg.norm(vi) * np.linalg.norm(vj)
            accum += (1-dp/denom)
    dispersion = accum/(2.0*numVecs*(numVecs-1))
    return dispersion

def writeToFile(aFile,metricString,offsetID,word,numString):
    aFile.write('-{0}- -{1}- -{2}- {3}\n'.format(metricString, str(offsetID), word, numString))

classes_per_File = 4
outFilePrefix = '/toyset_results_'
fileCount = 1
outFile_Name = outFilePrefix + str(fileCount) + '.txt'
OUTFILE = open(OUTPUT_DIRECTORY + outFile_Name, 'w')
directories = os.listdir(PROCESSING_DIRECTORY)
numClasses = float(len(directories))
classCount = 0

# progress bar setup
widgets = ['Test: ', Percentage(), ' ',
           Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets, maxval=int(numClasses))
pbar.start()

for dir in directories:
    pbar.update(classCount)
    classCount += 1
    if classCount % classes_per_File == 0:
        # we've stored enough in one file, open new one
        OUTFILE.close()
        fileCount += 1
        outFile_Name = outFilePrefix + str(fileCount) + '.txt'
        OUTFILE = open(OUTPUT_DIRECTORY + outFile_Name, 'w')

    try:
        offID = int(dir[1:])
        # chops off the 0 in front, for example ID 00123 becomes 123
        thisSet = senseIdToSynset[offID]
        logger.info("Processing synset " + str(offID))
    except:
        print "Cannot find the synset for offset ID " + str(offID)
        logger.error("Cannot find the synset for offset ID " + str(offID))
        continue

    t0 = time.time()
    try:
        vecs = processOneClass(PROCESSING_DIRECTORY + '/' + dir)
        t_elapsed = time.time() - t0
        numImgs = len(vecs)
        logger.info('{0} images took {1} seconds to process: '.format(str(numImgs),str(t_elapsed)))
    except:
        logger.error("Unexpected error processing images in " + dir)
        continue

    t0 = time.time()
    try:
        mean, std, ent, mp = computationsPerDimension(vecs)
        disp = calculateDispersion(vecs)
        t_elapsed = time.time() - t0
        logger.info("Vector computations took {0} seconds to process: ".format(str(t_elapsed)))
    except:
        logger.error("Unexpected error performing computations for " + dir)
        continue

    for lem in thisSet.lemmas():
        word = str(lem.name())
        writeToFile(OUTFILE,'mean',offID,word,str(mean.tolist()))
        writeToFile(OUTFILE,'maxpool',offID,word,str(mp.tolist()))
        writeToFile(OUTFILE,'std',offID,word,str(std.tolist()))
        writeToFile(OUTFILE,'entropy',offID,word,str(ent.tolist()))
        writeToFile(OUTFILE,'dispersion',offID,word,str(disp))

OUTFILE.close()
pbar.finish()
logger.info("Finished")


#TODO: add more options for arguments such as GPU vs CPU
#TODO: add functionality to extract from tar files
#TODO: another entropy measure
#TODO: put vector calculations in another file
#TODO: normalize values?
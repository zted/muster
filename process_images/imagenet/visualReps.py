"""
The purpose of this file is to generate visual representations for concept words into an output file
A trained Alexnet is used to evaluate the images, and several measures are calculated for each concept
word such as entropy, dispersion, mean, standard deviation.

*Execution instructions
Execute this file by calling it like this - python ../.../visualReps.py ../directory_containing_imgClasses

***WARNING - IMPORTANT***
Execute this file from the top level directory. If file structure is
.../muster/process_images/imagenet/visualReps, be in the .../muster
directory when executing this file (from command line)
"""

import logging
import os
import sys
import tarfile as trf
import time
import caffe
import numpy as np

## Setting some path variables:
HOMEDIR = os.environ['HOME']
CAFFE_ROOT = HOMEDIR + '/caffe'
OUTPUT_DIRECTORY = HOMEDIR + '/results'
sys.path.extend([HOMEDIR + '/packages'])
CWD = os.getcwd()
sys.path.extend([CWD+'/utils'])

import IOtools as IOT
import mathComputations as MC
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
handler = logging.FileHandler(OUTPUT_DIRECTORY+'/Log_EntireSet.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s --- %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info("LETS GET STARTED")

senseIdToSynset = {s.offset(): s for s in wn.all_synsets()}

# caffe.set_mode_cpu()
caffe.set_mode_gpu()

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

def processOneClass(thisDir,minPics = 50, maxPics = 500):
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
        raise ValueError("Not enough images to build feature representation")

    for imgName in os.listdir(thisDir):
        if count > maxPics:
            # we've seen enough images from this class
            break
        count += 1
        imgPath = thisDir + '/' + imgName
        try:
            img = caffe.io.load_image(imgPath)
            net.predict([img])
            feature_vec = net.blobs[extraction_layer].data[0].copy()
            # the copy() is needed, otherwise feature_vec will store a pointer
            allVecs.append(feature_vec[:])
        except IOError as e:
            logger.warn('I/O error({0}) in file {1}: {2}'.format(e.errno, imgName, e.strerror))
            continue
    if len(allVecs) < minPics:
        raise ValueError("Not enough images to build feature representation")
    return allVecs


classes_per_File = 20  # limiting size of results file
fileCount = 1  # used for numbering result files
classCount = 0  # used to track progress
classesProcessed = 0  # determines when to close results files and open new ones

outFilePrefix = '/Visual_Representations_'
outFile_Name = outFilePrefix + str(fileCount) + '.txt'
OUTFILE = open(OUTPUT_DIRECTORY + outFile_Name, 'w')

directories = os.listdir(PROCESSING_DIRECTORY)
numClasses = float(len(directories))

# progress bar setup
widgets = ['Test: ', Percentage(), ' ',
           Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets, maxval=int(numClasses))
pbar.start()

words_file = CWD + '/data/words_to_get.txt'
synsets_file = CWD + '/data/synsets_to_get.txt'
uniqueIDs = IOT.getOffsetIDs(synsets_file)
tempFolder = HOMEDIR + '/tmpProcessing'
os.mkdir(tempFolder)

for dir in directories:

    pbar.update(classCount)
    classCount += 1

    try:
        offID = int(dir[1:].strip('.tar'))
        # chops off the 0's in front, for example ID 00123 becomes 123
        dummyFlag = True
        # dummyFlag = offID in uniqueIDs
        # will add an option later on to incorporate whether we want to
        # process all synsets, or only certain ones
        if dummyFlag:
            thisSet = senseIdToSynset[offID]
            logger.info("Processing synset " + str(offID))
        else:
            logger.info(str(offID) + " not needed thus not processed")
            continue
    except:
        print "Cannot find the synset for offset ID in " + dir
        e = sys.exc_info()[1]
        logger.error(e)
        logger.error("Cannot find the synset for offset ID in " + dir)
        continue

    tempTar = trf.open('{0}/{1}'.format(PROCESSING_DIRECTORY,dir),'r:')
    tempTar.extractall(tempFolder)
    tempTar.close()
    procFolder = tempFolder
    # redundant now, but if we are working with non-tar files in the future,
    # then procFolder would not need to be tempFolder

    t0 = time.time()
    try:
        vecs = processOneClass(procFolder)
        t_elapsed = time.time() - t0
        numImgs = len(vecs)
        IOT.removeAllSubfiles(procFolder)
        logger.info('{0} images took {1} seconds to process'.format(str(numImgs),str(t_elapsed)))
    except ValueError as e:
        logger.error('ValueError for {0}: {1}'.format(dir, e))
        continue

    t0 = time.time()
    try:
        mean, std, ent, mp = MC.computationsPerDimension(vecs)
        disp = MC.calculateDispersion(vecs)
        meanEnt = MC.meanEntropy(mean)
        t_elapsed = time.time() - t0
        logger.info("Vector computations took {0} seconds to process".format(str(t_elapsed)))
    except:
        e = sys.exc_info()[1]
        logger.error(e)
        logger.error("Unexpected error performing computations for " + dir)
        continue

    word = str(thisSet.lemmas()[0].name())
    IOT.writeToFile(OUTFILE, 'mean', offID, word, str(mean.tolist()))
    IOT.writeToFile(OUTFILE, 'maxpool', offID, word, str(mp.tolist()))
    IOT.writeToFile(OUTFILE, 'std', offID, word, str(std.tolist()))
    IOT.writeToFile(OUTFILE, 'entropy', offID, word, str(ent.tolist()))
    IOT.writeToFile(OUTFILE, 'dispersion', offID, word, str(disp))
    IOT.writeToFile(OUTFILE, 'meanEntropy', offID, word, str(meanEnt))
    canOpenNew = True
    classesProcessed += 1
    logger.info('Completed processing synset ' + str(offID))

    if classesProcessed % classes_per_File == 0:
        # we've stored enough in one file, open new one
        canOpenNew = False
        OUTFILE.close()
        fileCount += 1
        outFile_Name = outFilePrefix + str(fileCount) + '.txt'
        OUTFILE = open(OUTPUT_DIRECTORY + outFile_Name, 'w')

IOT.removeAllSubfiles(procFolder)
os.rmdir(procFolder)
OUTFILE.close()
pbar.finish()
logger.info("Finished")

#TODO: add more options for arguments such as GPU vs CPU
#TODO: normalize values?
#TODO: add functionality to specify output directory

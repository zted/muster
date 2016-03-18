"""
The purpose of this file is to generate visual representations for concept words into an output file
A trained Alexnet is used to evaluate the images, and several measures are calculated for each concept
word such as entropy, dispersion, mean, standard deviation.

***INSTRUCTIONS - IMPORTANT***
Execute this file from the top level directory. If file structure is
muster/process_images/imagenet/visualReps, be in the muster directory
when executing this file (from command line)
"""

import logging
import os
import sys
import tarfile as trf
import time
import caffe

## Setting some path variables:
HOMEDIR = os.environ['HOME']
CAFFE_ROOT = HOMEDIR + '/caffe'
OUTPUT_DIRECTORY = HOMEDIR
sys.path.extend([HOMEDIR + '/packages'])
CWD = os.getcwd()
sys.path.extend([CWD+'/utils'])

from IOtools import *
from mathComputations import *
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

def processOneClass(thisDir,minPics = 4, maxPics = 500):
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
        img = caffe.io.load_image(imgPath)
        net.predict([img])
        feature_vec = net.blobs[extraction_layer].data[0].copy()
        # the copy() is needed, otherwise feature_vec will store a pointer
        allVecs.append(feature_vec[:])
    return allVecs


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

tempFolder = PROCESSING_DIRECTORY + '/tmpProcessing'
os.mkdir(tempFolder)

for dir in directories:
    pbar.update(classCount)

    if dir == 'fall11_whole.tar':
        # temporary workaround, because the zip file for the
        # entire imagenet is in the same folder as all the
        # image classes and it'll take too long to move
        continue

    classCount += 1

    if classCount % classes_per_File == 0:
        # we've stored enough in one file, open new one
        OUTFILE.close()
        fileCount += 1
        outFile_Name = outFilePrefix + str(fileCount) + '.txt'
        OUTFILE = open(OUTPUT_DIRECTORY + outFile_Name, 'w')

    try:
        offID = int(dir[1:].strip('.tar'))
        # chops off the 0 in front, for example ID 00123 becomes 123
        thisSet = senseIdToSynset[offID]
        logger.info("Processing synset " + str(offID))
    except:
        print "Cannot find the synset for offset ID in " + dir
        e = sys.exc_info()[0]
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
        removeAllSubfiles(procFolder)
        logger.info('{0} images took {1} seconds to process: '.format(str(numImgs),str(t_elapsed)))
    except:
        e = sys.exc_info()[0]
        logger.error(e)
        logger.error("Unexpected error processing images in " + dir)
        continue

    t0 = time.time()
    try:
        mean, std, ent, mp = computationsPerDimension(vecs)
        disp = calculateDispersion(vecs)
        meanEnt = meanEntropy(mean)
        t_elapsed = time.time() - t0
        logger.info("Vector computations took {0} seconds to process: ".format(str(t_elapsed)))
    except:
        e = sys.exc_info()[0]
        logger.error(e)
        logger.error("Unexpected error performing computations for " + dir)
        continue

    word = str(thisSet.lemmas()[0].name())
    writeToFile(OUTFILE,'mean',offID,word,str(mean.tolist()))
    writeToFile(OUTFILE,'maxpool',offID,word,str(mp.tolist()))
    writeToFile(OUTFILE,'std',offID,word,str(std.tolist()))
    writeToFile(OUTFILE,'entropy',offID,word,str(ent.tolist()))
    writeToFile(OUTFILE,'dispersion',offID,word,str(disp))
    writeToFile(OUTFILE,'meanEntropy',offID,word,str(meanEnt))

removeAllSubfiles(procFolder)
os.rmdir(procFolder)
OUTFILE.close()
pbar.finish()
logger.info("Finished")

#TODO: add more options for arguments such as GPU vs CPU
#TODO: another entropy measure
#TODO: normalize values?
#TODO: add functionality to specify output directory
#TODO: only process synsets that we specify
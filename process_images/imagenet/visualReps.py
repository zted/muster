"""
The purpose of this file is to generate visual representations for concept words into an output file
A trained Alexnet is used to evaluate the images, and several measures are calculated for each concept
word such as entropy, dispersion, mean, standard deviation.
"""

import numpy as np
import caffe
import os
from nltk.corpus import wordnet as wn
import sys
import time
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import logging

## Setting some path variables:
HOMEDIR = os.environ['HOME']
caffe_root = HOMEDIR + '/caffe'
sys.path.insert(0, caffe_root + 'python')

PROCESSING_DIRECTORY = '/media/hagrid/Untitled/ilsvrc2012/ILSVRC2012_img_train'
OUTPUT_DIRECTORY = HOMEDIR + '/Desktop'
# ^directory needs to contain one folder per synset, like n0123456
# and in each folder a batch of pictures associated with the synset

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = caffe_root + '/models/bvlc_alexnet/deploy.prototxt'
PRETRAINED = caffe_root + '/models/bvlc_alexnet/bvlc_alexnet.caffemodel'

# Prepare the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(OUTPUT_DIRECTORY+'/Log_toyset.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s --- %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info("LETS GET STARTED, here we gooo")

caffe.set_mode_cpu()
#caffe.set_mode_gpu()

# load alexnet's NN model
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# create a dictionary that maps synset offset IDs to synset objects
senseIdToSynset = {s.offset(): s for s in wn.all_synsets()}

def processOneClass(thisDir):
    """
    Processes all images in one directory
    :param thisDir: directory where all the images of a class are stored
    :return: visual representations vectors for each image in a list
    """
    allVecs = []
    count = 0
    for imgName in os.listdir(thisDir):
        count += 1
        imgPath = thisDir + '/' + imgName
        print "Processing " + imgPath
        img = caffe.io.load_image(imgPath)
        net.predict([img])
        feature_vec = net.blobs['fc8'].data[0].copy()
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

outfile = open(OUTPUT_DIRECTORY+'/toyset_results.txt', 'w')
for dirs in os.listdir(PROCESSING_DIRECTORY):
    try:
        offID = int(dirs[1:])
        thisSet = senseIdToSynset[offID]
        logger.info("Processing synset " + str(offID))
    except:
        print "Cannot find the synset for offset ID " + str(offID)
        logger.error("Cannot find the synset for offset ID " + str(offID))
        continue

    t0 = time.time()
    try:
        vecs = processOneClass(PROCESSING_DIRECTORY + '/' + dirs)
        t_elapsed = time.time() - t0
        num_Imgs = len(vecs)
        logger.info(str(num_Imgs) + " images took this long to process (seconds): " + str(t_elapsed))
    except:
        logger.error("Unexpected error processing images in " + dirs)
        continue

    t0 = time.time()
    try:
        mean, std, ent, mp = computationsPerDimension(vecs)
        disp = calculateDispersion(vecs)
        t_elapsed = time.time() - t0
        logger.info("Vector computations took this long to process (seconds): " + str(t_elapsed))
    except:
        logger.error("Unexpected error performing computations for " + dirs)
        continue

    for lem in thisSet.lemmas():
        word = str(lem.name())
        outfile.write('-mean- ' + word + ' ' + str(mean.tolist()) + '\n')
        outfile.write('-maxpool- ' + word + ' ' + str(mp.tolist()) + '\n')
        outfile.write('-std- ' + word + ' ' + str(std.tolist()) + '\n')
        outfile.write('-entropy- ' + word + ' ' + str(ent.tolist()) + '\n')
        outfile.write('-dispersion- ' + word + ' ' + str(disp) + '\n')

outfile.close()

logger.info("Finished")

# TODO: make a class for words to store the visual representations and measures in
# TODO: store these objects in some file that can be easily loadable
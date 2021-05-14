#https://github.com/calebrob6/imagenet_validation
#Importing packages
import sys, os, time, re, gc
from pathlib import Path
from glob import glob
import json

# Select GPU
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2
import statistics as st
import time

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

#tf.__version__
# Make sure GPU is available
#tf.config.list_physical_devices('GPU')

#Size in human
def humansize(nbytes):
    '''From https://stackoverflow.com/questions/14996453/python-libraries-to-calculate-human-readable-filesize-from-bytes'''
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

#Read images
def load_images(image_paths, returned_shard, n_shards=5):
    """ loads images into memory. It only load and returns images of the 'returned_shard'.
        image_paths: a list of paths to images
        n_shards: number of shards to loaded images be divided.
        returned_shard: the part of images to be returned. 0 <= returned_shard < n_shards
    """
    assert 0 <= returned_shard < n_shards, "The argument returned_shard must be between 0 and n_shards"
    shard_size = len(image_paths) // n_shards
    sharded_image_paths = image_paths[returned_shard*shard_size:(returned_shard+1)*shard_size] if returned_shard < n_shards - 1 \
                     else image_paths[returned_shard*shard_size:]
    images_list = np.zeros((len(sharded_image_paths), 224, 224, 3), dtype=np.uint8)
    for i, image_path in enumerate(sharded_image_paths):
        # Load (in BGR channel order)
        image = cv2.imread(image_path)
        # Resize
        height, width, _ = image.shape
        new_height = height * 256 // min(image.shape[:2])
        new_width = width * 256 // min(image.shape[:2])
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # Crop
        height, width, _ = image.shape
        startx = width//2 - (224//2)
        starty = height//2 - (224//2)
        image = image[starty:starty+224,startx:startx+224]
        assert image.shape[0] == 224 and image.shape[1] == 224, (image.shape, height, width)
        images_list[i, ...] = image[..., ::-1]
    return images_list

#Top k accuracy
def top_k_accuracy(y_true, y_pred, k=1, tf_enabled=True):
    '''
    Calculates top_k accuracy of predictions. Expects both y_true and y_pred to be one-hot encoded.
#    numpy implementation is from: https://github.com/chainer/chainer/issues/606
#    '''
    if tf_enabled:
        argsorted_y = tf.argsort(y_pred)[:,-k:]
        matches = tf.cast(tf.math.reduce_any(tf.transpose(argsorted_y) == tf.argmax(y_true, axis=1, output_type=tf.int32), axis=0), tf.float32)
        return tf.math.reduce_mean(matches).numpy()
    else:
        argsorted_y = np.argsort(y_pred)[:,-k:]
        return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()

#Paths to files
#root = "/home/dmarcondes/ILSVRC2012_img_val/"
#root = "/scratch/8940900/imagenet_val"
root = "imagenet_val/"
path_imagenet_val_dataset = Path(root+"images/") # path/to/data/
dir_images = Path(root+"jpeg/") # path/to/images/directory
path_labels = Path(root+"data/ILSVRC2012_validation_ground_truth.txt")
path_synset_words = Path(root+"data/synset_words.txt")
path_meta = Path(root+"data/meta.mat")

#Converting labels
meta = scipy.io.loadmat(str(path_meta))
original_idx_to_synset = {}
synset_to_name = {}

for i in range(1000):
    ilsvrc2012_id = int(meta["synsets"][i,0][0][0][0])
    synset = meta["synsets"][i,0][1][0]
    name = meta["synsets"][i,0][2][0]
    original_idx_to_synset[ilsvrc2012_id] = synset
    synset_to_name[synset] = name

synset_to_keras_idx = {}
keras_idx_to_name = {}
with open(str(path_synset_words), "r") as f:
    for idx, line in enumerate(f):
        parts = line.split(" ")
        synset_to_keras_idx[parts[0]] = idx
        keras_idx_to_name[idx] = " ".join(parts[1:])

convert_original_idx_to_keras_idx = lambda idx: synset_to_keras_idx[original_idx_to_synset[idx]]

#Creating y_val
with open(str(path_labels),"r") as f:
    y_val = f.read().strip().split("\n")
    y_val = np.array([convert_original_idx_to_keras_idx(int(idx)) for idx in y_val])

np.save(str(path_imagenet_val_dataset/"y_val.npy"), y_val)

#Number of images
image_paths = sorted(glob(str(dir_images/"*")))
n_images = len(image_paths)
n_images
exist_shards = len(sorted(glob(str(path_imagenet_val_dataset/"*")))) - 1

#Dividing the images in 500 shards if it has not been done before
n_shards = 1
if n_shards != exist_shards:
    for i in range(n_shards):
        images = load_images(image_paths, returned_shard=i, n_shards=n_shards)
        if i == 0:
            print("Total memory allocated for loading images:", humansize(images.nbytes))
        np.save(str(path_imagenet_val_dataset / "x_val_{}.npy".format(i+1)), images)
        if (i + 1) * 100 / n_shards % 5 == 0:
            print("{:.0f}% Completed.".format((i+1)/n_shards*100))
        images = None

#Show examples
examples = False
if examples:
    idx_shard = 1
    x_val = np.load(str(path_imagenet_val_dataset/"x_val_{}.npy").format(idx_shard))
    n_images2show = 15
    n_cols = 3
    n_rows = 15 // n_cols
    figsize = (20, 20)
    indices = np.random.choice(x_val.shape[0], size=n_images2show, replace=False)
    images = x_val[indices] / 255.
    fig, ax = plt.subplots(figsize=figsize, nrows=n_rows, ncols=n_cols)
    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i])
        label_index = (idx_shard - 1) * (n_images // n_shards) + indices[i]
        axi.set_title(keras_idx_to_name[y_val[label_index]], y=.9, fontdict={'fontweight':'bold'}, pad=10)
        axi.set_axis_off()
    plt.show()

#Load data
x_val_path = glob(str(path_imagenet_val_dataset / "x_val*.npy"))
x_val_path.sort(key=lambda f: int(re.sub('\D', '', f)))
x_val = np.load(x_val_path[0]).astype('float32')
x_val = tf.keras.applications.mobilenet_v2.preprocess_input(x_val)
y_val = np.load(str(path_imagenet_val_dataset / "y_val.npy"))
y_val_one_hot = to_categorical(y_val, 1000)

#Model
model = tf.keras.applications.MobileNetV2(include_top=True,weights="imagenet")
depth = len(model.layers)
tab = np.array([])

for i in range(depth):
    if len(model.layers[i].get_weights()) > 0 or i == 0:
        print("layer "+str(i+1)+" from "+str(depth))
        err10 = np.array([])
        err5 = np.array([])
        err1 = np.array([])
        for r in range(3):
            print("Repetition " + str(r+1))
            modelRandom = tf.keras.applications.Xception(include_top=True,weights = None)
            if i > 0:
                for j in range(i):
                    modelRandom.layers[j].set_weights(model.layers[j].get_weights())
            y_pred = model.predict(x_val, verbose=1, use_multiprocessing=True, batch_size=64, callbacks=None)
            m10 = top_k_accuracy(y_val_one_hot, y_pred, k=10)
            m5 = top_k_accuracy(y_val_one_hot, y_pred, k=5)
            m1 = top_k_accuracy(y_val_one_hot, y_pred, k=1)
            err10 = np.concatenate([err10,[m10]])
            err5 = np.concatenate([err5,[m5]])
            err1 = np.concatenate([err1,[m1]])
            y_pred = None
            col = gc.collect()
        if(len(tab) > 0):
            tab = np.append(tab,np.array([(i,np.mean(err10),st.stdev(err10),np.percentile(err10,0),np.percentile(err10,25),np.percentile(err10,50),np.percentile(err10,75),np.percentile(err10,100),
            np.mean(err5),st.stdev(err5),np.percentile(err5,0),np.percentile(err5,25),np.percentile(err5,50),np.percentile(err5,75),np.percentile(err5,100)),
            np.mean(err1),st.stdev(err1),np.percentile(err1,0),np.percentile(err1,25),np.percentile(err1,50),np.percentile(err1,75),np.percentile(err1,100)]))
        else:
            tab = np.array([(i,np.mean(err10),st.stdev(err10),np.percentile(err10,0),np.percentile(err10,25),np.percentile(err10,50),np.percentile(err10,75),np.percentile(err10,100),
            np.mean(err5),st.stdev(err5),np.percentile(err5,0),np.percentile(err5,25),np.percentile(err5,50),np.percentile(err5,75),np.percentile(err5,100)),
            np.mean(err1),st.stdev(err1),np.percentile(err1,0),np.percentile(err1,25),np.percentile(err1,50),np.percentile(err1,75),np.percentile(err1,100)])
        np.savetxt("results.txt",tab,delimiter= ";")

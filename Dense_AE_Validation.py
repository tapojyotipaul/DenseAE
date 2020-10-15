# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:45:06 2020

@author: tapojyoti.paul
"""

import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# #from tensorflow.python.keras.backend import set_session
#tf.compat.v1.ConfigProto()
# config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config)) 
import numpy as np
from keras import initializers, regularizers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Activation, concatenate
import time
print("##################################################################")
print("Step 1: Loading Packages Complete")

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################
import time

########################################################################
# import additional python-library
########################################################################
import numpy 
# from import
from tqdm import tqdm
import yaml
import librosa
# original lib
#import common as com
#import keras_model
########################################################################

def yaml_load():
    with open("denseae.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################
# load parameter.yaml
########################################################################
param = yaml_load()
########################################################################
print("##################################################################")
print("Step 2: Loading YAML file Complete")
#########################################################################

from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
# #tensorflow.random.set_seed(x)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

print("##################################################################")
print("Step 3: Setting Seed Complete")

########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################

def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        print("file_broken or not exists!!")
        #logger.error("file_broken or not exists!! : {}".format(wav_name))
print("##################################################################")
print("Step 4: Loading 1st set of function Complete")

def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    #print("mel_spectrogram.shape: ",str(mel_spectrogram.shape))
    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array

def list_to_vector_array_baseline(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset

def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        if idx == 0:
            dataset = numpy.zeros((len(file_list),vector_array.shape[0] , dims), float)
        #dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
        dataset[idx, :] = vector_array

    return dataset
def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    print("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    #if len(files) == 0:
        #com.logger.exception("no_wav_file!!")

    #com.logger.info("train_file num : {num}".format(num=len(files)))
    return files

def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        print("load_directory <- development")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
        dirs = sorted(glob.glob(dir_path))
    else:
        print("load_directory <- evaluation")
        dir_path = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
        dirs = sorted(glob.glob(dir_path))
    return dirs
########################################################################
mode = True

# print("##################################################################")
# print("Step 5: Loading train function Complete")

# dirs = select_dirs(param=param, mode=mode)
# for idx, target_dir in enumerate(dirs):
#     break

# print(target_dir)
# files = file_list_generator(target_dir,ext="wav")
# print (files[-10:])
# print("##################################################################")
# print("Step 6: Loading train data loading Complete")

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    print("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        print("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            print("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        print("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            print("no_wav_file!!")
        print("\n=========================================")

    return files, labels

print("##################################################################")
print("Step 7: Loading test funtions Complete")


print("##################################################################")
print("Step 8: Define the model")

#inputDim = train_data.shape[1]
inputDim = 640
inputLayer = Input(shape=(inputDim,))

h = Dense(128)(inputLayer)
h = BatchNormalization()(h)
h = Activation('relu')(h)

h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)

h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)

h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)

h = Dense(8)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)

h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)

h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)

h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)

h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)

h = Dense(inputDim)(h)

denseAe1 = Model(inputs=inputLayer, outputs=h)
denseAe1.summary()

denseAe1.load_weights(r'/home/ubuntu/DenseAE/model_fan.hdf5')

denseAe1.compile(**param["fit"]["compile"])
print("Model Load Complete")


print("##################################################################")
print("Step 9: Started Validation of Data")

dirs = select_dirs(param=param, mode=True)
for idx, target_dir in enumerate(dirs):
    print(idx, target_dir)
    
import itertools
import re
import csv
from sklearn import metrics
csv_lines = []


machine_id_list = get_machine_id_list_for_test(target_dir)
machine_type = "fan"

if mode:
    # results by type
    csv_lines.append([machine_type])
    csv_lines.append(["id", "AUC", "pAUC"])
    performance = []

test_files, y_true = test_file_list_generator(target_dir,"id_06")
anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                         result=param["result_directory"],
                                                                         machine_type=machine_type,id_str="id_06")
anomaly_score_list = []

print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
y_pred = [0. for k in test_files]
for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
    data = file_to_vector_array(file_path,
                                n_mels=param["feature"]["n_mels"],
                                frames=param["feature"]["frames"],
                                n_fft=param["feature"]["n_fft"],
                                hop_length=param["feature"]["hop_length"],
                                power=param["feature"]["power"])
    errors = numpy.mean(numpy.square(data - denseAe1.predict(data)), axis=1)
    y_pred[file_idx] = numpy.mean(errors)
    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

if mode:
    # append AUC and pAUC to lists
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
    csv_lines.append(["id_06".split("_", 1)[1], auc, p_auc])
    performance.append([auc, p_auc])
    print("AUC : {}".format(auc))
    print("pAUC : {}".format(p_auc))
    
# Label for the output

print ("AUC on Actual Test Data is Checked... Calculating Computation time for different observations....")
X = data

STATS = '#, median, mean, std_dev, min_time, max_time, quantile_10, quantile_90'
import pandas as pd

def get_test_data(size: int = 1):
    """Generates a test dataset of the specified size""" 
    num_rows = len(X)
    test_df = X.copy()

    while num_rows < size:
        test_df = np.append(test_df, test_df, axis=0)
        num_rows = len(test_df)

    return test_df[:size]


def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    time_array = np.array(time_list)

    median = np.median(time_array)
    mean = np.mean(time_array)
    std_dev = np.std(time_array)
    max_time = np.amax(time_array)
    min_time = np.amin(time_array)
    quantile_10 = np.quantile(time_array, 0.1)
    quantile_90 = np.quantile(time_array, 0.9)

    basic_key = ["median","mean","std_dev","min_time","max_time","quantile_10","quantile_90"]
    basic_value = [median,mean,std_dev,min_time,max_time,quantile_10,quantile_90]

    dict_basic = dict(zip(basic_key, basic_value))
    
    return pd.DataFrame(dict_basic, index = [0])

import argparse
import logging

from pathlib import Path
from timeit import default_timer as timer

NUM_LOOPS = 100
def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_df = get_test_data(num_observations)
    data = test_df

    num_rows = len(test_df)
    # print(f"Running {NUM_LOOPS} inference loops with batch size {num_rows}...")

    run_times = []
    inference_times = []
    for _ in range(NUM_LOOPS):

        start_time = timer()
        denseAe1.predict(data)
        end_time = timer()

        total_time = end_time - start_time
        run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/num_rows
        inference_times.append(inference_time)

    print(num_observations, ", ", calculate_stats(inference_times))
    return calculate_stats(inference_times)

STATS = '#, median, mean, std_dev, min_time, max_time, quantile_10, quantile_90'

if __name__=='__main__':
    ob_ct = 1  # Start with a single observation
    logging.info(STATS)
    temp_df = pd.DataFrame()
    print("Inferencing Started")
    while ob_ct <= 100000:
        temp = run_inference(ob_ct)
        temp["No_of_Observation"] = ob_ct
        temp_df = temp_df.append(temp)
        ob_ct *= 10
    print("Summary........")
    print(temp_df)
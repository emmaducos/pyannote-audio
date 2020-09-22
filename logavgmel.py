#!/usr/bin/env python
# encoding: utf-8
"""
To compute the averaged log mel spectrograms of waveforms
Needs a file with the uris of the data on each lines.
for every sound sample, it will:
- load the waveform
- get some information from it with pyannote functions
- create a RawAudio data loader (pyannote-audio specialty)
- compute the log mel spectrogram with a function from the librosa library
- compute an averaged version
- save it into a numpy array format
emma ducos - emma.ducos@hotmail.fr
"""

import os
import numpy as np
from pyannote.audio.features.utils import get_audio_duration, get_audio_sample_rate
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.features.with_librosa import LibrosaMelSpectrogram
from pyannote.core import Segment



# paths
# put here where to find the data
TUT_DATASET = "/home/emma/coml/dataset/TUT"

# this is based on the TUT dataset architecture
# needs to be fitted to your dataset structure
TUT_DEVELOPMENT = os.path.join(TUT_DATASET, "TUT-sound-events-2017-development/")
TEST_FILE = os.path.join(TUT_DEVELOPMENT, "audio/street/a001.wav")
LSTS = os.path.join(TUT_DATASET, "lsts")

# one for training, one for evaluation
TRAIN = os.path.join(LSTS, "dev_fold1_train.lst")
EVAL = os.path.join(LSTS, "dev_fold1_evaluate.lst")

# this is where you will find the result
SAVE_DIR = os.path.join(TUT_DATASET, "logavgmel")


f = open(TRAIN, "r")


for uri in f.readlines():
    print(uri)
    # load waveform
    current_file = {'audio': os.path.join(TUT_DEVELOPMENT, "audio/street/{uri}.wav".format(uri=uri.strip("\n")))}
    duration = get_audio_duration(current_file=current_file)
    sample_rate = get_audio_sample_rate(current_file=current_file)
    raw_audio = RawAudio(sample_rate=sample_rate, mono=True, augmentation=None)
    # print(sample_rate)
    # print(duration)
    start = 0.0
    segment = Segment(start, duration)
    # compute mel
    n_mels = 40
    waveform = raw_audio.crop(current_file, segment=segment)
    # print(waveform.shape)
    librosa = LibrosaMelSpectrogram(sample_rate=sample_rate,
                                    augmentation=None,
                                    n_mels=n_mels)
    melspec = librosa.get_features(y=waveform, sample_rate=sample_rate)
    # print(melspec.shape)

    logavgmel = np.mean(melspec, axis=0, keepdims=True)
    # print(logavgmel.shape)

    # save, use numpy
    np.save(os.path.join(SAVE_DIR, "{uri}_logavgmel.npy".format(uri=uri.strip("\n"))), logavgmel)

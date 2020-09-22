# `pyannote-audio` | neural building blocks for speaker diarization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyannote/pyannote-audio/blob/develop/notebooks/introduction_to_pyannote_audio_speaker_diarization_toolkit.ipynb)


! Attention ! This branch and fork concerns the work made by Emma Ducos during an internship at the lab of Pr. Emmanuel Dupoux (Cognitive Machine Learning, LSCP, DEC, ENS).


`pyannote.audio` is an open-source toolkit written in Python for speaker diarization. Based on [PyTorch](pytorch.org) machine learning framework, it provides a set of trainable end-to-end neural building blocks that can be combined and jointly optimized to build speaker diarization pipelines:

<p align="center"> 
<img src="pipeline.png">
</p>

`pyannote.audio` also comes with [pretrained models](https://github.com/pyannote/pyannote-audio-hub) covering a wide range of domains for voice activity detection, speaker change detection, overlapped speech detection, and speaker embedding:

![segmentation](tutorials/pretrained/model/segmentation.png)

## Installation

`pyannote.audio` only supports Python 3.7 (or later) on Linux and macOS. It might work on Windows but there is no garantee that it does, nor any plan to add official support for Windows.

The instructions below assume that `pytorch` has been installed using the instructions from https://pytorch.org.

Until a proper release of `pyannote.audio` is available on `PyPI`, it must be installed from source:

```bash
$ git clone https://github.com/emmaducos/pyannote-audio.git
$ cd pyannote-audio
$ git checkout emma
$ conda env create -f env.yml   # This will create a conda environment called `pyannote`
$ conda activate pyannote       # You must activate this environment each time you want to run a pyannote command
```


## Documentation

Part of the API is described in [this](tutorials/pretrained/model) tutorial.  

Documentation is a work in progress. Go to the main branch to get a better idea.



### Specificity of this branch work
This is work trying to replicate the work described in this paper:

```yaml
@techreport{Jeong2017,
Author = "Jeong, Il-Young and Lee, Subin and Han, Yoonchang and Lee, Kyogu",
institution = "DCASE2017 Challenge",
title = "Audio Event Detection Using Multiple-Input Convolutional Neural Network",
abstract = "This paper describes the model and training framework from our submission for DCASE 2017 task 3: sound event detection in real life audio. Our model basically follows convolutional neural network architecture, yet uses two input data of the short- and long-term audio signal. In the training stage, we calculated validation errors more frequently than one epoch with adaptive thresholds. We also used class-wise early stopping to find the best model for each class. The proposed model shows a meaningful improvements in cross validation experiments compared to the baseline system using the simple neural network.",
month = "September",
year = "2017"
}
```

From the DCASE 2017 challenge, on task 3 "sound event detection on real-life audio".

This fork has been made in April 2020, from the work of MarvinLvn:develop.
It is actually (22/09/2020) 58 commits behind this branch, and I don't know how many from the main one, so some features of the original piece of software may not be present here.

The core of the software can be found inside the `pyannote/audio/` folder.
Inside this folder, in the following folders: `application/`, `features/`, `labeling/tasks/`, `models/`.
I have added a `dcase.py` file and change the `__init__.py` file accordingly.
In `pyannote/audio/applications/pyannote-audio.py`, I have made changes so you can call a `dcase` argument.
Other than that, all the changes are inside the `dcase.py` files, that are children classes of the classes of MarvinLvn's Multilabel Detection pipeline.
I have added a `TODO: dcase` comment everywhere I made changes to the code under.

The current code should launch for training and validation procedure, but do not expect results. 
There is no evaluation procedure coded yet for this specific task.
To get ideas, look at: [DCASE 2017 sed_eval toolbox](https://github.com/TUT-ARG/sed_eval)

I have not tried the other available pipelines, so they may need debugging.

I will try to add a comment on the why's everywhere I added a piece of code. 
Please let me know if you have any questions on the matter by contacting me at this address:
emma.ducos@hotmail.fr


## Tutorials
Be sure to first follow this tutorial:
* [Prepare your own dataset for training or fine-tuning](tutorials/data_preparation)

You need to have the rttm files that describes the data and its annotation in a certain format.
You can find my script to creates these ones in `preprocess.py`

You have to compute first the log mel averaged spectograms of your input.
You can use the script `logavgmel.py`.

Then in `pyannote/audio/applications/dcase.py`, in the `validate_init()` function,
you have to manually add the path of the logavgmels you computed with the precedent script. 
You also need to do that in `pyannote/audio/labeling/task/dcase.py`, in the `_long_and_short_logmel_samples()` function.

Other tutorials that are close to this task:
* Train models on your own data
  * [Multilabel detection](tutorials/models/multilabel_detection)
* Tune pipelines on your own data
  * [Speech activity detection pipeline](tutorials/pipelines/speech_activity_detection)
  
For the other models, see the main branch `develop`, or go to the original `pyannote-audio` github.
The other models/tasks/pipelines might not work properly.




### Scripts to launch 
To launch the training, you can use the script:
```bash
#!/bin/sh

export EXP_DIR=/path/to/your/experiment/directory

pyannote-audio dcase train --subset=train --debug --cpu --to=10 --parallel=4 ${EXP_DIR} TUT2017.SpeakerDiarization.MyProtocol
```
Don't forget to chant the experiment directory path in `EXP_DIR`.

You can simutaneously launch the validation with this script:
```bash
#!/bin/sh

export CUDA_VISIBLE_DEVICES=gpu2
export EXP_DIR=/path/to/your/experiment/directory
export TRN_DIR=${EXP_DIR}/train/TUT2017.SpeakerDiarization.MyProtocol.train
export VAL_DIR=${TRN_DIR}/validate_average_detection_fscore/TUT2017.SpeakerDiarization.MyProtocol.development

pyannote-audio dcase validate --subset=development --every=1 ${TRN_DIR} TUT2017.SpeakerDiarization.MyProtocol
```
Don't forget to chant the experiment directory path in `EXP_DIR`, and the name of the gpu device for quicker computation.

#### config.yml
The `config.yml` file that I used for my experiments.
```yaml
# A multilabel sound event detection model is trained.
# Here, training relies on 2s-long audio chunks,
# batches of 32 audio chunks, and saves model to
# disk every one (1) day worth of audio.
# The labels_spec specifies how to build the classes
# of the model given the raw label.

# Here, we consider 6 regular labels which are
# car, children, people_speaking, people_walking, 
# large_vehicle, brakes_squeaking. It will 
# account for 6 dimensions in the predicted vector. 
# To sum up, our model will output a vector of 6 scores :
# [car, children, people_speaking, people_walking, large_vehicle, brakes_squeaking]

# The `logagmel` parameters will take into account long-term input
# but It is not working if it is not True
# the other parameters have not been changed
task:
   name: Dcase
   params:
      logavgmel: True
      duration: 2.0
      batch_size: 32
      per_epoch: 1
      labels_spec:
         regular: ['car', 'children', 'people_speaking', 'people_walking', 'large_vehicle', 'brakes_squeaking']

# this part takes advantage of pyannote wrapper for feature extraction
# to create the short-term log mel spectrogram
# the `sample_rate` and the number of mel bins `n_mels` are given by Jeong:2018
feature_extraction:
   name: LibrosaMelSpectrogram
   params:
      sample_rate: 44100 
      n_mels: 40

# The 'params' are not used in the dcase model, 
# but left because it creates some bugs, 
# since my model 'Dcase' is based on PyanNet
architecture:
   name: pyannote.audio.models.Dcase
   params:
      sincnet: {'skip': True}
      rnn:
         unit: LSTM
         hidden_size: 128
         num_layers: 2
         bidirectional: True
      ff:
         hidden_size: [128, 128]  

# We use a constant learning rate of 1e-2
# I have yet to change anything about this part
scheduler:
   name: ConstantScheduler
   params:
      learning_rate: 0.01
```

## Citation

If you use `pyannote.audio` please use the following citation

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```

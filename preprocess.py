"""
To preprocess TUT annotation files into rttm and lst formats
This program:
- load the annotation
- get the right information from it
- create the rttm/lst file with it
- save in a text format

emma ducos - emma.ducos@hotmail.fr
"""

import pandas as pd
import os

# if true then compute
rttms = True
lsts = True

# can be 'test' or 'train' depending on what part of the dataset you want to work with
step = 'test'

# the paths, depends on your dataset architecture
TUT_DATASET_PATH = "/home/emma/coml/dataset/TUT"
TUT_DEVELOPMENT_PATH = os.path.join(TUT_DATASET_PATH, "TUT-sound-events-2017-development")

# where you want to save them
RTTM_PATH = os.path.join(TUT_DATASET_PATH, 'rttms')
LST_PATH = os.path.join(TUT_DATASET_PATH, "lsts")

# load the information of the dataset
dev_fold1_df = pd.read_csv(os.path.join(TUT_DEVELOPMENT_PATH,
                                        "evaluation_setup/street_fold1_{step}.txt".format(step=step)),
                           sep="\t", header=None,
                           names=["audio_file", "scene_label", "event_onset", "event_offset", "event_label"])
print(dev_fold1_df.head())

# get the audio file
audio_file = dev_fold1_df.audio_file.to_list()
# print(audio_file)

# get the onset
event_onset = dev_fold1_df.event_onset.to_list()
# print(event_onset)
# print(len(event_onset))

# get the offset
event_offset = dev_fold1_df.event_offset.to_list()
# print(event_offset)
# print(len(event_offset))

# get the event label
event_label = dev_fold1_df.event_label.to_list()
# print(event_label)
# print(len(event_label))


# you can also put lsts to compute them here
if rttms:
    # get the uris, depends on your file format
    uris = []
    for file in audio_file:
        file = file.split('/')
        file = file[2].split('.')
        uris.append(file[0])
    # print(uris)
    # print(len(uris))

    starts = event_onset

    durations = []
    identifiers = []
    for i in range(len(event_onset)):
        durations.append(event_offset[i] - event_onset[i])
        identifiers.append(event_label[i].replace(" ", "_"))
    # print(durations)
    # print(identifiers)

    # create the rttm file
    rttm = open(os.path.join(RTTM_PATH, 'dev_fold1_{step}.rttm'.format(step=step)), 'w+')

    for i in range(len(uris)):
        rttm.write("SPEAKER {uri} 1 {start:.6f} {duration:.6f} <NA> <NA> {identifier} <NA> <NA>\n".format(uri=uris[i],
                                                                                                          start=starts[
                                                                                                              i],
                                                                                                          duration=
                                                                                                          durations[i],
                                                                                                          identifier=
                                                                                                          identifiers[
                                                                                                              i]))
    rttm.close()

if lsts:
    # same as for rttm but for the lst format
    uris = []
    for file in audio_file:
        file = file.split('/')
        file = file[2].split('.')
        if file[0] not in uris:
            uris.append(file[0])
    # print(uris)
    # print(len(uris))

    lst = open(os.path.join(LST_PATH, "dev_fold1_{step}.lst".format(step=step)), 'w+')
    for i in range(len(uris)):
        lst.write("{uri}\n".format(uri=uris[i]))
    lst.close()

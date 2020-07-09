#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Marvin LAVECHIN - marvinlavechin@gmail.com

"""Multilabel detection dcase"""

from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import cycle
from .base import LabelingTask
from .base import LabelingTaskGenerator
from pyannote.audio.train.task import Task, TaskType, TaskOutput
from ..gradient_reversal import GradientReversal
from pyannote.audio.models.models import RNN
from pyannote.core import Timeline, Annotation, SlidingWindowFeature
from pyannote.core.utils.numpy import one_hot_encoding
from pyannote.database import get_annotated, get_protocol

import tqdm
#import logging
#logging.basicConfig(filename='/home/emma/coml/experiments/pynt_test/print.log', level=logging.DEBUG)
from .labels_detection import MultilabelDetection, MultilabelDetectionGenerator
from pyannote.core.utils.random import random_segment
from pyannote.core.utils.random import random_subsegment



class DcaseGenerator(MultilabelDetectionGenerator):
    """Batch generator for training multilabel detection, following jeong2017 (DCASE 2017) ideas

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}, optional
        Protocol and subset
    labels_spec : `dict`
        Describes the labels that must be predicted.
        1) Must contain a 'regular' key listing the labels appearing 'as-is' in the dataset.
        2) Might contain a 'union' key listing the {key, values} where key is the name of
        the union_label that needs to be predicted, and values is the list of labels
        that will construct the union_label (useful to construct speech classes).
        3) Might contain a 'intersection' key listing the {key, values} where key is the name of
        the intersection_label that needs to be predicted, and values is the list of labels
        that will construct the intersection_label (useful to construct overlap classes).
    resolution : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
    alignment : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models
        that include the feature extraction step (e.g. SincNet) and
        therefore use a different cropping mode. Defaults to 'center'.

    logavgmel : bool, optional

    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.


    Usage
    -----
    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/mfcc')

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('BabyTrain.SpeakerRole.JSALT')

    # labels specification
    # This labels_spec will create a model that predicts
    # ['CHI', 'FEM', 'MAL', 'SPEECH', 'ADULT_SPEECH', 'OVL'] classes
    >>> labels_spec = {'regular': ['CHI', 'FEM', 'MAL'],
    >>>                'union': {
    >>>                     'SPEECH' : ['CHI', 'FEM', 'MAL']
    >>>                     'ADULT_SPEECH': ['FEM','MAL']
    >>>                 },
    >>>                 'intersection': {
    >>>                     'OVL' : ['CHI', 'FEM', 'MAL']
    >>>                 }
    >>>

     # instantiate batch generator
    >>> batches = DcaseGenerator(precomputed, protocol, labels_spec)

    # iterate over training set
    >>> for batch in batches(protocol, subset='train'):
    >>>     # batch['X'] is a (batch_size, n_samples, n_features) numpy array
    >>>     # batch['y'] is a (batch_size, n_samples, n_tasks) numpy array
    >>>     pass
    """

    def __init__(self,
                 feature_extraction,
                 protocol,
                 labels_spec,
                 subset='train',
                 resolution=None,
                 alignment=None,
                 logavgmel: bool = False,
                 duration=3.2,
                 batch_size=32,
                 per_epoch: float = None):

        self.logavgmel = logavgmel
        self.labels_spec = labels_spec
        super().__init__(feature_extraction,
                         protocol,
                         labels_spec=labels_spec,
                         subset=subset,
                         resolution=resolution,
                         alignment=alignment,
                         duration=duration,
                         batch_size=batch_size,
                         per_epoch=per_epoch)

    def initialize_y(self, current_file):
        # eprint("tasks dcase initialize_y")
        # First, one hot encode the regular classes
        annotation = current_file['annotation'].subset(self.labels_spec['regular'])
        y, _ = one_hot_encoding(annotation,
                                get_annotated(current_file),
                                self.resolution,
                                labels=self.labels_spec["regular"],
                                mode='center')
        y_data = y.data
        # Then, one hot encode the meta classes
        for derivation_type in ['union', 'intersection']:
            for meta_label, regular_labels in self.labels_spec[derivation_type].items():
                derived = Dcase.derives_label(current_file["annotation"], derivation_type, meta_label,
                                                            regular_labels)
                z, _ = one_hot_encoding(derived, get_annotated(current_file),
                                        self.resolution,
                                        labels=[meta_label],
                                        mode='center')

                y_data = np.hstack((y_data, z.data))

        return SlidingWindowFeature(self.postprocess_y(y_data),
                                    y.sliding_window)

    @property
    def specifications(self):
        specs = {
            'task': Task(type=TaskType.MULTI_LABEL_CLASSIFICATION,
                         output=TaskOutput.SEQUENCE),
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': self.labels_spec["regular"] \
                             + list(self.labels_spec['union']) \
                             + list(self.labels_spec['intersection'])},
        }

        for key, classes in self.file_labels_.items():

            # TODO. add an option to handle this list
            # TODO. especially useful for domain-adversarial stuff
            if key in ['duration', 'audio', 'uri']:
                continue
            specs[key] = {'classes': classes}

        return specs

    def samples(self):
        if self.exhaustive:
            return self._sliding_samples()
        if self.logavgmel:
            return self._long_and_short_logmel_samples()
        else:
            return self._random_samples()


    def _long_and_short_logmel_samples(self):
        """Random samples
        TODO
        changed to follow jeong2017

        Returns
        -------
        samples : generator
            Generator that yields {'X': ..., 'y': ...} samples indefinitely.
        """
        eprint("tasks dcase random sample")
        uris = list(self.data_)
        eprint(uris)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        #eprint(durations)
        probabilities = durations / np.sum(durations)
        #eprint(probabilities)

        # compute the logmel average on the entire uri

        while True:

            # choose file at random with probability
            # proportional to its (annotated) duration
            uri = uris[np.random.choice(len(uris), p=probabilities)]
            # eprint(uri)
            datum = self.data_[uri]
            # eprint(datum)
            current_file = datum['current_file']
            # eprint(current_file)

            # load long average logmel on the entire waveform of the uri

            # choose one segment at random with probability
            # proportional to its duration
            # eprint(datum['segments'][0])
            segment = next(random_segment(datum['segments'], weighted=True))
            # eprint("segment:", segment)

            # choose fixed-duration subsegment at random
            subsegment = next(random_subsegment(segment, self.duration))
            # eprint("subsegment:", subsegment)

            X = self.feature_extraction.crop(current_file,
                                             subsegment,
                                             mode='center',
                                             fixed=self.duration)
            # eprint(X)
            y = self.crop_y(datum['y'],
                            subsegment)
            # eprint(y)
            sample = {'X': X, 'y': y}
            # eprint(sample)
            if self.mask is not None:
                mask = self.crop_y(current_file[self.mask],
                                   subsegment)
                sample['mask'] = mask

            for key, classes in self.file_labels_.items():
                sample[key] = classes.index(current_file[key])

            yield sample


class Dcase(MultilabelDetection):
    """Train multilabel detection

    - Regular labels : those are extracted directly from the annotation and are kept unchanged.
    - Union meta-label : those are extracted by taking the union of multiple regular labels.
    - Intersection meta-label : those are extracted by taking the intersection of multiple regular labels.

    Parameters
    ----------
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    label_spec: `dict`
        regular: list
            List of classes that need to be predicted.
        union:
            Dictionnary of union meta-labels whose keys are the meta-label names,
            and values are a list of regular classes
        intersection:
            Dictionnary of intersection meta-labels whose keys are the meta-label names,
            and values are a list of regular classes
    logavgmel: bool, optional TODO
    """

    def __init__(self, labels_spec, logavgmel=False, **kwargs):
        # labels_spec = {'regular': ['car', 'children', 'people_speaking', 'people_walking', 'large_vehicle', 'brakes_squeaking']}
        super(Dcase, self).__init__(labels_spec, **kwargs)

        # Labels related attributes
        self.labels_spec = labels_spec
        labels_spec_key = self.labels_spec.keys()
        if 'regular' not in labels_spec_key:
            self.labels_spec['regular'] = dict()
        if 'union' not in labels_spec_key:
            self.labels_spec['union'] = dict()
        if 'intersection' not in labels_spec_key:
            self.labels_spec['intersection'] = dict()

        self.regular_labels = self.labels_spec['regular']
        self.union_labels = list(self.labels_spec['union'])
        self.intersection_labels = list(self.labels_spec['intersection'])

        self.label_names = self.regular_labels + \
                           self.union_labels + \
                           self.intersection_labels

        if set(self.union_labels).intersection(self.intersection_labels):
            raise ValueError("Union keys and intersection keys in "
                             "labels_spec should be mutually exclusive.")

        self.nb_regular_labels = len(labels_spec["regular"])

        self.n_classes_ = self.nb_regular_labels + len(self.union_labels) + len(self.intersection_labels)

        self.logavgmel = logavgmel

    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            resolution=None, alignment=None):
        """
        resolution : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        alignment : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.
        """
        eprint("tasks dcase get_batch_generator")
        return DcaseGenerator(
            feature_extraction,
            protocol, subset=subset,
            resolution=resolution,
            alignment=alignment,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            labels_spec=self.labels_spec)

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

import numpy as np

from pyannote.audio.train.task import Task, TaskType, TaskOutput

from pyannote.core import SlidingWindowFeature
from pyannote.core.utils.numpy import one_hot_encoding
from pyannote.database import get_annotated

from .labels_detection import MultilabelDetection, MultilabelDetectionGenerator
from pyannote.core.utils.random import random_segment
from pyannote.core.utils.random import random_subsegment

try:
    from typing import Literal
except ImportError as e:
    from typing_extensions import Literal

import torch

ARBITRARY_LR = 0.1


class DcaseGenerator(MultilabelDetectionGenerator):
    """Batch generator for training multilabel detection, following jeong2017 (DCASE 2017) ideas
    #TODO: dcase
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
    #TODO: dcase
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

        # TODO: dcase
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

    def samples(self):
        if self.exhaustive:
            return self._sliding_samples()
        # TODO: dcase
        if self.logavgmel:
            return self._long_and_short_logmel_samples()
        else:
            return self._random_samples()

    def _long_and_short_logmel_samples(self):
        """Random samples
        TODO: dcase
        changed to follow jeong2017
        This is the batch generator to get matching short/long samples

        Returns
        -------
        samples : generator
            Generator that yields {'X': ..., 'y': ...} samples indefinitely.
        """
        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)

        while True:
            # choose file at random with probability
            # proportional to its (annotated) duration
            uri = uris[np.random.choice(len(uris), p=probabilities)]
            datum = self.data_[uri]
            current_file = datum['current_file']

            # TODO: dcase
            # load long average logmel on the entire waveform of the uri
            LOGAVGMEL = "/home/emma/coml/dataset/TUT/logavgmel/{uri}_logavgmel.npy".format(uri=uri.split('/')[1])
            # transpose to get (nb_mel, nb_frame)
            logavgmel = np.load(LOGAVGMEL).T

            # choose one segment at random with probability
            # proportional to its duration
            segment = next(random_segment(datum['segments'], weighted=True))

            # choose fixed-duration subsegment at random
            subsegment = next(random_subsegment(segment, self.duration))

            X = self.feature_extraction.crop(current_file,
                                             subsegment,
                                             mode='center',
                                             fixed=self.duration)
            # TODO: dcase
            # transposed to get (nb_mel, nb_frame)
            X = X.T

            y = self.crop_y(datum['y'],
                            subsegment)

            # TODO: dcase
            # add 'logavgmel info in batch'
            sample = {'X': X, 'y': y, 'logavgmel': logavgmel}

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
    TODO: dcase
    logavgmel: bool, optional
    """

    def __init__(self, labels_spec, logavgmel=False, **kwargs):
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
        # TODO: dcase
        self.logavgmel = logavgmel

    def get_batch_generator(self,
                            feature_extraction,
                            protocol,
                            subset='train',
                            resolution=None,
                            alignment=None):
        """
        TODO: dcase
        resolution : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        alignment : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.
        """
        return DcaseGenerator(
            feature_extraction,
            protocol, subset=subset,
            resolution=resolution,
            alignment=alignment,
            # TODO: dcase
            logavgmel=self.logavgmel,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            labels_spec=self.labels_spec)

    def batch_loss(self, batch):
        """Compute loss for current `batch`
        TODO: dcase
        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)
            ['mask'] (`numpy.ndarray`, optional)
            ['logavgmel'] (`numpy.ndarray`, optional)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Loss
        """
        # forward pass
        X = torch.tensor(batch['X'],
                         dtype=torch.float32,
                         device=self.device_)
        # TODO: dcase
        if 'logavgmel' in batch:
            logavgmel = torch.tensor(batch['logavgmel'],
                                     dtype=torch.float32,
                                     device=self.device_)
        X_logavgmel = {'X': X, 'logavgmel': logavgmel}
        fX = self.model_(X_logavgmel)

        mask = None
        if self.task_.is_multiclass_classification:

            fX = fX.view((-1, self.n_classes_))

            target = torch.tensor(
                batch['y'],
                dtype=torch.int64,
                device=self.device_).contiguous().view((-1,))

            if 'mask' in batch:
                mask = torch.tensor(
                    batch['mask'],
                    dtype=torch.float32,
                    device=self.device_).contiguous().view((-1,))

        elif self.task_.is_multilabel_classification or \
                self.task_.is_regression:

            target = torch.tensor(
                batch['y'],
                dtype=torch.float32,
                device=self.device_)
            # there is repetition on the second dimension,
            # plus need input and target to be of the same dim for loss
            target = target[:, 0, :].unsqueeze(1)

            if 'mask' in batch:
                mask = torch.tensor(
                    batch['mask'],
                    dtype=torch.float32,
                    device=self.device_)

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=self.device_)

        return {
            'loss': self.loss_func_(fX, target,
                                    weight=weight,
                                    mask=mask),
        }

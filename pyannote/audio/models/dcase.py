#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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
# Hervé BREDIN - http://herve.niderb.fr
# Emma DUCOS - emma.ducos@hotmail.fr

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sincnet import SincNet
from .models import Embedding, FF, RNN, PyanNet
from pyannote.audio.train.model import Model
from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import RESOLUTION_CHUNK
from pyannote.audio.train.model import RESOLUTION_FRAME


class Dcase(PyanNet):
    """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output
    TODO
    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters. Use {'skip': True} to use handcrafted features
        instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
        becomes [ features -> RNN -> ...].
    rnn : `dict`, optional
        Recurrent network parameters. Defaults to `RNN` default parameters.
    ff : `dict`, optional
        Feed-forward layers parameters. Defaults to `FF` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    """

    @staticmethod
    def get_alignment(sincnet=None, **kwargs):
        """
        """

        if sincnet is None:
            sincnet = dict()

        if sincnet.get('skip', False):
            return 'center'

        return SincNet.get_alignment(**sincnet)

    supports_packed = False

    @staticmethod
    def get_resolution(sincnet: Optional[dict] = None,
                       rnn: Optional[dict] = None,
                       **kwargs) -> Resolution:
        """Get sliding window used for feature extraction

        Parameters
        ----------
        sincnet : dict, optional
        rnn : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
            Returns RESOLUTION_CHUNK if model returns one vector per input
            chunk, RESOLUTION_FRAME if model returns one vector per input
            frame, and specific sliding window otherwise.
        """

        if rnn is None:
            rnn = {'pool': None}

        if rnn.get('pool', None) is not None:
            return RESOLUTION_CHUNK

        if sincnet is None:
            sincnet = {'skip': False}

        if sincnet.get('skip', False):
            return RESOLUTION_FRAME

        return SincNet.get_resolution(**sincnet)

    def init(self,
             sincnet: Optional[dict] = None,
             rnn: Optional[dict] = None,
             ff: Optional[dict] = None,
             embedding: Optional[dict] = None):
        """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output
        TODO
        add cnn
        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters. Use {'skip': True} to use handcrafted features
            instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
            becomes [ features -> RNN -> ...].
        rnn : `dict`, optional
            Recurrent network parameters. Defaults to `RNN` default parameters.
        ff : `dict`, optional
            Feed-forward layers parameters. Defaults to `FF` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        n_features = self.n_features

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if not sincnet.get('skip', False):
            if n_features != 1:
                msg = (
                    f'SincNet only supports mono waveforms. '
                    f'Here, waveform has {n_features} channels.'
                )
                raise ValueError(msg)
            self.sincnet_ = SincNet(**sincnet)
            n_features = self.sincnet_.dimension

        if rnn is None:
            rnn = dict()
        self.rnn = rnn
        self.rnn_ = RNN(n_features, **rnn)
        n_features = self.rnn_.dimension

        if ff is None:
            ff = dict()
        self.ff = ff
        self.ff_ = FF(n_features, **ff)
        n_features = self.ff_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
            return

        # instantiate all layers

        self.conv_logavgmel_1 = nn.Conv1d(in_channels=40,
                                          out_channels=64,
                                          kernel_size=1)

        self.conv_logavgmel_2 = nn.Conv1d(in_channels=64,
                                          out_channels=64,
                                          kernel_size=1)

        self.conv_logmel_1 = nn.Conv1d(in_channels=40,
                                       out_channels=64,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       padding_mode="replicate")

        self.conv_logmel_2 = nn.Conv1d(in_channels=64,
                                       out_channels=64,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       padding_mode="replicate")

        self.pool_logmel = nn.MaxPool1d(kernel_size=3)

        self.conv_merged = self.conv_logmel_2
        self.pool_merged = self.pool_logmel

        self.fc1 = nn.Linear(in_features=2,
                             out_features=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=64,
                             out_features=len(self.classes))
        self.activation_ = self.task.default_activation

    def forward(self, input, return_intermediate=None):
        """Forward pass
        TODO
        Parameters
        ----------
        input : `dict`
            ['X'] (batch_size, n_samples, 1) `torch.Tensor`
                Batch of waveforms. In case SincNet is skipped, a tensor with shape
                (batch_size, n_samples, n_features) is expected.
            ['logavgmel'] (n_mel, 1) `torch.Tensor`
        return_intermediate : `int`, optional
            Index of RNN layer. Returns RNN intermediate hidden state.
            Defaults to only return the final output.

        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        intermediate : `torch.Tensor`
            Intermediate network output (only when `return_intermediate`
            is provided).
        """
        # print(">>>models dcase forward")

        logmel = input['X']
        # print("logmel: ", logmel.shape)  # (64, 40, 200)
        logavgmel = input['logavgmel']
        # print("logavgmel: ", logavgmel.shape)  # (64, 40, 1)

        # if self.sincnet.get('skip', False):
        #     output = logmel
        # else:
        #     output = self.sincnet_(logmel)
        #
        # if return_intermediate is None:
        #     output = self.rnn_(output)
        # else:
        #     if return_intermediate == 0:
        #         intermediate = output
        #         output = self.rnn_(output)
        #     else:
        #         return_intermediate -= 1
        #         # get RNN final AND intermediate outputs
        #         output, intermediate = self.rnn_(output, return_intermediate=True)
        #         # only keep hidden state of requested layer
        #         intermediate = intermediate[return_intermediate]

        # conv1d layer x2 on logavgmel
        logavgmel = self.conv_logavgmel_1(logavgmel)
        logavgmel = F.relu(logavgmel)
        # print("conv_long_1 :", logavgmel.shape)  # (64, 64, 1)
        logavgmel = self.conv_logavgmel_2(logavgmel)
        logavgmel = F.relu(logavgmel)
        # print("conv_long_2 :", logavgmel.shape)  # (64, 64, 1)

        # (conv pool) x2 on logmel
        logmel = self.conv_logmel_1(logmel)
        logmel = F.relu(logmel)
        # print("conv_short_1 :", logmel.shape)  # (64, 64, 200)
        logmel = self.pool_logmel(logmel)
        # print("pool_short_1 :", logmel.shape)  # (64, 64, 66)
        logmel = self.conv_logmel_2(logmel)
        logmel = F.relu(logmel)
        logmel = self.pool_logmel(logmel)
        # print("conv relu pool short 2 :", logmel.shape)  # (64, 64, 22)

        # repeat layer on logavgmel
        logavgmel = logavgmel.repeat(1, 1, logmel.shape[2])
        # print("repeat long :", logavgmel.shape)

        # merge logmel (short) with logavgmel (long)
        # sum operation with add
        output = logmel.add(logavgmel)
        # print("add :", output.shape)

        # (conv relu pool) x2 on the merged tensor
        output = self.conv_merged(output)
        output = F.relu(output)
        output = self.pool_merged(output)
        # print("conv relu pool merged 1: ", output.shape)
        output = self.conv_merged(output)
        output = F.relu(output)
        output = self.pool_merged(output)
        # print("conv relu pool merged 2: ", output.shape)

        # fully connected layer
        output = self.fc1(output)
        output = F.relu(output)
        # print("fc merged :", output.shape)

        # dropout
        output = self.dropout(output)
        # print("dropout :", output.shape)

        # if self.task.is_representation_learning:
        #     return self.embedding_(output)

        # last fc and activation
        # output = output.view(64, -1)
        # print(output.shape)
        output = self.fc2(output.transpose(1, 2))
        output = self.activation_(output)
        # print("output layer :", output.shape)

        if return_intermediate is None:
            return output

        intermediate = NotImplementedError

        return output, intermediate

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer - 1)

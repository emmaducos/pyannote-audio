# The MIT License (MIT)
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHOR
# Hervé Bredin - http://herve.niderb.fr

import warnings
from typing import Optional
from typing import Union
from typing import Text
from pathlib import Path

import torch
import numpy as np

from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature

from pyannote.audio.train.model import RESOLUTION_FRAME
from pyannote.audio.train.model import RESOLUTION_CHUNK

from pyannote.audio.augmentation import Augmentation
# TODO: dcase
# name of the mother class
from pyannote.audio.features import Pretrained

from pyannote.audio.applications.config import load_config
from pyannote.audio.applications.config import load_specs
from pyannote.audio.applications.config import load_params

from pyannote.database import get_unique_identifier


class Dcase(Pretrained):
    """

    Parameters
    ----------
    validate_dir : Path
        Path to a validation directory.
    epoch : int, optional
        If provided, force loading this epoch.
        Defaults to reading epoch in validate_dir/params.yml.
    augmentation : Augmentation, optional
    duration : float, optional
        Use audio chunks with that duration. Defaults to the fixed duration
        used during training, when available.
    step : float, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.25.
    device : optional
    return_intermediate : optional
    """

    # TODO: add progress bar (at least for demo purposes)

    def __init__(self, validate_dir: Path = None,
                 epoch: int = None,
                 augmentation: Optional[Augmentation] = None,
                 duration: float = None,
                 step: float = None,
                 batch_size: int = 32,
                 device: Optional[Union[Text, torch.device]] = None,
                 return_intermediate=None,
                 progress_hook=None):

        try:
            validate_dir = Path(validate_dir)
        except TypeError as e:
            msg = (
                f'"validate_dir" must be str, bytes or os.PathLike object, '
                f'not {type(validate_dir).__name__}.'
            )
            raise TypeError(msg)

        strict = epoch is None
        self.validate_dir = validate_dir.expanduser().resolve(strict=strict)

        train_dir = self.validate_dir.parents[1]
        root_dir = train_dir.parents[1]

        config_yml = root_dir / 'config.yml'
        config = load_config(config_yml, training=False)

        # use feature extraction from config.yml configuration file
        self.feature_extraction_ = config['feature_extraction']

        # TODO: dcase
        # to accomodate with the mother class
        super().__init__(validate_dir=validate_dir,
                         epoch=epoch,
                         duration=duration,
                         augmentation=augmentation,
                         step=step,
                         batch_size=batch_size,
                         device=device,
                         return_intermediate=return_intermediate,
                         progress_hook=progress_hook)

        self.feature_extraction_.augmentation = self.augmentation

        specs_yml = train_dir / 'specs.yml'
        specifications = load_specs(specs_yml)

        if epoch is None:
            params_yml = self.validate_dir / 'params.yml'
            params = load_params(params_yml)
            self.epoch_ = params['epoch']

            # keep track of pipeline parameters
            self.pipeline_params_ = params.get('params', {})

            labels_spec_ = params.get('labels_spec', {})
            if labels_spec_:
                self.labels_spec_ = labels_spec_

            # Handle the case of the multilabel task
            label_list = getattr(config['task'], 'label_names', [])
            for label in label_list:
                if label in params:
                    setattr(self, "{}_pipeline_params_".format(label), params[label]['params'])
        else:
            self.epoch_ = epoch

        self.preprocessors_ = config['preprocessors']

        self.weights_pt_ = train_dir / 'weights' / f'{self.epoch_:04d}.pt'

        model = config['get_model_from_specs'](specifications)
        model.load_state_dict(
            torch.load(self.weights_pt_,
                       map_location=lambda storage, loc: storage))

        # defaults to using GPU when available
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # send model to device
        self.model_ = model.eval().to(self.device)

        # initialize chunks duration with that used during training
        self.duration = getattr(config['task'], 'duration', None)

        # override chunks duration by user-provided value
        if duration is not None:
            # warn that this might be sub-optimal
            if self.duration is not None and duration != self.duration:
                msg = (
                    f'Model was trained with {self.duration:g}s chunks and '
                    f'is applied on {duration:g}s chunks. This might lead '
                    f'to sub-optimal results.'
                )
                warnings.warn(msg)
            # do it anyway
            self.duration = duration

        if step is None:
            step = 0.25
        self.step = step
        self.chunks_ = SlidingWindow(duration=self.duration,
                                     step=self.step * self.duration)

        self.batch_size = batch_size

        self.return_intermediate = return_intermediate
        self.progress_hook = progress_hook

    def __call__(self, current_file):
        """Extract features from file
        #TODO: dcase
        from `features/base.py`
        to get the new format of `current_file` and still extract the short term features

        Parameters
        ----------
        current_file : dict
            `pyannote.database` files.

        Returns
        -------
        features : `pyannote.core.SlidingWindowFeature`
            Extracted features
        """

        # load waveform, re-sample, convert to mono, augment, normalize
        y, sample_rate = self.raw_audio_(current_file, return_sr=True)

        # compute features
        features = self.get_features(y.data, sample_rate, current_file)
        # print("features.pretrained.__call__ features :", type(features))

        # basic quality check
        if np.any(np.isnan(features)):
            uri = get_unique_identifier(current_file)
            msg = f'Features extracted from "{uri}" contain NaNs.'
            warnings.warn(msg.format(uri=uri))

        features = SlidingWindowFeature(features,
                                        self.feature_extraction_.sliding_window)
        return features

    def get_features(self, y, sample_rate, current_file) -> np.ndarray:
        """
        #TODO: dcase
        the main function to get the short term features to be in a dictionnary format
        with its corresponding long term features
        """

        features = {'SlidingWindowFeature': SlidingWindowFeature(self.feature_extraction_.get_features(y, sample_rate),
                                                                 self.feature_extraction_.sliding_window),

                    'logavgmel': current_file['logavgmel']}

        result = self.model_.slide(features,
                                   self.chunks_,
                                   batch_size=self.batch_size,
                                   device=self.device,
                                   return_intermediate=self.return_intermediate,
                                   progress_hook=self.progress_hook).data

        return result

    def get_context_duration(self) -> float:
        # FIXME: add half window duration to context?
        return self.feature_extraction_.get_context_duration()

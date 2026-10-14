"""
Audio effects for data augmentation (the processors used by configs/data/*.yaml).

Several audio effects can be combined into an augmentation chain.

Section 2, TL21
AI Speech and Sound Group, SL1
"""

import functools
from itertools import permutations
from random import shuffle
from typing import List, Optional, Tuple, Union

import numpy as np
import pymixconsole as pymc
from numba import jit
from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList
from pymixconsole.processor import Processor
from scipy.signal import butter, cheby1, ellip, oaconvolve, sosfilt


def convert_audio2data(x):
    """
    Convert audio data from the format it was stored in to float32.

    Args:
        x (Numpy array): input with `x.dtype` either `np.int16`, `np.int32`, `np.float32` or `np.float64`.

    Returns:
        Numpy array: output with values in [-1., 1.) where `dtype` is `np.float32`.
    """
    if x.dtype in [np.float32, np.float64]:
        return x.astype(dtype=np.float32)
    else:
        return (x.astype(dtype=np.float64) / (1.0 + np.iinfo(x.dtype).max)).astype(
            np.float32
        )


def sample_data(
    data: Tuple[int, Union[np.ndarray, functools.partial]],
    start: int = 0,
    length: Optional[int] = None,
) -> np.ndarray:
    """
    Load one stem specified by `data`.

    Returns the audio beginning from `start` either up to the end (if `length` is None)
    or until the provided `length`. For the case that `start + length > n_samples`, we do a wrap-around and
    load the remaining samples from the beginning of `data`.

    Args:
        data: Data with shape (n_samples, data).
        start: Start index.
        length: Length of sample. If `length` is not None, `length` samples are returned (possibly with a wrap-around).
            Otherwise, everything until the end of `data` is returned.

    Returns:
        samples: data with shape `n_samples x n_channels`
    """
    n_samples, audio = data

    # determine whether we have to load the audio or whether it was already loaded
    is_loaded = True if type(audio) is np.ndarray else False

    # if `length` is not None, then only select subset
    do_wrap_around = False
    if length is not None:
        if start + length > n_samples:
            # we need to wrap around and concatenate `start:` and `:stop`
            do_wrap_around = True
            stop = length - (n_samples - start)
        else:
            # no wrap around as it is inside the array/file boundaries
            stop = start + length
    else:
        stop = None

    if is_loaded:
        if not do_wrap_around:
            samples = convert_audio2data(audio[start:stop])
        else:
            samples = np.vstack(
                (convert_audio2data(audio[start:]), convert_audio2data(audio[:stop]))
            )
    else:
        if not do_wrap_around:
            samples = convert_audio2data(audio(start=start, stop=stop)[0])
        else:
            samples = np.vstack(
                (
                    convert_audio2data(audio(start=start)[0]),
                    convert_audio2data(audio(stop=stop)[0]),
                )
            )

    return samples


# Monkey-Patch `Processor` for convenience
# (a) Allow `None` as blocksize if processor can work on variable-length audio
def new_init(
    self,
    name,
    parameters,
    block_size,
    sample_rate=None,
    normalize=None,
    dtype="float32",
    **kwargs,
):
    """
    Initialize processor.

    Args:
        self: Reference to object
        name (str): Name of processor.
        parameters (parameter_list): Parameters for this processor.
        block_size (int): Size of blocks for blockwise processing.
            Can also be `None` if full audio can be processed at once.
        sample_rate (int): Sample rate of input audio. Use `None` if effect is independent of this value.
        normalize (str): Defines, whether the processed signal is normalized.
            Possible values are `'rms'`, `'max'` and `None`.
        dtype (str): data type of samples
        **kwargs: Additional keyword arguments (e.g., weight) are ignored for compatibility with Hydra configs.

    Raises:
        ValueError: If `normalize` is not equal to `'rms'`, `'max'` or `False`.
    """
    self.name = name
    self.parameters = parameters
    self.block_size = block_size
    self.sample_rate = sample_rate
    self.dtype = dtype
    if normalize not in [None, "rms", "max"]:
        raise ValueError(
            f"Unknown value {normalize} for `normalize`. Must be either `rms`, `max` or `None`"
        )
    self.normalize = normalize


# (b) make code simpler
def new_update(self, parameter_name):
    """
    Update processor after randomization of parameters.

    Args:
        self: Reference to object.
        parameter_name (str): Parameter whose value has changed.
    """
    pass


# (c) representation for nice print
def new_repr(self):
    """
    Create human-readable representation.

    Args:
        self: Reference to object.

    Returns:
        string representation of object.
    """
    return f"Processor(name={self.name!r}, parameters={self.parameters!r}"


Processor.__init__ = new_init
Processor.__repr__ = new_repr
Processor.update = new_update


class AugmentationChain:
    """Basic audio Fx chain which is used for data augmentation."""

    def __init__(
        self,
        fxs: Optional[List[Tuple[Union[Processor, "AugmentationChain"], float]]] = [],
        shuffle: Optional[bool] = False,
        apply_to: Optional[str] = "both",
        count_probs: Optional[List[float]] = None,
    ):
        """
        Create augmentation chain from the dictionary `fxs`.

        Args:
            fxs (list of tuples): Each tuple is `(effect, value)` where `effect` is a
                `pymc.processor` or `AugmentationChain`. In the legacy mode (`count_probs`
                is None) `value` is the independent probability the effect is applied. In
                count-based mode (`count_probs` set) `value` is the *selection weight*.
            shuffle (bool): If `True` the effect application order is randomized per call.
            apply_to (str): Apply the chain to `'both'`, `'input'` or `'target'`.
            count_probs (list of float, optional): Categorical distribution over the
                *number* of active effects per item (index = count). When given, the chain
                samples a count k ~ count_probs then picks k distinct effects by weight
                (without replacement), instead of rolling each effect independently. This
                yields sparse-but-controlled chains (mostly 1-2 effects). The list is
                truncated to the eligible-pool size and renormalized per call.

        Raises:
            ValueError: If `apply_to` is not equal to `both`, `input` or `target`.
        """
        self.fxs = fxs
        self.shuffle = shuffle
        self.count_probs = (
            None if count_probs is None else np.asarray(count_probs, dtype=float)
        )
        if apply_to not in ["both", "input", "target"]:
            raise ValueError(
                f"Unknown value {apply_to} for `apply_to`. Must be `both`, `input` or `target`"
            )
        else:
            self.apply_to = apply_to

    def apply_processor(self, x, processor: Processor):
        """
        Pass audio in `x` through `processor` and output the respective processed audio.

        Args:
            x (Numpy array): Input audio of shape `n_samples` x `n_channels`.
            processor (Processor): Audio effect that we want to apply.

        Returns:
            Numpy array: Processed audio of shape `n_samples` x `n_channels` (same size as `x')
        """
        n_samples_input = x.shape[0]

        if processor.block_size is None:
            y = processor.process(x)
        else:
            # make sure that n_samples is a multiple of `processor.block_size`
            if x.shape[0] % processor.block_size != 0:
                n_pad = processor.block_size - x.shape[0] % processor.block_size
                x = np.pad(x, ((0, n_pad), (0, 0)), mode="reflective")

            y = np.zeros_like(x)
            for idx in range(0, x.shape[0], processor.block_size):
                y[idx : idx + processor.block_size, :] = processor.process(
                    x[idx : idx + processor.block_size, :]
                )

        if processor.normalize is not None:
            if (
                processor.normalize == "rms"
            ):  # normalize output energy such that it is the same as the input energy
                scale = np.sqrt(
                    np.mean(np.square(x)) / np.maximum(1e-7, np.mean(np.square(y)))
                )
            elif (
                processor.normalize == "max"
            ):  # normalize output signal by its max. amplitude
                scale = (1 + 1e-7) / (np.max(np.abs(y)) + 1e-7)
            y *= scale

        # return audio of same length as x
        return y[:n_samples_input, :]

    def _apply_fx(self, fx, input_y, target_y):
        """Apply a single effect (or nested chain) respecting `self.apply_to`."""
        if isinstance(fx, Processor):
            # randomize all effect parameters (also calls `update()` for each processor)
            fx.randomize()
            if self.apply_to in ("both", "input"):
                input_y = self.apply_processor(input_y, fx)
            if self.apply_to in ("both", "target"):
                target_y = self.apply_processor(target_y, fx)
        else:
            input_y, target_y = fx(input_y, target_y)
        return input_y, target_y

    def __call__(self, input_x, target_x, mono=False):
        """
        Apply augmentation chain to audio in `input_x` and `target_x`.

        Args:
            input_x (Numpy array): Audio samples of shape `n_samples` x `n_channels`.
            target_x (Numpy array): Audio samples of shape `n_samples` x `n_channels`.
            mono (bool): If `True` the source is mono (or duplicated mono); effects tagged
                `stereo_only` are skipped so they can't act on a degenerate L==R signal.

        Returns:
            input_y, target_y (Numpy arrays): Processed audio, same shapes as the inputs.
        """
        input_y, target_y = input_x, target_x

        # ----- legacy mode: independent per-effect probabilities -----
        if self.count_probs is None:
            if self.shuffle:
                shuffle(self.fxs)
            for fx, p in self.fxs:
                if mono and getattr(fx, "stereo_only", False):
                    continue
                if np.random.rand() < p:
                    input_y, target_y = self._apply_fx(fx, input_y, target_y)
            return input_y, target_y

        # ----- count-based mode: sample k effects, then which (by weight) -----
        eligible = [
            (fx, w) for fx, w in self.fxs
            if not (mono and getattr(fx, "stereo_only", False))
        ]
        if not eligible:
            return input_y, target_y

        weights = np.array([max(float(w), 0.0) for _, w in eligible], dtype=float)
        n_usable = int(np.count_nonzero(weights))
        if n_usable == 0:
            weights = np.ones(len(eligible))
            n_usable = len(eligible)

        # count distribution truncated to what we can actually draw, then renormalized
        kmax = min(len(eligible), n_usable, len(self.count_probs) - 1)
        probs = np.array(self.count_probs[: kmax + 1], dtype=float)
        if probs.sum() <= 0:
            probs = np.ones(len(probs))
        probs /= probs.sum()
        k = int(np.random.choice(len(probs), p=probs))

        if k > 0:
            w = weights / weights.sum()
            idx = np.random.choice(len(eligible), size=k, replace=False, p=w)
            chosen = [eligible[i][0] for i in idx]
            if self.shuffle:
                shuffle(chosen)
            for fx in chosen:
                input_y, target_y = self._apply_fx(fx, input_y, target_y)

        return input_y, target_y

    def __repr__(self):
        """
        Human-readable representation.

        Returns:
            string representation of object.
        """
        return f"AugmentationChain(fxs={self.fxs!r}, shuffle={self.shuffle!r})"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DISTORTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Equaliser(Processor):
    """
    Five band parametric equaliser (two shelves and three central bands).

    All gains are set in dB values and range from `MIN_GAIN` dB to `MAX_GAIN` dB.
    This processor is implemented as cascade of five biquad IIR filters
    that are implemented using the infamous cookbook formulae from RBJ.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        low_shelf_gain (float), low_shelf_freq (float)
        first_band_gain (float), first_band_freq (float), first_band_q (float)
        second_band_gain (float), second_band_freq (float), second_band_q (float)
        third_band_gain (float), third_band_freq (float), third_band_q (float)

    original from https://github.com/csteinmetz1/pymixconsole/blob/master/pymixconsole/processors/equaliser.py
    """

    def __init__(
        self,
        n_channels,
        sample_rates,
        gain_range=(-15.0, 15.0),
        q_range=(0.1, 2.0),
        hard_clip=False,
        name="Equaliser",
        parameters=None,
        parameter_ranges=None,
        **kwargs,
    ):
        """
        Initialize processor.

        Args:
            n_channels (int): Number of audio channels.
            sample_rates (list of ints): Sample rates of audio.
            gain_range (tuple of floats): minimum and maximum gain that can be used.
            q_range (tuple of floats): minimum and maximum q value.
            hard_clip (bool): Whether we clip to [-1, 1.] after processing.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            parameter_ranges (dict): Optional dict to override parameter ranges.
                Format: {"param_name": {"min": value, "max": value}}
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name, parameters=parameters, block_size=None, **kwargs)

        self.n_channels = n_channels

        MIN_GAIN, MAX_GAIN = gain_range
        MIN_Q, MAX_Q = q_range

        # Apply parameter ranges if provided
        if parameter_ranges:
            if "gain_range" in parameter_ranges:
                MIN_GAIN = parameter_ranges["gain_range"].get("min", MIN_GAIN)
                MAX_GAIN = parameter_ranges["gain_range"].get("max", MAX_GAIN)
            if "q_range" in parameter_ranges:
                MIN_Q = parameter_ranges["q_range"].get("min", MIN_Q)
                MAX_Q = parameter_ranges["q_range"].get("max", MAX_Q)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(
                Parameter("sample_rate", 44100, "string", options=sample_rates)
            )
            # low shelf parameters -------
            low_shelf_freq_min, low_shelf_freq_max = 30.0, 200.0
            if parameter_ranges and "low_shelf_freq" in parameter_ranges:
                low_shelf_freq_min = parameter_ranges["low_shelf_freq"].get("min", low_shelf_freq_min)
                low_shelf_freq_max = parameter_ranges["low_shelf_freq"].get("max", low_shelf_freq_max)

            self.parameters.add(
                Parameter(
                    "low_shelf_gain", 0.0, "float", minimum=MIN_GAIN, maximum=MAX_GAIN
                )
            )
            self.parameters.add(
                Parameter("low_shelf_freq", 80.0, "float", minimum=low_shelf_freq_min, maximum=low_shelf_freq_max)
            )
            # first band parameters ------
            first_band_freq_min, first_band_freq_max = 200.0, 1000.0
            if parameter_ranges and "first_band_freq" in parameter_ranges:
                first_band_freq_min = parameter_ranges["first_band_freq"].get("min", first_band_freq_min)
                first_band_freq_max = parameter_ranges["first_band_freq"].get("max", first_band_freq_max)

            self.parameters.add(
                Parameter(
                    "first_band_gain", 0.0, "float", minimum=MIN_GAIN, maximum=MAX_GAIN
                )
            )
            self.parameters.add(
                Parameter(
                    "first_band_freq", 400.0, "float", minimum=first_band_freq_min, maximum=first_band_freq_max
                )
            )
            self.parameters.add(
                Parameter("first_band_q", 0.7, "float", minimum=MIN_Q, maximum=MAX_Q)
            )
            # second band parameters -----
            second_band_freq_min, second_band_freq_max = 1000.0, 3000.0
            if parameter_ranges and "second_band_freq" in parameter_ranges:
                second_band_freq_min = parameter_ranges["second_band_freq"].get("min", second_band_freq_min)
                second_band_freq_max = parameter_ranges["second_band_freq"].get("max", second_band_freq_max)

            self.parameters.add(
                Parameter(
                    "second_band_gain", 0.0, "float", minimum=MIN_GAIN, maximum=MAX_GAIN
                )
            )
            self.parameters.add(
                Parameter(
                    "second_band_freq", 2000.0, "float", minimum=second_band_freq_min, maximum=second_band_freq_max
                )
            )
            self.parameters.add(
                Parameter("second_band_q", 0.7, "float", minimum=MIN_Q, maximum=MAX_Q)
            )
            # third band parameters ------
            third_band_freq_min, third_band_freq_max = 3000.0, 8000.0
            if parameter_ranges and "third_band_freq" in parameter_ranges:
                third_band_freq_min = parameter_ranges["third_band_freq"].get("min", third_band_freq_min)
                third_band_freq_max = parameter_ranges["third_band_freq"].get("max", third_band_freq_max)

            self.parameters.add(
                Parameter(
                    "third_band_gain", 0.0, "float", minimum=MIN_GAIN, maximum=MAX_GAIN
                )
            )
            self.parameters.add(
                Parameter(
                    "third_band_freq", 4000.0, "float", minimum=third_band_freq_min, maximum=third_band_freq_max
                )
            )
            self.parameters.add(
                Parameter("third_band_q", 0.7, "float", minimum=MIN_Q, maximum=MAX_Q)
            )
            # high shelf parameters ------
            high_shelf_freq_min, high_shelf_freq_max = 5000.0, 10000.0
            if parameter_ranges and "high_shelf_freq" in parameter_ranges:
                high_shelf_freq_min = parameter_ranges["high_shelf_freq"].get("min", high_shelf_freq_min)
                high_shelf_freq_max = parameter_ranges["high_shelf_freq"].get("max", high_shelf_freq_max)

            self.parameters.add(
                Parameter(
                    "high_shelf_gain", 0.0, "float", minimum=MIN_GAIN, maximum=MAX_GAIN
                )
            )
            self.parameters.add(
                Parameter(
                    "high_shelf_freq", 8000.0, "float", minimum=high_shelf_freq_min, maximum=high_shelf_freq_max
                )
            )
        else:
            self.parameters = parameters

        self.bands = [
            "low_shelf",
            "first_band",
            "second_band",
            "third_band",
            "high_shelf",
        ]
        self.filters = self.setup_filters()
        self.hard_clip = hard_clip

    def setup_filters(self):
        """
        Create IIR filters.

        Returns:
            IIR filters
        """
        filters = {}

        for band in self.bands:
            G = getattr(self.parameters, band + "_gain").value
            fc = getattr(self.parameters, band + "_freq").value
            rate = self.parameters.sample_rate.value

            if band in ["low_shelf", "high_shelf"]:
                Q = 0.707
                filter_type = band
            else:
                Q = getattr(self.parameters, band + "_q").value
                filter_type = "peaking"

            filters[band] = pymc.components.iirfilter.IIRfilter(
                G, Q, fc, rate, filter_type, n_channels=self.n_channels
            )

        return filters

    def update_filter(self, band):
        """
        Update filters.

        Args:
            band (str): Band that should be updated.
        """
        self.filters[band].G = getattr(self.parameters, band + "_gain").value
        self.filters[band].fc = getattr(self.parameters, band + "_freq").value
        self.filters[band].rate = self.parameters.sample_rate.value

        if band in ["first_band", "second_band", "third_band"]:
            self.filters[band].Q = getattr(self.parameters, band + "_q").value

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        if parameter_name is not None:
            bands = ["_".join(parameter_name.split("_")[:2])]
        else:
            bands = self.bands

        for band in bands:
            self.update_filter(band)

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): equalized audio of size `n_samples x n_channels`.
        """
        # pymixconsole's `apply_filter` squeezes the DATA but not its filter
        # state, so a mono `(T, 1)` array becomes 1-D while `zi` stays
        # `(order, n_channels)` and scipy's lfilter raises "object too deep for
        # desired array". Squeezing here takes the library's own 1-D branch
        # (`zi[:, 0]`), which works whatever `n_channels` was configured with.
        # The tail of this method restores the channel axis. Stereo `(T, 2)`
        # input is untouched.
        squeezed_mono = x.ndim == 2 and x.shape[1] == 1
        if squeezed_mono:
            x = x[:, 0]

        for _band, iirfilter in self.filters.items():
            iirfilter.reset_state()
            x = iirfilter.apply_filter(x)

        if self.hard_clip:
            x = np.clip(x, -1.0, 1.0)

        # make sure that we have float32 as IIR filtering returns float64
        x = x.astype(np.float32)

        # make sure that we have two dimensions (if `n_channels == 1`)
        if x.ndim == 1:
            x = x[:, np.newaxis]

        return x


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPRESSOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@jit(nopython=True)
def compressor_process(
    x, threshold, attack_time, release_time, ratio, makeup_gain, sample_rate
):
    """
    Apply compressor.

    Args:
        x (Numpy array): audio data.
        threshold: threshold in dB.
        attack_time: attack_time in ms.
        release_time: release_time in ms.
        ratio: ratio.
        makeup_gain: makeup_gain.
        sample_rate: sample rate.

    Returns:
        compressed audio.
    """
    M = x.shape[0]
    x_g = np.zeros(M)
    x_l = np.zeros(M)
    y_g = np.zeros(M)
    y_l = np.zeros(M)
    c = np.zeros(M)
    yL_prev = 0.0

    alpha_attack = np.exp(-1 / (0.001 * sample_rate * attack_time))
    alpha_release = np.exp(-1 / (0.001 * sample_rate * release_time))

    for i in np.arange(M):
        if np.abs(x[i]) < 0.000001:
            x_g[i] = -120.0
        else:
            x_g[i] = 20 * np.log10(np.abs(x[i]))

        if x_g[i] >= threshold:
            y_g[i] = threshold + (x_g[i] - threshold) / ratio
        else:
            y_g[i] = x_g[i]

        x_l[i] = x_g[i] - y_g[i]

        if x_l[i] > yL_prev:
            y_l[i] = alpha_attack * yL_prev + (1 - alpha_attack) * x_l[i]
        else:
            y_l[i] = alpha_release * yL_prev + (1 - alpha_release) * x_l[i]

        c[i] = np.power(10.0, (makeup_gain - y_l[i]) / 20.0)
        yL_prev = y_l[i]

    y = x * c

    return y


class Compressor(Processor):
    """
    Single band stereo dynamic range compressor.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        threshold (float)
        attack_time (float)
        release_time (float)
        ratio (float)
        makeup_gain (float)
    """

    def __init__(self, sample_rates, name="Compressor", parameters=None, parameter_ranges=None, **kwargs):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            parameter_ranges (dict): Optional dict to override parameter ranges.
                Format: {"param_name": {"min": value, "max": value}}
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(
                Parameter("sample_rate", 44100, "string", options=sample_rates)
            )

            # Apply parameter ranges if provided
            threshold_min, threshold_max = -40.0, 0.0
            attack_min, attack_max = 0.03, 30.0
            release_min, release_max = 50.0, 100.0
            ratio_min, ratio_max = 2.0, 10.0
            makeup_min, makeup_max = -3.0, 6.0

            if parameter_ranges:
                if "threshold" in parameter_ranges:
                    threshold_min = parameter_ranges["threshold"].get("min", threshold_min)
                    threshold_max = parameter_ranges["threshold"].get("max", threshold_max)
                if "attack_time" in parameter_ranges:
                    attack_min = parameter_ranges["attack_time"].get("min", attack_min)
                    attack_max = parameter_ranges["attack_time"].get("max", attack_max)
                if "release_time" in parameter_ranges:
                    release_min = parameter_ranges["release_time"].get("min", release_min)
                    release_max = parameter_ranges["release_time"].get("max", release_max)
                if "ratio" in parameter_ranges:
                    ratio_min = parameter_ranges["ratio"].get("min", ratio_min)
                    ratio_max = parameter_ranges["ratio"].get("max", ratio_max)
                if "makeup_gain" in parameter_ranges:
                    makeup_min = parameter_ranges["makeup_gain"].get("min", makeup_min)
                    makeup_max = parameter_ranges["makeup_gain"].get("max", makeup_max)

            self.parameters.add(
                Parameter(
                    "threshold", 0.0, "float", units="dB", minimum=threshold_min, maximum=threshold_max
                )
            )
            self.parameters.add(
                Parameter(
                    "attack_time", 2.0, "float", units="ms", minimum=attack_min, maximum=attack_max
                )
            )
            self.parameters.add(
                Parameter(
                    "release_time",
                    50.0,
                    "float",
                    units="ms",
                    minimum=release_min,
                    maximum=release_max,
                )
            )
            self.parameters.add(
                Parameter("ratio", 2.0, "float", minimum=ratio_min, maximum=ratio_max)
            )
            self.parameters.add(
                Parameter(
                    "makeup_gain", 0.0, "float", units="dB", minimum=makeup_min, maximum=makeup_max
                )
            )
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): compressed audio of size `n_samples x n_channels`.
        """
        if not self.parameters.threshold.value == 0.0:
            y = np.zeros_like(x)

            for ch in range(x.shape[1]):
                y[:, ch] = compressor_process(
                    x[:, ch],
                    self.parameters.threshold.value,
                    self.parameters.attack_time.value,
                    self.parameters.release_time.value,
                    self.parameters.ratio.value,
                    self.parameters.makeup_gain.value,
                    self.parameters.sample_rate.value,
                )
        else:
            y = x

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%% CONVOLUTIONAL REVERB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ConvolutionalReverb(Processor):
    """
    Convolutional Reverb.

    Important: Due to convolving the audio sequence with some impulse response, we should ignore the
               first/last samples of the augmented audio sequence using `config['AUGMENTER_PADDING']`.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        wet_dry (float): Wet/dry ratio.
        decay (float): Applies a fade out to the impulse response.
        pre_delay (float): Value in ms. Shifts the IR in time and allows.
            A positive value produces a traditional delay between the dry signal and the wet.
            A negative delay is, in reality, zero delay, but effectively trims off the start of IR,
            so the reverb response begins at a point further in.
    """

    def __init__(
        self,
        impulse_responses,
        sample_rates,
        name="ConvolutionalReverb",
        parameters=None,
        **kwargs,
    ):
        """
        Initialize processor.

        Args:
            impulse_responses (list): List with impulse responses created by `common_dataprocessing.create_dataset`
            sample_rates (list of ints): Sample rates that we should assume (used for fade-out computation)
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.

        Raises:
            ValueError: if no impulse responses are provided.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if impulse_responses is None:
            raise ValueError(
                "List of impulse responses must be provided for ConvolutionalReverb processor."
            )
        self.impulse_responses = impulse_responses

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(
                Parameter("sample_rate", 44100, "string", options=sample_rates)
            )
            self.parameters.add(
                Parameter("index", 0, "int", minimum=0, maximum=len(impulse_responses))
            )
            self.parameters.add(
                Parameter("wet_dry", 1.0, "float", minimum=0.1, maximum=1.0)
            )
            self.parameters.add(
                Parameter("decay", 1.0, "float", minimum=0.1, maximum=1.0)
            )
            self.parameters.add(
                Parameter("pre_delay", 0, "int", units="ms", minimum=-100, maximum=100)
            )
        else:
            self.parameters = parameters

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        # copy IR from current index (to avoid modifying it in-place)
        self.h = np.copy(
            sample_data(
                self.impulse_responses.loc[self.parameters.index.value][
                    "impulse_response"
                ]
            )
        )

        # fade out the impulse based on the decay setting (starting from peak value) - constant 20ms fade-out
        if self.parameters.decay.value < 1.0:
            idx_peak = np.argmax(np.max(np.abs(self.h), axis=1), axis=0)
            fstart = np.minimum(
                self.h.shape[0],
                idx_peak
                + int(self.parameters.decay.value * (self.h.shape[0] - idx_peak)),
            )
            fstop = np.minimum(
                self.h.shape[0], fstart + int(0.020 * self.parameters.sample_rate.value)
            )
            flen = fstop - fstart

            fade = np.arange(1, flen + 1, dtype=self.dtype) / flen
            fade = np.power(0.1, fade * 5)
            self.h[fstart:fstop, :] *= fade[:, np.newaxis]
            self.h = self.h[:fstop]

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): reverbed audio of size `n_samples x n_channels`.
        """
        # reshape IR to the correct size
        n_channels = x.shape[1]
        if self.h.shape[1] == 1 and n_channels > 1:
            self.h = np.hstack(
                [self.h] * n_channels
            )  # repeat mono IR for multi-channel input
        if self.h.shape[1] > 1 and n_channels == 1:
            self.h = self.h[
                :, np.random.randint(self.h.shape[1]), np.newaxis
            ]  # randomly choose one IR channel

        if self.parameters.wet_dry.value == 0.0:
            return x
        else:
            # perform convolution to get wet signal
            y = oaconvolve(x, self.h, mode="full", axes=0)

            # cut out wet signal (compensating for the delay that the IR is introducing + predelay)
            idx = np.argmax(np.max(np.abs(self.h), axis=1), axis=0)
            idx += int(
                0.001
                * self.parameters.pre_delay.value
                * self.parameters.sample_rate.value
            )

            idx = np.clip(idx, 0, self.h.shape[0] - 1)

            y = y[idx : idx + x.shape[0], :]

            # return weighted sum of dry and wet signal
            return (
                1.0 - self.parameters.wet_dry.value
            ) * x + self.parameters.wet_dry.value * y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%% HAAS EFFECT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def haas_process(x, delay, feedback, wet_channel):
    """
    Add Haas effect to audio.

    Args:
        x (Numpy array): input audio.
        delay: Delay that we apply to one of the channels (in samples).
        feedback: Feedback value.
        wet_channel: Which channel we process (`left` or `right`).

    Returns:
        (Numpy array): Audio with Haas effect.
    """
    y = np.copy(x)
    if wet_channel == "left":
        y[:, 0] += feedback * np.roll(x[:, 0], delay)
    elif wet_channel == "right":
        y[:, 1] += feedback * np.roll(x[:, 1], delay)

    return y


class Haas(Processor):
    """
    Haas Effect Processor.

    Randomly selects one channel and applies a short delay to it.

    Important: This audio effect uses `np.roll` to perform the shift of one channel. Hence, you should use
               `config['AUGMENTER_PADDING']` to ignore the first samples of the augmented audio sequence.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        delay (float)
        feedback (float)
        wet_channel (string)
    """

    def __init__(
        self,
        sample_rates,
        delay_range=(-0.040, 0.040),
        name="Haas",
        parameters=None,
        **kwargs,
    ):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            delay_range (tuple of floats): minimum/maximum delay in milliseconds for Haas effect.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(
                Parameter("sample_rate", 44100, "string", options=sample_rates)
            )
            self.parameters.add(
                Parameter(
                    "delay",
                    delay_range[1],
                    "float",
                    units="ms",
                    minimum=delay_range[0],
                    maximum=delay_range[1],
                )
            )
            self.parameters.add(
                Parameter("feedback", 0.35, "float", minimum=0.33, maximum=0.66)
            )
            self.parameters.add(
                Parameter("wet_channel", "left", "string", options=["left", "right"])
            )
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): audio with Haas effect of size `n_samples x n_channels`.
        """
        assert x.shape[1] == 1 or x.shape[1] == 2, (
            "Haas effect only works with monaural or stereo audio."
        )

        if x.shape[1] < 2:
            x = np.repeat(x, 2, axis=1)

        y = haas_process(
            x,
            int(self.parameters.delay.value * self.parameters.sample_rate.value),
            self.parameters.feedback.value,
            self.parameters.wet_channel.value,
        )

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PANNER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Panner(Processor):
    """
    Simple stereo panner (adjusting amplitude and delay).

    If input is mono, output is stereo.
    Original edited from https://github.com/csteinmetz1/pymixconsole/blob/master/pymixconsole/processors/panner.py

    Important: This audio effect uses `np.roll` to perform the shift of one channel. Hence, you should use
               `config['AUGMENTER_PADDING']` to ignore the first samples of the augmented audio sequence.

    Processor parameters:
        sample_rate (int): Current (assumed) sample rate of the audio.
        pan (float): Panning angle. Can take values in [0, 1] where `0` corresponds to fully panned to left
            and `1` corresponds to fully panned to the right.
        pan_law (str): Pan law to be used for amplitude panning. Can be '-4.5dB', 'linear' or 'constant_power'.
        pan_mode (str): Scheme that is used for panning. Can be 'amplitude', 'delay' or 'both'.
        pan_maxdelay (float): Maximum delay if we pan a source fully to the left/right.
            For example `2. / 343.` is the maximum delay if the microphones are 2 meters apart.
    """

    def __init__(
        self,
        sample_rates,
        maxdelay_range=(0, 2.0 / 343.0),
        name="Panner",
        parameters=None,
        **kwargs,
    ):
        """
        Initialize processor.

        Args:
            sample_rates (list of ints): Sample rates of input audio.
            maxdelay_range (tuple of floats): minimum/maximum delay for panning effect. `2. / 343.` corresponds to
                the maximum delay if we have a microphone array where the microphones are 2 meters apart.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        # default processor class constructor
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(
                Parameter("sample_rate", 44100, "string", options=sample_rates)
            )
            self.parameters.add(
                Parameter("pan", 0.5, "float", minimum=0.1, maximum=0.9)
            )
            self.parameters.add(
                Parameter(
                    "pan_law",
                    "-4.5dB",
                    "string",
                    options=["-4.5dB", "linear", "constant_power"],
                )
            )
            self.parameters.add(
                Parameter(
                    "pan_mode",
                    "amplitude",
                    "string",
                    options=["amplitude", "delay", "both"],
                )
            )
            self.parameters.add(
                Parameter(
                    "pan_maxdelay",
                    (maxdelay_range[0] + maxdelay_range[1]) / 2.0,
                    "float",
                    units="ms",
                    minimum=maxdelay_range[0],
                    maximum=maxdelay_range[1],
                )
            )
        else:
            self.parameters = parameters

        # setup the coefficents based on default params
        self.update()

    def _calculate_pan_coefficents(self):
        """
        Calculate panning coefficients from the chosen pan law.

        Based on the set pan law determine the gain value
        to apply for the left and right channel to achieve panning effect.
        This operates on the assumption that the input channel is mono.
        The output data will be stereo at the moment, but could be expanded
        to a higher channel count format.
        The panning value is in the range [0, 1], where
        0 means the signal is panned completely to the left, and
        1 means the signal is apanned copletely to the right.

        Raises:
            ValueError: `self.parameters.pan_law` is not supported.
        """
        self.gains = np.zeros(2, dtype=self.dtype)

        # first scale the linear [0, 1] to [0, pi/2]
        theta = self.parameters.pan.value * (np.pi / 2)

        if self.parameters.pan_law.value == "linear":
            self.gains[0] = ((np.pi / 2) - theta) * (2 / np.pi)
            self.gains[1] = theta * (2 / np.pi)
        elif self.parameters.pan_law.value == "constant_power":
            self.gains[0] = np.cos(theta)
            self.gains[1] = np.sin(theta)
        elif self.parameters.pan_law.value == "-4.5dB":
            self.gains[0] = np.sqrt(((np.pi / 2) - theta) * (2 / np.pi) * np.cos(theta))
            self.gains[1] = np.sqrt(theta * (2 / np.pi) * np.sin(theta))
        else:
            raise ValueError(f"Invalid pan_law {self.parameters.pan_law.value}.")

    def _calculate_pan_delay(self):
        """Calculate delay for the chosen pan angle."""
        self.shifts = np.zeros(2, dtype=np.int32)

        # compute overall shift that we need between the two channels
        pan_maxdelay_samples = (
            self.parameters.pan_maxdelay.value * self.parameters.sample_rate.value
        )
        shift = 2 * int(pan_maxdelay_samples * np.abs(self.parameters.pan.value - 0.5))

        if self.parameters.pan.value < 0.5:
            # panning to the left
            self.shifts[0] = -shift // 2
            self.shifts[1] = shift - shift // 2
        else:
            self.shifts[0] = shift - shift // 2
            self.shifts[1] = -shift // 2

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): panned audio of size `n_samples x n_channels`.
        """
        assert x.shape[1] == 1 or x.shape[1] == 2, (
            "Panner only works with monaural or stereo audio."
        )

        # convert to stereo if signal is monaural
        if x.shape[1] < 2:
            x = np.repeat(x, 2, axis=1)

        y = np.copy(x)
        if self.parameters.pan_mode.value in ["delay", "both"]:
            y[:, 0] = np.roll(y[:, 0], self.shifts[0])
            y[:, 1] = np.roll(y[:, 1], self.shifts[1])

        if self.parameters.pan_mode.value in ["amplitude", "both"]:
            y *= self.gains

        return y

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        self._calculate_pan_coefficents()
        self._calculate_pan_delay()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class SwapChannels(Processor):
    """
    Swap channels in multi-channel audio.

    Processor parameters:
        index (int) Selects the permutation that we are using.
            Please note that "no permutation" is one of the permutations in `self.permutations` at index `0`.
            Hence, this effect should be applied with probability "1.".
    """

    def __init__(self, n_channels, name="SwapChannels", parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            n_channels (int): Number of channels in audio that we want to process.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        self.permutations = tuple(permutations(range(n_channels), n_channels))

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(
                Parameter("index", 0, "int", minimum=0, maximum=len(self.permutations))
            )
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): channel-swapped audio of size `n_samples x n_channels`.
        """
        return x[:, self.permutations[self.parameters.index.value]]


# %%%%%%%%%%%%%%%%%%%%%%% Monauralize %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Monauralize(Processor):
    """
    Monauralizes audio (i.e., removes spatial information).

    Process parameters:
        seed_channel (int): channel that we use for overwriting the others.
            `-1` refers to using the mean over all channels.
    """

    def __init__(self, n_channels, name="Monauralize", parameters=None, **kwargs):
        """
        Initialize processor.

        Args:
            n_channels (int): Number of channels in audio that we want to process.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
            **kwargs (dict): Keyword arguments that are passed to the Processor class.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(
                Parameter("seed_channel", 0, "int", minimum=-1, maximum=n_channels)
            )
        else:
            self.parameters = parameters

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): monauralized audio of size `n_samples x n_channels`.
        """
        tile_reps = (1, x.shape[1])
        if self.parameters.seed_channel.value == -1:
            # use average as seed
            return np.tile(np.mean(x, axis=1, keepdims=True), tile_reps)
        else:
            # use one channel as seed
            return np.tile(x[:, [self.parameters.seed_channel.value]], tile_reps)


# %%%%%%%%%%%%%%%%%%%%%%% ChunkShuffle %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _pr(parameter_ranges, key, dmin, dmax):
    """Return (min, max) for `key`, overridden by `parameter_ranges` if present."""
    if parameter_ranges and key in parameter_ranges:
        pr = parameter_ranges[key]
        return pr.get("min", dmin), pr.get("max", dmax)
    return dmin, dmax


def _opts(parameter_ranges, key, default_options):
    """Return discrete `options` for `key`, overridden by `parameter_ranges`."""
    if parameter_ranges and key in parameter_ranges and "options" in parameter_ranges[key]:
        return list(parameter_ranges[key]["options"])
    return list(default_options)


def _design_sos(btype, wn, order, ftype):
    """Design a stable second-order-sections filter. `wn` normalized to Nyquist."""
    if ftype == "ellip":
        return ellip(order, 0.5, 60.0, wn, btype=btype, output="sos")
    if ftype == "cheby1":
        return cheby1(order, 0.5, wn, btype=btype, output="sos")
    return butter(order, wn, btype=btype, output="sos")


class _FilterBase(Processor):
    """Shared init for the steep IIR filters (lowpass/highpass)."""

    btype = "lowpass"

    def __init__(self, sample_rates, cutoff_default, name, parameters=None,
                 parameter_ranges=None, **kwargs):
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)
        if parameters:
            self.parameters = parameters
            return
        cmin, cmax = _pr(parameter_ranges, "cutoff", *cutoff_default)
        order_opts = _opts(parameter_ranges, "order", [4, 6, 8, 10, 12])
        ftype_opts = _opts(parameter_ranges, "ftype", ["butter", "ellip", "cheby1"])
        self.parameters = ParameterList()
        self.parameters.add(Parameter("sample_rate", sample_rates[0], "string", options=sample_rates))
        self.parameters.add(
            Parameter("log_cutoff", float(np.log10(np.sqrt(cmin * cmax))), "float",
                      minimum=float(np.log10(cmin)), maximum=float(np.log10(cmax)))
        )
        self.parameters.add(Parameter("order", order_opts[0], "string", options=order_opts))
        self.parameters.add(Parameter("ftype", ftype_opts[0], "string", options=ftype_opts))

    def process(self, x):
        fs = float(self.parameters.sample_rate.value)
        nyq = 0.5 * fs
        cutoff = 10.0 ** self.parameters.log_cutoff.value
        wn = float(np.clip(cutoff / nyq, 1e-3, 0.99))
        sos = _design_sos(self.btype, wn, int(self.parameters.order.value),
                          self.parameters.ftype.value)
        return sosfilt(sos, x, axis=0).astype(np.float32)


class SteepLowpass(_FilterBase):
    """Steep low-pass (bandwidth limiting / BWE degradation)."""

    btype = "lowpass"

    def __init__(self, sample_rates, name="SteepLowpass", parameters=None,
                 parameter_ranges=None, **kwargs):
        super().__init__(sample_rates, (500.0, 16000.0), name, parameters,
                         parameter_ranges, **kwargs)


class SteepHighpass(_FilterBase):
    """Steep high-pass (rumble / subsonic / DC removal)."""

    btype = "highpass"

    def __init__(self, sample_rates, name="SteepHighpass", parameters=None,
                 parameter_ranges=None, **kwargs):
        super().__init__(sample_rates, (20.0, 500.0), name, parameters,
                         parameter_ranges, **kwargs)


class Bandpass(Processor):
    """Band-pass (telephone / speech-band limiting). low/high log-uniform in Hz."""

    def __init__(self, sample_rates, name="Bandpass", parameters=None,
                 parameter_ranges=None, **kwargs):
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)
        if parameters:
            self.parameters = parameters
            return
        lo_min, lo_max = _pr(parameter_ranges, "low", 100.0, 800.0)
        hi_min, hi_max = _pr(parameter_ranges, "high", 2000.0, 7000.0)
        order_opts = _opts(parameter_ranges, "order", [2, 4, 6, 8])
        ftype_opts = _opts(parameter_ranges, "ftype", ["butter", "ellip", "cheby1"])
        self.parameters = ParameterList()
        self.parameters.add(Parameter("sample_rate", sample_rates[0], "string", options=sample_rates))
        self.parameters.add(Parameter("log_low", float(np.log10(np.sqrt(lo_min * lo_max))),
                                      "float", minimum=float(np.log10(lo_min)),
                                      maximum=float(np.log10(lo_max))))
        self.parameters.add(Parameter("log_high", float(np.log10(np.sqrt(hi_min * hi_max))),
                                      "float", minimum=float(np.log10(hi_min)),
                                      maximum=float(np.log10(hi_max))))
        self.parameters.add(Parameter("order", order_opts[0], "string", options=order_opts))
        self.parameters.add(Parameter("ftype", ftype_opts[0], "string", options=ftype_opts))

    def process(self, x):
        fs = float(self.parameters.sample_rate.value)
        nyq = 0.5 * fs
        lo = 10.0 ** self.parameters.log_low.value
        hi = 10.0 ** self.parameters.log_high.value
        lo, hi = min(lo, hi), max(lo, hi)
        wlo = float(np.clip(lo / nyq, 1e-3, 0.98))
        whi = float(np.clip(hi / nyq, wlo + 1e-3, 0.99))
        sos = _design_sos("bandpass", [wlo, whi], int(self.parameters.order.value),
                          self.parameters.ftype.value)
        return sosfilt(sos, x, axis=0).astype(np.float32)


class WidthReduction(Processor):
    """Reduce stereo width by scaling the M/S Side channel (1->unchanged, 0->mono)."""

    stereo_only = True

    def __init__(self, name="WidthReduction", parameters=None, parameter_ranges=None, **kwargs):
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)
        if parameters:
            self.parameters = parameters
            return
        smin, smax = _pr(parameter_ranges, "side_gain", 0.0, 1.0)
        self.parameters = ParameterList()
        self.parameters.add(Parameter("side_gain", 0.5, "float", minimum=smin, maximum=smax))

    def process(self, x):
        if x.shape[1] < 2:
            return x
        g = float(self.parameters.side_gain.value)
        L, R = x[:, 0], x[:, 1]
        M = 0.5 * (L + R)
        S = 0.5 * (L - R) * g
        out = np.empty_like(x)
        out[:, 0] = M + S
        out[:, 1] = M - S
        return out.astype(np.float32)


class BalanceShift(Processor):
    """L/R level imbalance (pan/balance restoration)."""

    stereo_only = True

    def __init__(self, name="BalanceShift", parameters=None, parameter_ranges=None, **kwargs):
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)
        if parameters:
            self.parameters = parameters
            return
        gmin, gmax = _pr(parameter_ranges, "lr_gain_db", -12.0, 12.0)
        self.parameters = ParameterList()
        self.parameters.add(Parameter("lr_gain_db", 0.0, "float", units="dB",
                                      minimum=gmin, maximum=gmax))

    def process(self, x):
        if x.shape[1] < 2:
            return x
        half = 0.5 * float(self.parameters.lr_gain_db.value)
        gl = 10.0 ** (half / 20.0)
        gr = 10.0 ** (-half / 20.0)
        out = x.copy()
        out[:, 0] *= gl
        out[:, 1] *= gr
        return out.astype(np.float32)


class ChannelDropout(Processor):
    """Zero one stereo channel (dual-mono / dead-channel recovery)."""

    stereo_only = True

    def __init__(self, name="ChannelDropout", parameters=None, parameter_ranges=None, **kwargs):
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)
        if parameters:
            self.parameters = parameters
            return
        which_opts = _opts(parameter_ranges, "which", ["left", "right"])
        self.parameters = ParameterList()
        self.parameters.add(Parameter("which", which_opts[0], "string", options=which_opts))

    def process(self, x):
        if x.shape[1] < 2:
            return x
        out = x.copy()
        out[:, 0 if self.parameters.which.value == "left" else 1] = 0.0
        return out.astype(np.float32)


class Tremolo(Processor):
    """Shallow amplitude modulation (LFO gain). Same gain on all channels.

    Kept shallow because at SAME-L's 11 Hz frame rate slow/deep modulation drifts
    across frames; small depth keeps the per-frame operator near its mean.
    """

    def __init__(self, sample_rates, name="Tremolo", parameters=None, parameter_ranges=None, **kwargs):
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)
        if parameters:
            self.parameters = parameters
            return
        rmin, rmax = _pr(parameter_ranges, "rate_hz", 4.0, 14.0)
        dmin, dmax = _pr(parameter_ranges, "depth", 0.05, 0.4)
        self.parameters = ParameterList()
        self.parameters.add(Parameter("sample_rate", sample_rates[0], "string", options=sample_rates))
        self.parameters.add(Parameter("rate_hz", 8.0, "float", minimum=rmin, maximum=rmax))
        self.parameters.add(Parameter("depth", 0.2, "float", minimum=dmin, maximum=dmax))

    def process(self, x):
        n = x.shape[0]
        fs = float(self.parameters.sample_rate.value)
        rate = float(self.parameters.rate_hz.value)
        depth = float(self.parameters.depth.value)
        t = np.arange(n) / fs
        phase = np.random.uniform(0, 2 * np.pi)
        m = 1.0 - 0.5 * depth + 0.5 * depth * np.sin(2.0 * np.pi * rate * t + phase)
        return (x * m[:, None]).astype(np.float32)


Processor.stereo_only = False
for _stereo_cls in (Monauralize, SwapChannels, Haas, Panner):
    _stereo_cls.stereo_only = True



"""Pitch shifting as a chain `Processor`, backed by audiomentations.

Uses audiomentations' `PitchShift` (signalsmith_stretch backend, length
preserving) instead of the vendored SoX-based one, which needs an
unmaintained dependency. Wrapped as a `Processor` so it takes part in the
chain's own firing/ordering and normalization logic.

Each slot is shifted independently; keep `weight` and the semitone range
small so a shifted source still sounds like that instrument.
"""

from __future__ import annotations

import numpy as np
from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList
from pymixconsole.processor import Processor


class PitchShift(Processor):
    """Shift pitch by a random number of semitones, preserving length.

    Processor parameters:
        steps: shift in semitones, drawn per call from
            `[min_semitones, max_semitones]`.
    """

    def __init__(
        self,
        sample_rates=(44100, 48000),
        min_semitones: float = -2.0,
        max_semitones: float = 2.0,
        method: str = "signalsmith_stretch",
        name: str = "PitchShift",
        parameters=None,
        **kwargs,
    ):
        super().__init__(name=name, parameters=parameters, block_size=None, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(
                Parameter("sample_rate", int(sample_rates[0]), "string",
                          options=[int(s) for s in sample_rates])
            )
            self.parameters.add(
                Parameter("steps", 0.0, "float",
                          minimum=float(min_semitones), maximum=float(max_semitones))
            )
        else:
            self.parameters = parameters

        self.method = method
        self._shifter = None

    def _backend(self, semitones: float):
        from audiomentations import PitchShift as _AMPitchShift

        # p=1.0: the chain already decides whether this effect fires at all, so
        # a second internal probability would silently halve the configured
        # weight and make `count_probs` mean something other than it says.
        t = _AMPitchShift(min_semitones=semitones, max_semitones=semitones,
                          method=self.method, p=1.0)
        return t

    def process(self, x):
        """[n_samples, n_channels] -> same shape, pitch-shifted."""
        steps = float(self.parameters.steps.value)
        if abs(steps) < 1e-3:
            return x

        sr = int(self.parameters.sample_rate.value)
        # audiomentations wants [channels, samples]; the chain works in
        # [samples, channels].
        y = self._backend(steps)(
            samples=np.ascontiguousarray(x.T.astype("float32")), sample_rate=sr
        )
        y = np.ascontiguousarray(np.asarray(y, dtype="float32").T)

        # The backend is length-preserving, but do not depend on it: a length
        # change here would silently misalign the slot against the rest of the
        # item, and every consumer assumes the slots share a time axis.
        n = x.shape[0]
        if y.shape[0] > n:
            y = y[:n]
        elif y.shape[0] < n:
            y = np.pad(y, ((0, n - y.shape[0]), (0, 0)))
        return y

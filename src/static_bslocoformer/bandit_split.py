from typing import Callable, Optional

from abc import abstractmethod
import os

from einops import rearrange
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules import activation

from torchaudio import functional as taF
from torchaudio.functional.functional import _create_triangular_filterbank

from spafe.fbanks import bark_fbanks
from spafe.utils.converters import hz2bark, hz2erb

from librosa import hz_to_midi, midi_to_hz


def band_widths_from_specs(band_specs):
    return [e - i for i, e in band_specs]


def check_nonzero_bandwidth(band_specs):
    # pprint(band_specs)
    for fstart, fend in band_specs:
        if fend - fstart <= 0:
            raise ValueError("Bands cannot be zero-width")


def check_no_overlap(band_specs):
    fend_prev = -1
    for fstart_curr, fend_curr in band_specs:
        if fstart_curr <= fend_prev:
            raise ValueError("Bands cannot overlap")


def check_no_gap(band_specs):
    fstart, _ = band_specs[0]
    assert fstart == 0

    fend_prev = -1
    for fstart_curr, fend_curr in band_specs:
        if fstart_curr - fend_prev > 1:
            raise ValueError("Bands cannot leave gap")
        fend_prev = fend_curr


def get_band_specs(band_specs, n_fft, fs, n_bands=None):
    if "tribark" in band_specs:
        assert n_bands is not None
        specs = TriangularBarkBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif "bark" in band_specs:
        assert n_bands is not None
        specs = BarkBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif "erb" in band_specs:
        assert n_bands is not None
        specs = EquivalentRectangularBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif "musical" in band_specs:
        assert n_bands is not None
        specs = MusicalBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif band_specs == "dnr:mel" or "mel" in band_specs:
        assert n_bands is not None
        specs = MelBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif band_specs == "full":
        n_bins = n_fft // 2 + 1
        bsm = [(0, n_bins) for _ in range(n_bands)]
        freq_weights = [torch.ones(n_bins) / n_bands for _ in range(n_bins)]
        overlapping_band = True
    else:
        raise NameError
    return bsm, freq_weights, overlapping_band


class BandsplitSpecification:
    def __init__(self, nfft: int, fs: int) -> None:
        self.fs = fs
        self.nfft = nfft
        self.nyquist = fs / 2
        self.max_index = nfft // 2 + 1

        self.split500 = self.hertz_to_index(500)
        self.split1k = self.hertz_to_index(1000)
        self.split2k = self.hertz_to_index(2000)
        self.split4k = self.hertz_to_index(4000)
        self.split8k = self.hertz_to_index(8000)
        self.split16k = self.hertz_to_index(16000)
        self.split20k = self.hertz_to_index(20000)

        self.above20k = [(self.split20k, self.max_index)]
        self.above16k = [(self.split16k, self.split20k)] + self.above20k

    def index_to_hertz(self, index: int):
        return index * self.fs / self.nfft

    def hertz_to_index(self, hz: float, round: bool = True):
        index = hz * self.nfft / self.fs

        if round:
            index = int(np.round(index))

        return index

    def get_band_specs_with_bandwidth(self, start_index, end_index, bandwidth_hz):
        band_specs = []
        lower = start_index

        while lower < end_index:
            upper = int(np.floor(lower + self.hertz_to_index(bandwidth_hz)))
            upper = min(upper, end_index)

            band_specs.append((lower, upper))
            lower = upper

        return band_specs

    @abstractmethod
    def get_band_specs(self):
        raise NotImplementedError


class PerceptualBandsplitSpecification(BandsplitSpecification):
    def __init__(
        self,
        nfft: int,
        fs: int,
        fbank_fn: Callable[[int, int, float, float, int], torch.Tensor],
        n_bands: int,
        f_min: float = 0.0,
        f_max: float = None,
    ) -> None:
        super().__init__(nfft=nfft, fs=fs)
        self.n_bands = n_bands
        if f_max is None:
            f_max = fs / 2

        self.filterbank = fbank_fn(n_bands, fs, f_min, f_max, self.max_index)

        weight_per_bin = torch.sum(self.filterbank, dim=0, keepdim=True)  # (1, n_freqs)
        normalized_mel_fb = self.filterbank / weight_per_bin  # (n_mels, n_freqs)

        freq_weights = []
        band_specs = []
        for i in range(self.n_bands):
            active_bins = torch.nonzero(self.filterbank[i, :]).squeeze().tolist()
            if isinstance(active_bins, int):
                active_bins = (active_bins, active_bins)
            if len(active_bins) == 0:
                continue
            start_index = active_bins[0]
            end_index = active_bins[-1] + 1
            band_specs.append((start_index, end_index))
            freq_weights.append(normalized_mel_fb[i, start_index:end_index])

        self.freq_weights = freq_weights
        self.band_specs = band_specs

    def get_band_specs(self):
        return self.band_specs

    def get_freq_weights(self):
        return self.freq_weights

    def save_to_file(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)

        import pickle

        with open(os.path.join(dir_path, "mel_bandsplit_spec.pkl"), "wb") as f:
            pickle.dump(
                {
                    "band_specs": self.band_specs,
                    "freq_weights": self.freq_weights,
                    "filterbank": self.filterbank,
                },
                f,
            )


def mel_filterbank(n_bands, fs, f_min, f_max, n_freqs):
    fb = taF.melscale_fbanks(
        n_mels=n_bands,
        sample_rate=fs,
        f_min=f_min,
        f_max=f_max,
        n_freqs=n_freqs,
    ).T

    fb[0, 0] = 1.0

    return fb


class MelBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=mel_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


def musical_filterbank(n_bands, fs, f_min, f_max, n_freqs, scale="constant"):
    nfft = 2 * (n_freqs - 1)
    df = fs / nfft
    # init freqs
    f_max = f_max or fs / 2
    f_min = f_min or 0
    f_min = fs / nfft

    n_octaves = np.log2(f_max / f_min)
    n_octaves_per_band = n_octaves / n_bands
    bandwidth_mult = np.power(2.0, n_octaves_per_band)

    low_midi = max(0, hz_to_midi(f_min))
    high_midi = hz_to_midi(f_max)
    midi_points = np.linspace(low_midi, high_midi, n_bands)
    hz_pts = midi_to_hz(midi_points)

    low_pts = hz_pts / bandwidth_mult
    high_pts = hz_pts * bandwidth_mult

    low_bins = np.floor(low_pts / df).astype(int)
    high_bins = np.ceil(high_pts / df).astype(int)

    fb = np.zeros((n_bands, n_freqs))

    for i in range(n_bands):
        fb[i, low_bins[i] : high_bins[i] + 1] = 1.0

    fb[0, : low_bins[0]] = 1.0
    fb[-1, high_bins[-1] + 1 :] = 1.0

    return torch.as_tensor(fb)


class MusicalBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=musical_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


def bark_filterbank(n_bands, fs, f_min, f_max, n_freqs):
    nfft = 2 * (n_freqs - 1)
    fb, _ = bark_fbanks.bark_filter_banks(
        nfilts=n_bands, nfft=nfft, fs=fs, low_freq=f_min, high_freq=f_max, scale="constant"
    )

    return torch.as_tensor(fb)


class BarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=bark_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


def triangular_bark_filterbank(n_bands, fs, f_min, f_max, n_freqs):
    all_freqs = torch.linspace(0, fs // 2, n_freqs)

    # calculate mel freq bins
    m_min = hz2bark(f_min)
    m_max = hz2bark(f_max)

    m_pts = torch.linspace(m_min, m_max, n_bands + 2)
    f_pts = 600 * torch.sinh(m_pts / 6)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    fb = fb.T

    first_active_band = torch.nonzero(torch.sum(fb, dim=-1))[0, 0]
    first_active_bin = torch.nonzero(fb[first_active_band, :])[0, 0]

    fb[first_active_band, :first_active_bin] = 1.0

    return fb


class TriangularBarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(
            fbank_fn=triangular_bark_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max
        )


def minibark_filterbank(n_bands, fs, f_min, f_max, n_freqs):
    fb = bark_filterbank(n_bands, fs, f_min, f_max, n_freqs)

    fb[fb < np.sqrt(0.5)] = 0.0

    return fb


class MiniBarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=minibark_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


def erb_filterbank(
    n_bands: int,
    fs: int,
    f_min: float,
    f_max: float,
    n_freqs: int,
) -> Tensor:
    # freq bins
    A = (1000 * np.log(10)) / (24.7 * 4.37)
    all_freqs = torch.linspace(0, fs // 2, n_freqs)

    # calculate mel freq bins
    m_min = hz2erb(f_min)
    m_max = hz2erb(f_max)

    m_pts = torch.linspace(m_min, m_max, n_bands + 2)
    f_pts = (torch.pow(10, (m_pts / A)) - 1) / 0.00437

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    fb = fb.T

    first_active_band = torch.nonzero(torch.sum(fb, dim=-1))[0, 0]
    first_active_bin = torch.nonzero(fb[first_active_band, :])[0, 0]

    fb[first_active_band, :first_active_bin] = 1.0

    return fb


class EquivalentRectangularBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=erb_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


class NormFC(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        bandwidth: int,
        in_channel: int,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()

        self.treat_channel_as_feature = treat_channel_as_feature

        if normalize_channel_independently:
            raise NotImplementedError

        reim = 2

        self.norm = nn.LayerNorm(in_channel * bandwidth * reim)

        fc_in = bandwidth * reim

        if treat_channel_as_feature:
            fc_in *= in_channel
        else:
            assert emb_dim % in_channel == 0
            emb_dim = emb_dim // in_channel

        self.fc = nn.Linear(fc_in, emb_dim)

    def forward(self, xb):
        # xb = (batch, n_time, in_chan, reim * band_width)

        batch, n_time, in_chan, ribw = xb.shape
        xb = self.norm(xb.reshape(batch, n_time, in_chan * ribw))
        # (batch, n_time, in_chan * reim * band_width)

        if not self.treat_channel_as_feature:
            xb = xb.reshape(batch, n_time, in_chan, ribw)
            # (batch, n_time, in_chan, reim * band_width)

        zb = self.fc(xb)
        # (batch, n_time, emb_dim)
        # OR
        # (batch, n_time, in_chan, emb_dim_per_chan)

        if not self.treat_channel_as_feature:
            batch, n_time, in_chan, emb_dim_per_chan = zb.shape
            # (batch, n_time, in_chan, emb_dim_per_chan)
            zb = zb.reshape((batch, n_time, in_chan * emb_dim_per_chan))

        return zb  # (batch, n_time, emb_dim)


class BandSplitModule(nn.Module):
    def __init__(
        self,
        band_specs: list[tuple[float, float]],
        emb_dim: int,
        in_channel: int,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()

        check_nonzero_bandwidth(band_specs)

        if require_no_gap:
            check_no_gap(band_specs)

        if require_no_overlap:
            check_no_overlap(band_specs)

        self.band_specs = band_specs
        # list of [fstart, fend) in index.
        # Note that fend is exclusive.
        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)
        self.emb_dim = emb_dim

        self.norm_fc_modules = nn.ModuleList(
            [  # type: ignore
                (
                    NormFC(
                        emb_dim=emb_dim,
                        bandwidth=bw,
                        in_channel=in_channel,
                        normalize_channel_independently=normalize_channel_independently,
                        treat_channel_as_feature=treat_channel_as_feature,
                    )
                )
                for bw in self.band_widths
            ]
        )

    def forward(self, x: torch.Tensor):
        # x = complex spectrogram (batch, in_chan, n_freq, n_time)

        batch, in_chan, _, n_time = x.shape

        z = torch.zeros(size=(batch, self.n_bands, n_time, self.emb_dim), device=x.device)

        xr = torch.view_as_real(x)  # batch, in_chan, n_freq, n_time, 2
        xr = torch.permute(xr, (0, 3, 1, 4, 2))  # batch, n_time, in_chan, 2, n_freq
        batch, n_time, in_chan, reim, band_width = xr.shape
        for i, nfm in enumerate(self.norm_fc_modules):
            # print(f"bandsplit/band{i:02d}")
            fstart, fend = self.band_specs[i]
            xb = xr[..., fstart:fend]
            # (batch, n_time, in_chan, reim, band_width)
            xb = torch.reshape(xb, (batch, n_time, in_chan, -1))
            # (batch, n_time, in_chan, reim * band_width)
            # z.append(nfm(xb))  # (batch, n_time, emb_dim)
            z[:, i, :, :] = nfm(xb.contiguous())

        # z = torch.stack(z, dim=1)

        return z


class BaseNormMLP(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channel: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs=None,
        complex_mask: bool = True,
    ):
        super().__init__()
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}
        self.hidden_activation_kwargs = hidden_activation_kwargs
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = torch.jit.script(
            nn.Sequential(
                nn.Linear(in_features=emb_dim, out_features=mlp_dim),
                activation.__dict__[hidden_activation](**self.hidden_activation_kwargs),
            )
        )

        self.bandwidth = bandwidth
        self.in_channel = in_channel

        self.complex_mask = complex_mask
        self.reim = 2 if complex_mask else 1
        self.glu_mult = 2


class NormMLP(BaseNormMLP):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channel: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs=None,
        complex_mask: bool = True,
    ) -> None:
        super().__init__(
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            bandwidth=bandwidth,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
        )

        self.output = torch.jit.script(
            nn.Sequential(
                nn.Linear(
                    in_features=mlp_dim,
                    out_features=bandwidth * in_channel * self.reim * 2,
                ),
                nn.GLU(dim=-1),
            )
        )

    def reshape_output(self, mb):
        # print(mb.shape)
        batch, n_time, _ = mb.shape
        if self.complex_mask:
            mb = mb.reshape(batch, n_time, self.in_channel, self.bandwidth, self.reim).contiguous()
            # print(mb.shape)
            # NOTE: torch.view_as_complex does not support bfloat16, which is why we need to convert mb to float32
            mb = torch.view_as_complex(mb.to(torch.float32))  # (batch, n_time, in_channel, bandwidth)
        else:
            mb = mb.reshape(batch, n_time, self.in_channel, self.bandwidth)

        mb = torch.permute(mb, (0, 2, 3, 1))  # (batch, in_channel, bandwidth, n_time)

        return mb

    def forward(self, qb):
        # qb = (batch, n_time, emb_dim)

        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("qb0")

        qb = self.norm(qb)  # (batch, n_time, emb_dim)

        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("qb1")

        qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("qb2")
        mb = self.output(qb)  # (batch, n_time, bandwidth * in_channel * reim)
        # if torch.any(torch.isnan(qb)):
        #     raise ValueError("mb")
        mb = self.reshape_output(mb)  # (batch, in_channel, bandwidth, n_time)

        return mb


class MultAddNormMLP(NormMLP):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channel: int | None,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs=None,
        complex_mask: bool = True,
    ) -> None:
        super().__init__(
            emb_dim, mlp_dim, bandwidth, in_channel, hidden_activation, hidden_activation_kwargs, complex_mask
        )

        self.output2 = torch.jit.script(
            nn.Sequential(
                nn.Linear(
                    in_features=mlp_dim,
                    out_features=bandwidth * in_channel * self.reim * 2,
                ),
                nn.GLU(dim=-1),
            )
        )

    def forward(self, qb):
        qb = self.norm(qb)  # (batch, n_time, emb_dim)
        qb = self.hidden(qb)  # (batch, n_time, mlp_dim)
        mmb = self.output(qb)  # (batch, n_time, bandwidth * in_channel * reim)
        mmb = self.reshape_output(mmb)  # (batch, in_channel, bandwidth, n_time)
        amb = self.output2(qb)  # (batch, n_time, bandwidth * in_channel * reim)
        amb = self.reshape_output(amb)  # (batch, in_channel, bandwidth, n_time)

        return mmb, amb


class MaskEstimationModuleSuperBase(nn.Module):
    pass


class MaskEstimationModuleBase(MaskEstimationModuleSuperBase):
    def __init__(
        self,
        band_specs: list[tuple[float, float]],
        emb_dim: int,
        mlp_dim: int,
        in_channel: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: dict = None,
        complex_mask: bool = True,
        norm_mlp_cls: type[nn.Module] = NormMLP,
        norm_mlp_kwargs: dict = None,
    ) -> None:
        super().__init__()

        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)

        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        if norm_mlp_kwargs is None:
            norm_mlp_kwargs = {}

        self.norm_mlp = nn.ModuleList(
            [
                (
                    norm_mlp_cls(
                        bandwidth=self.band_widths[b],
                        emb_dim=emb_dim,
                        mlp_dim=mlp_dim,
                        in_channel=in_channel,
                        hidden_activation=hidden_activation,
                        hidden_activation_kwargs=hidden_activation_kwargs,
                        complex_mask=complex_mask,
                        **norm_mlp_kwargs,
                    )
                )
                for b in range(self.n_bands)
            ]
        )

    def compute_masks(self, q):
        batch, n_bands, n_time, emb_dim = q.shape

        masks = []

        for b, nmlp in enumerate(self.norm_mlp):
            # print(f"maskestim/{b:02d}")
            qb = q[:, b, :, :]
            mb = nmlp(qb)
            masks.append(mb)

        return masks


class OverlappingMaskEstimationModule(MaskEstimationModuleBase):
    def __init__(
        self,
        in_channel: int,
        band_specs: list[tuple[float, float]],
        freq_weights: list[torch.Tensor],
        n_freq: int,
        emb_dim: int,
        mlp_dim: int,
        cond_dim: int = 0,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: dict = None,
        complex_mask: bool = True,
        norm_mlp_cls: type[nn.Module] = NormMLP,
        norm_mlp_kwargs: dict = None,
        use_freq_weights: bool = True,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)

        # if cond_dim > 0:
        #     raise NotImplementedError
        super().__init__(
            band_specs=band_specs,
            emb_dim=emb_dim + cond_dim,
            mlp_dim=mlp_dim,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            norm_mlp_cls=norm_mlp_cls,
            norm_mlp_kwargs=norm_mlp_kwargs,
        )

        self.n_freq = n_freq
        self.band_specs = band_specs
        self.in_channel = in_channel

        if freq_weights is not None:
            for i, fw in enumerate(freq_weights):
                self.register_buffer(f"freq_weights/{i}", fw.detach())

                self.use_freq_weights = use_freq_weights
        else:
            self.use_freq_weights = False

        self.cond_dim = cond_dim

    def forward(self, q, cond=None):
        # q = (batch, n_bands, n_time, emb_dim)

        batch, n_bands, n_time, emb_dim = q.shape

        if cond is not None:
            print(cond)
            if cond.ndim == 2:
                cond = cond[:, None, None, :].expand(-1, n_bands, n_time, -1)
            elif cond.ndim == 3:
                assert cond.shape[1] == n_time
            else:
                raise ValueError(f"Invalid cond shape: {cond.shape}")

            q = torch.cat([q, cond], dim=-1)
        elif self.cond_dim > 0:
            cond = torch.ones(
                (batch, n_bands, n_time, self.cond_dim),
                device=q.device,
                dtype=q.dtype,
            )
            q = torch.cat([q, cond], dim=-1)
        else:
            pass

        mask_list = self.compute_masks(q)  # [n_bands  * (batch, in_channel, bandwidth, n_time)]

        masks = torch.zeros(
            (batch, self.in_channel, self.n_freq, n_time),
            device=q.device,
            dtype=mask_list[0].dtype,
        )

        for im, mask in enumerate(mask_list):
            fstart, fend = self.band_specs[im]
            if self.use_freq_weights:
                fw = self.get_buffer(f"freq_weights/{im}")[:, None]
                mask = mask * fw
            masks[:, :, fstart:fend, :] += mask

        return masks


class MaskEstimationModule(OverlappingMaskEstimationModule):
    def __init__(
        self,
        band_specs: list[tuple[float, float]],
        emb_dim: int,
        mlp_dim: int,
        in_channel: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: dict = None,
        complex_mask: bool = True,
        **kwargs,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        check_no_overlap(band_specs)
        super().__init__(
            in_channel=in_channel,
            band_specs=band_specs,
            freq_weights=None,
            n_freq=None,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
        )

    def forward(self, q, cond=None):
        # q = (batch, n_bands, n_time, emb_dim)

        masks = self.compute_masks(q)  # [n_bands  * (batch, in_channel, bandwidth, n_time)]

        # TODO: currently this requires band specs to have no gap and no overlap
        masks = torch.concat(masks, dim=2)  # (batch, in_channel, n_freq, n_time)

        return masks


class BanditEncoder(BandSplitModule):
    def __init__(
        self,
        band_specs: str,
        n_fft: int,
        fs: int,
        n_bands: int,
        emb_dim: int,
        in_channel: int,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ):
        super().__init__(
            get_band_specs(band_specs, n_fft, fs, n_bands=n_bands)[0],
            emb_dim,
            in_channel,
            require_no_overlap,
            require_no_gap,
            normalize_channel_independently,
            treat_channel_as_feature,
        )

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, input):
        output = rearrange(input, "b c t f -> b c f t")
        output = super().forward(output)  # (b f t c)
        output = rearrange(output, "b f t c -> b c t f")
        return output


class BanditDecoder(OverlappingMaskEstimationModule):
    def __init__(
        self,
        band_specs: str,
        n_fft: int,
        fs: int,
        n_bands: int,
        emb_dim: int,
        mlp_dim: int,
        in_channel: int,
        n_src: int,
        cond_dim: int = 0,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: dict = None,
        complex_mask: bool = True,
        norm_mlp_cls: type[nn.Module] = NormMLP,
        norm_mlp_kwargs: dict = None,
        use_freq_weights: bool = True,
    ):
        band_specs, freq_weights, _ = get_band_specs(band_specs, n_fft, fs, n_bands=n_bands)
        n_freq = n_fft // 2 + 1
        super().__init__(
            in_channel * n_src,
            band_specs,
            freq_weights,
            n_freq,
            emb_dim,
            mlp_dim,
            cond_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            norm_mlp_cls=norm_mlp_cls,
            norm_mlp_kwargs=norm_mlp_kwargs,
            use_freq_weights=use_freq_weights,
        )
        self.n_src = n_src
        self.n_chan = (
            in_channel  # must not use `self.in_channel` as it's defined as in_channel * n_src in the super class
        )

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, q, cond=None, **kwargs):
        q = rearrange(q, "b d t f -> b f t d")
        masks = super().forward(q, cond)  # (batch, in_channel * n_src, n_freq, n_time)
        masks = rearrange(masks, "b (n c) f t -> b n c t f", n=self.n_src, c=self.n_chan)
        return masks

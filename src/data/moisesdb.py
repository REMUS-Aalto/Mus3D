import json
from pathlib import Path
from typing import ClassVar


import numpy as np
import soundfile as sf
import torch
from einops import rearrange

STEM_TAX = {
    "bass": [
        "bass_guitar",
        "bass_synthesizer_(moog_etc)",
        "contrabass_double_bass_(bass_of_instrings)",
    ],
    "bowed strings": [
        "cello_(solo)",
        "cello_section",
        "other_strings",
        "string_section",
        "viola_section",
        "viola_(solo)",
    ],
    "drums": [
        "cymbals",
        "drum_machine",
        "full_acoustic_drumkit",
        "hi_hat",
        "kick_drum",
        "overheads",
        "snare_drum",
        "toms",
    ],
    "guitar": [
        "acoustic_guitar",
        "clean_electric_guitar",
        "distorted_electric_guitar",
    ],
    "other": ["fx_processed_sound,_scratches,_gun_shots,_explosions_etc"],
    "other keys": [
        "organ,_electric_organ",
        "other_sounds_(hapischord,_melotron_etc)",
        "synth_lead",
        "synth_pad",
    ],
    "other plucked": ["banjo,_mandolin,_ukulele,_harp_etc"],
    "percussion": [
        "a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)",
        "pitched_percussion_(mallets,_glockenspiel,_...)",
    ],
    "piano": [
        "grand_piano",
        "electric_piano_(rhodes,_wurlitzer,_piano_sound_alike)",
    ],
    "vocals": [
        "background_vocals",
        "lead_female_singer",
        "lead_male_singer",
        # Not MoisesDB leaves: they let datasets with coarser vocal metadata
        # (e.g. MUSDB25) label lead/unspecified vocals without guessing gender.
        # MoisesDB songs never populate these; see the paper.
        "lead_vocals",
        "vocals_unspecified",
        "other_(vocoder,_beatboxing_etc)",
    ],
    "wind": [
        "brass_(trumpet,_trombone,_french_horn,_brass_etc)",
        "flutes_(piccolo,_bamboo_flute,_panpipes,_flutes_etc)",
        "other_wind",
        "reeds_(saxophone,_clarinets,_oboe,_english_horn,_bagpipe)",
    ],
}

MAX_N_SRC= len(STEM_TAX)

# leaf stem name -> MoisesDB level-1 category, the grouping the SEG protocol
# does its Case-B subgrouping over.
LEAF_TO_GROUP = {leaf: group for group, leaves in STEM_TAX.items() for leaf in leaves}


def _read_with_sf(file, frames, offset):
    return sf.read(file, frames, offset, dtype="float32", fill_value=0)


class MoisesDBDataset(torch.utils.data.IterableDataset):
    """Songs described by an MSS-schema dataset-info JSON (`dset_info`).

    `dset_dir` is the root of all corpora and the paths in `dset_info` are
    relative to it. `__getitem__` returns one song's aligned stems (validation
    and demos); the training iterators live in the subclasses.
    """

    def __init__(
        self,
        dset_dir: str | Path,
        sample_size: int = 10 * 44100,  # PaSST model queries need 10s of audio
        sample_rate: int = 44100,
        dset_info: str | Path | None = None,
        min_rms: float = 1e-4,
        n_crop_tries: int = 8,
    ):
        super().__init__()
        # RMS floor for "not silence", and how many crop offsets to try before
        # giving up. Subclasses that take `min_rms` of their own overwrite it.
        self.min_rms = min_rms
        self.n_crop_tries = n_crop_tries
        self.dset_dir = Path(dset_dir)
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        if dset_info is None:
            raise ValueError("dset_info (an MSS-schema dataset-info JSON) is required")
        self._init_from_info(dset_info)
        self.root_seed = 0xEFC1F4C6C256453564CC8D66C6457374

    def _init_from_info(self, dset_info):
        """Build the same state from an MSS-schema JSON split file.

        The schema is::

            {"song_info":  {id: {"sources": [[leaf, relpath], ...],
                                 "artist": ..., "name": ...}},
             "stem_paths": {leaf: [relpath, ...]},
             "stem_tax":   {family: [leaf, ...]}}

        Leaf names appear with SPACES in `song_info` and with UNDERSCORES in
        `stem_tax`/`stem_paths`; `STEM_TAX` here uses underscores, so that is the
        canonical form and `_leaf` normalises to it. The taxonomy itself is the
        same MoisesDB level-1 taxonomy this module already encodes -- verified
        leaf-for-leaf against `LEAF_TO_GROUP` -- so `LEAF_TO_GROUP` stays the
        single source of truth for grouping rather than the file's `stem_tax`.
        """
        with open(dset_info, "r") as f:
            info = json.load(f)

        self.song_info = info["song_info"]
        self.song_ids = list(self.song_info)
        self.stem_paths = {
            leaf: [str(self.dset_dir / p) for p in paths]
            for leaf, paths in info["stem_paths"].items()
        }
        self.stem_names = list(self.stem_paths)

        unknown = set(self.stem_names) - set(LEAF_TO_GROUP)
        if unknown:
            raise ValueError(
                f"{dset_info}: leaves absent from STEM_TAX: {sorted(unknown)}"
            )

        self.available_tax = {
            group: [leaf for leaf in leaves if self.stem_paths.get(leaf)]
            for group, leaves in STEM_TAX.items()
        }
        self.available_tax = {g: v for g, v in self.available_tax.items() if v}

    def _leaf(self, source) -> str:
        """Source entry `["bass guitar", "./MoisesDB/.../x.wav"]` -> `bass_guitar`."""
        return source[0].replace(" ", "_")

    def _resolve(self, rel) -> str:
        """Relative path from a JSON manifest -> an existing file under `dset_dir`.

        The manifests are written on more than one machine and disagree about
        the corpus directory's case (`./MoisesDB/` vs `./moisesdb/`), which on
        Linux is the difference between a file and a `LibsndfileError` several
        frames later. Probe the alternatives rather than fail.
        """
        p = self.dset_dir / rel
        if p.exists():
            return str(p)
        parts = Path(rel).parts
        parts = tuple(x for x in parts if x != ".")
        if parts:
            for cand in {parts[0].lower(), parts[0].upper(), "MoisesDB", "MedleyDB"}:
                q = self.dset_dir.joinpath(cand, *parts[1:])
                if q.exists():
                    return str(q)
        return str(p)

    def _audio_path(self, source) -> str:
        """Source entry -> the file on disk."""
        return self._resolve(source[-1])

    def _crop_active(self, path, rng) -> torch.Tensor:
        """[2, sample_size] from a random offset, retried while near-silent.

        Source files are trimmed but not silent-free -- an instrument can rest
        for a bar in the middle -- and the ~7% of stems with no trimmed copy are
        full-length. Retrying costs a read only when it fires.

        Giving up returns the last crop rather than raising: callers that have
        an `active_tracks` pass drop it there, and a track that is silent at
        every one of `n_crop_tries` offsets is one the corpus barely contains.
        """
        path = str(path)
        frames = sf.info(path).frames
        audio = None
        for _ in range(max(1, self.n_crop_tries)):
            offset = rng.integers(0, max(1, frames - self.sample_size))
            audio, _ = _read_with_sf(path, self.sample_size, offset)
            if audio.ndim == 1:                  # defensive: mono source file
                audio = np.stack([audio, audio], axis=-1)
            if float(np.sqrt(np.mean(np.square(audio)))) >= self.min_rms:
                break
        return torch.from_numpy(rearrange(audio, "t c -> c t"))

    def __len__(self):
        return len(self.song_info)

    def __getitem__(self, index):
        """One song's aligned stems, for validation and demo sampling.

        Items have varying shapes, so this is not meant for batched loading.
        """
        rng = np.random.default_rng([0, self.root_seed])
        id = self.song_ids[index]
        info = self.song_info[id]
        sources = info["sources"]
        stem_names = [self._leaf(s).replace("_", " ") for s in sources]

        # Random offset, shared by every stem of the song
        sf_info = sf.info(self._audio_path(sources[0]))
        offset = rng.integers(0, max(1, sf_info.frames - self.sample_size))

        stems = torch.empty((len(sources), 2, self.sample_size))
        for i, s in enumerate(sources):
            name = self._audio_path(s)

            audio, _ = _read_with_sf(name, self.sample_size, offset)
            audio = rearrange(audio, "t c -> c t")

            stems[i] = torch.from_numpy(audio)

        # Same unit definition used at training time; see `_merge_units`.
        stems, stem_names = self._merge_units(stems, stem_names)

        return stems, [id, info["artist"], info["name"], offset, stem_names]

    def __iter__(self):
        raise TypeError(f"{type(self).__name__} has no training iterator; "
                        f"index it (validation) or use a training subclass")

    def _merge_units(self, stems, stem_names):
        """(stems, display names) -> merged units. No-op on the base class.

        `MoisesDBOriginalMixDataset` overrides this with the real thing; the
        random-mix loader here draws one leaf per family from unrelated songs,
        so it has nothing to merge by construction.
        """
        return stems, stem_names

class MoisesDBOriginalMixDataset(MoisesDBDataset):
    """Segments of ORIGINAL mixes: every track of a song cropped at the SAME offset.

    Provides the aligned segment loader, unit merging, activity gating, unit
    dropout and the hierarchy-level draw shared by the SEG and separator
    training sets below. Used directly as the validation set.
    """

    def __init__(
        self,
        dset_dir: str | Path,
        sample_size: int = 10 * 44100,
        sample_rate: int = 44100,
        k_max: int | None = None,
        min_rms: float = 1e-4,
        rel_floor_db: float = -50.0,
        min_active_tracks: int = 2,
        max_retries: int = 20,
        p_track_dropout: float = 0.0,
        p_solo: float = 0.05,
        p_stem: float = 0.5,
        p_full: float = 0.15,
        k_bias: float = 1.0,
        min_keep_frac: float = 0.0,
        premix_leaves: tuple[str, ...] = ("full_acoustic_drumkit", "drum_machine"),
        gain_db: float = 0.0,
        lufs_floor: float | None = -44.0,
        song_weighting: str = "uniform",
        merge_same_leaf: bool = False,
        merge_vocals: bool = True,
        **kwargs,
    ):
        super().__init__(
            dset_dir, sample_size=sample_size, sample_rate=sample_rate, **kwargs
        )
        self.k_max = k_max
        self.min_rms = min_rms
        self.rel_floor_db = rel_floor_db
        self.min_active_tracks = min_active_tracks
        self.max_retries = max_retries
        self.p_track_dropout = p_track_dropout
        # P(the mixture is reduced to a SINGLE unit). Rare by design; see
        # `_drop_units`.
        self.p_solo = p_solo
        # Hierarchy level, drawn per item: `p_stem` is P(level = 1), where every
        # level-1 family is summed into one target. `p_full` is P(every unit is
        # a target), i.e. the grounding case with a silent residual.
        self.p_stem = p_stem
        self.p_full = p_full
        # Leaves that can stand in for a whole level-1 family (e.g. a full
        # drumkit file); see `_choose_level`.
        # Sum same-leaf tracks into one unit; see `_merge_same_leaf`.
        self.merge_same_leaf = merge_same_leaf
        # Sum the whole `vocals` family into one unit; see `_merge_vocals`.
        self.merge_vocals = merge_vocals
        self.premix_leaves = tuple(premix_leaves or ())
        # See `sample_targets`: > 1 biases towards fine, per-track targets.
        self.k_bias = k_bias
        # Floor on track dropout, as a fraction of active tracks; dropout runs
        # before K is drawn, so it also bounds K. See the paper.
        self.min_keep_frac = min_keep_frac
        # Per-track gain, uniform in +/- `gain_db`, applied before mixing.
        # 0.0 (off) by default; see the paper for why this matters.
        self.gain_db = gain_db
        # Absolute loudness floor for a track to become a slot, in LUFS.
        # None disables it (RMS-only gating).
        self.lufs_floor = lufs_floor
        self._meter_cache = None
        # How songs are drawn: "uniform" is per-song, "units" weights by track
        # count so short arrangements don't dominate the step marginal.
        self.song_weighting = song_weighting
        self._song_p = None
        if song_weighting == "units":
            n = np.array([max(len(self.song_info[s]["sources"]), 1)
                          for s in self.song_ids], dtype=np.float64)
            self._song_p = n / n.sum()
        elif song_weighting != "uniform":
            raise ValueError(f"song_weighting must be uniform|units, got "
                             f"{song_weighting!r}")

    def _pick_song(self, rng):
        """One song id, honouring `song_weighting`."""
        if self._song_p is None:
            return self.song_ids[rng.integers(len(self.song_ids))]
        return self.song_ids[rng.choice(len(self.song_ids), p=self._song_p)]

    @property
    def _meter(self):
        """BS.1770 meter, built once per worker and only if actually needed."""
        if self._meter_cache is None:
            from source_embedding_generator.protocol import TorchLoudness

            self._meter_cache = TorchLoudness(sample_rate=self.sample_rate)
        return self._meter_cache

    def _active(self, tracks):
        """`active_tracks` with this dataset's thresholds, meter included."""
        from source_embedding_generator.protocol import active_tracks

        return active_tracks(
            tracks, self.min_rms, self.rel_floor_db,
            lufs_floor=self.lufs_floor,
            meter=None if self.lufs_floor is None else self._meter,
        )

    def _load_song_segment(self, song_id, rng):
        from source_embedding_generator.protocol import active_tracks

        """-> ([N, C, T] tracks, [N] level-1 group names), or None if too quiet."""
        info = self.song_info[song_id]
        sources = info["sources"]

        first = self._audio_path(sources[0])
        frames = sf.info(first).frames
        offset = rng.integers(0, max(1, frames - self.sample_size))

        tracks = torch.empty((len(sources), 2, self.sample_size))
        groups = []
        for i, s in enumerate(sources):
            audio, _ = _read_with_sf(
                self._audio_path(s), self.sample_size, offset
            )
            tracks[i] = torch.from_numpy(rearrange(audio, "t c -> c t"))
            groups.append(LEAF_TO_GROUP[self._leaf(s)])

        # Gain applied before the silence tests, so activity is judged on the
        # audio the model actually receives.
        if self.gain_db > 0.0:
            g = rng.uniform(-self.gain_db, self.gain_db, size=len(sources))
            tracks *= torch.from_numpy(
                10.0 ** (g / 20.0)).float().view(-1, 1, 1)

        # Merge before the silence tests, so activity is judged on the unit
        # that actually becomes a slot.
        leaves = [self._leaf(s) for s in sources]
        tracks, groups, leaves = self._merge_same_leaf(tracks, groups, leaves)
        tracks, groups, leaves = self._merge_vocals(tracks, groups, leaves)

        # Reject near-silent segments.
        if tracks.sum(dim=0).pow(2).mean().sqrt() < self.min_rms:
            return None

        # Drop tracks not playing in this crop, before K is sampled.
        keep = self._active(tracks)
        if keep.sum() < self.min_active_tracks:
            return None
        # Leaf names of the surviving tracks, for inspection tooling only.
        self.last_leaves = [l for l, k in zip(leaves, keep.tolist()) if k]
        return tracks[keep], [g for g, k in zip(groups, keep.tolist()) if k]

    def _merge_same_leaf(self, tracks, groups, leaves):
        """Sum tracks sharing a taxonomy LEAF within one song into one unit.

        Only same-leaf tracks merge, never same-family (see the paper).
        No-op unless `merge_same_leaf` is set. Order is preserved by first
        occurrence.
        """
        if not self.merge_same_leaf or len(set(leaves)) == len(leaves):
            return tracks, groups, leaves
        order, at = [], {}
        for i, l in enumerate(leaves):
            if l in at:
                tracks[at[l]] += tracks[i]
            else:
                at[l] = i
                order.append(i)
        return (tracks[order], [groups[i] for i in order],
                [leaves[i] for i in order])

    def _merge_vocals(self, tracks, groups, leaves):
        """Sum every `vocals`-family track in one song into a single unit.

        Lead and background vocals are always one source, at both hierarchy
        levels; see the paper. No-op at `LEVEL_STEM`, where the family is
        already summed. Order is preserved by first occurrence.
        """
        if not self.merge_vocals:
            return tracks, groups, leaves
        vocal_idx = [i for i, g in enumerate(groups) if g == "vocals"]
        if len(vocal_idx) < 2:
            return tracks, groups, leaves
        keep = vocal_idx[0]
        for i in vocal_idx[1:]:
            tracks[keep] += tracks[i]
        drop = set(vocal_idx[1:])
        order = [i for i in range(len(groups)) if i not in drop]
        return (tracks[order], [groups[i] for i in order],
                [leaves[i] for i in order])

    def _merge_units(self, stems, stem_names):
        """`__getitem__`'s units: same-leaf and vocal-family merges applied.

        Matches the merges `_load_song_segment` does on the training path.
        """
        leaves = [n.replace(" ", "_") for n in stem_names]
        groups = [LEAF_TO_GROUP.get(l, "other") for l in leaves]
        stems, groups, leaves = self._merge_same_leaf(stems, groups, leaves)
        stems, groups, leaves = self._merge_vocals(stems, groups, leaves)
        return stems, [l.replace("_", " ") for l in leaves]

    def _drop_units(self, tracks, groups, level, rng):
        """Drop whole units at the given level, from mixture and targets alike.

        A unit is a raw track at `LEVEL_TRACK`, a whole level-1 family at
        `LEVEL_STEM`; dropping it removes it from both, since the mixture is
        always the sum of the slots. Three outcomes, in order of precedence:

        `p_solo`  keep exactly one unit (the degenerate stem-as-mix case).
        `p_track_dropout`  keep a random subset, floored by `min_keep_frac`.
        otherwise  keep everything.
        """
        from source_embedding_generator.protocol import LEVEL_STEM

        if level == LEVEL_STEM:
            by_family: dict[str, list[int]] = {}
            for i, g in enumerate(groups):
                by_family.setdefault(g, []).append(i)
            units = list(by_family.values())
        else:
            units = [[i] for i in range(len(tracks))]

        n = len(units)
        keep_units = list(range(n))
        if n > 1 and rng.random() < self.p_solo:
            keep_units = [int(rng.integers(n))]
        elif n > self.min_active_tracks and rng.random() < self.p_track_dropout:
            lo = max(2, int(np.ceil(self.min_keep_frac * n)))
            lo = min(lo, n - 1)
            n_keep = int(rng.integers(lo, n))
            keep_units = [int(i) for i in rng.choice(n, size=n_keep, replace=False)]

        idx = sorted(i for u in keep_units for i in units[u])
        self.last_kept_idx = idx
        return tracks[idx], [groups[i] for i in idx]

    def _choose_level(self, rng):
        """Hierarchy level for the segment just loaded.

        `p_stem` decides it, EXCEPT when a premixed leaf (see `premix_leaves`)
        is the ONLY track in its family here -- then stem level is forced.

        The condition is "only one", not "present at all", because the two cases
        are different data:

        *   Kit alone in `drums`. The family IS that one track, so the track- and
            stem-level partitions are byte-identical; the only thing that differs
            is the flag the model is conditioned with. A mixed-down kit is a
            stem, so it is labelled one, and `LEVEL_TRACK` keeps meaning "a raw
            track" rather than sometimes meaning "a whole family".
        *   Kit plus a separate kick or snare (17 of 118 songs). The song does
            have track-level structure, so both levels are legitimate and both
            are used.

        Read `last_leaves`, set by `_load_song_segment`: the tracks that survived
        the activity check, so a kit silent in this crop constrains nothing.
        Called before `_drop_units`, because the level defines what a unit is.
        """
        from source_embedding_generator.protocol import LEVEL_STEM, LEVEL_TRACK

        leaves = list(getattr(self, "last_leaves", None) or ())
        for leaf in leaves:
            if leaf not in self.premix_leaves:
                continue
            fam = LEAF_TO_GROUP[leaf]
            if sum(LEAF_TO_GROUP[l] == fam for l in leaves) == 1:
                return LEVEL_STEM
        return LEVEL_STEM if rng.random() < self.p_stem else LEVEL_TRACK

class MoisesDBOriginalMixSeparationDataset(MoisesDBOriginalMixDataset):
    """Aligned original-mix segments shaped for the QUERY-CONDITIONED SEPARATOR.

    Yields `(stems [MAX_N_SRC, 2, T], level)`: `sample_targets` picks K >= `min_slots`
    units at the drawn hierarchy level, each summed into a slot, zero-padded. The
    mixture is the sum of the slots and the target is slot 0 (the trainer permutes).
    Random: per-track gain in +/- `gain_db`, unit dropout, the level and the
    K-slot partition (the effect chain is applied by `EffectChainDataset`).
    """

    # The SEG datasets expose `collate_slots`, which packs variable-K items and
    # returns a TUPLE. This one yields fixed-shape tensors, so the default
    # collate is correct -- and `launch.py` reads this attribute, so inheriting
    # SEG's would hand the separator trainer a tuple where it indexes
    # `stems[:, 0]`. None means "use the default".
    collate_fn = None

    def __init__(self, *args, max_n_src: int = MAX_N_SRC, min_slots: int = 2,
                 items_per_segment: int = 1, gain_db: float = 6.0, **kwargs):
        super().__init__(*args, gain_db=gain_db, **kwargs)
        self.max_n_src = max_n_src
        # Items yielded per assembled segment; > 1 trades batch diversity for
        # fewer file reads. See `__iter__`.
        self.items_per_segment = items_per_segment
        # Minimum slots per item; 2 is the fewest that poses a separation
        # problem at all (a 1-slot item has mix == target).
        self.min_slots = min_slots

    def __iter__(self):
        from source_embedding_generator.protocol import sample_targets

        worker_info = torch.utils.data.get_worker_info()
        wid = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng([wid, self.root_seed])

        cached, reuse_left = None, 0
        while True:
            try:
                # Reuse one assembled segment for `items_per_segment` items to
                # cut file reads; level, dropout and the K-partition are still
                # redrawn each time, so only the audio repeats.
                if reuse_left > 0:
                    loaded, reuse_left = cached, reuse_left - 1
                else:
                    loaded = None
                    for _ in range(self.max_retries):
                        song_id = self._pick_song(rng)
                        loaded = self._load_song_segment(song_id, rng)
                        if loaded is not None:
                            break
                    if loaded is None:
                        continue
                    cached, reuse_left = loaded, self.items_per_segment - 1

                # Level first, then drop at that granularity: `_drop_units`
                # (not `_drop_tracks`) keeps a stem-level item consistent.
                level = self._choose_level(rng)
                tracks, groups = self._drop_units(*loaded, level, rng)

                # `sample_targets` picks a subset of level-`level` units as
                # slots; the mixture is the sum of the slots, so there is no
                # residual here (unlike SEG). Redraw the partition, rather
                # than the whole segment, until it clears `min_slots`.
                targets = []
                for _ in range(8):
                    targets, _, _, level = sample_targets(
                        len(tracks), groups, k_max=self.max_n_src, rng=rng,
                        k_bias=self.k_bias, level=level, p_full=self.p_full,
                    )
                    # p_full's "grounding" case is not bounded by k_max, so
                    # cap explicitly to protect the fixed-shape allocation.
                    targets = targets[: self.max_n_src]
                    if len(targets) >= self.min_slots:
                        break
                # No partition at this level can clear min_slots; give up.
                if len(targets) < self.min_slots:
                    continue

                stems = torch.zeros((self.max_n_src, 2, self.sample_size))
                for i, idx in enumerate(targets):
                    stems[i] = tracks[idx].sum(dim=0)
                yield stems, level

            except sf.LibsndfileError:
                print("[warn] Soundfile error.")
                continue


class MoisesDBRecursiveDataset(MoisesDBOriginalMixDataset):
    """One step of a RECURSIVE extraction per item, rather than a whole set.

    Yields `(audio [C + 2, C, T], labels, level)` where the first C slots are
    units already extracted (conditioning), the second-to-last is the target to
    generate, and the last is the residual. See `protocol.sample_recursive`.

    Inherits songs, activity gating and track dropout unchanged from the set
    dataset; only how units are split differs.
    """

    def __init__(self, *args, p_stem: float = 0.5, **kwargs):
        # k_bias/k_max/p_full are set-protocol knobs with no meaning here.
        kwargs.pop("k_bias", None)
        kwargs.pop("p_full", None)
        super().__init__(*args, **kwargs)
        self.p_stem = p_stem

    @property
    def collate_fn(self):
        from source_embedding_generator.protocol import collate_recursive

        return collate_recursive

    def __iter__(self):
        from source_embedding_generator.protocol import (
            LEVEL_STEM,
            LEVEL_TRACK,
            build_recursive_slots,
            sample_recursive,
        )

        worker_info = torch.utils.data.get_worker_info()
        wid = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng([wid, self.root_seed])

        while True:
            try:
                loaded = None
                for _ in range(self.max_retries):
                    song_id = self._pick_song(rng)
                    loaded = self._load_song_segment(song_id, rng)
                    if loaded is not None:
                        break
                if loaded is None:
                    continue
                # Level first: dropping must match the granularity the
                # decomposition uses (see `_drop_units`).
                level = self._choose_level(rng)
                tracks, groups = self._drop_units(*loaded, level, rng)

                available, target, residual, level = sample_recursive(
                    len(tracks), groups, rng=rng, level=level
                )
                # Labels at the granularity of the unit: leaf names at track
                # level, family names at stem level. Falls back to `groups`
                # if `last_leaves`/`last_kept_idx` are unavailable.
                names = groups
                if level == LEVEL_TRACK:
                    leaves = list(getattr(self, "last_leaves", None) or ())
                    kept = list(getattr(self, "last_kept_idx", None) or ())
                    if (leaves and kept and len(kept) == len(tracks)
                            and max(kept) < len(leaves)):
                        names = [leaves[i] for i in kept]
                audio, labels = build_recursive_slots(
                    tracks, available, target, residual, names
                )
                # Number of units the segment decomposes into; not
                # recoverable downstream, carried for logging only.
                n_units = (len(set(groups)) if level == LEVEL_STEM
                           else len(tracks))
                yield audio, labels, level, n_units

            except sf.LibsndfileError:
                print("[warn] Soundfile error.")
                continue


class VDBGPOTaxonomy:
    """Maps all MoisesDB leaves to the 6 VDBGPO groups."""
    REDUCED_TAX: ClassVar = {
        "vocals": [
            "background_vocals",
            "lead_female_singer",
            "lead_male_singer",
            "lead_vocals",
            "vocals_unspecified",
            "other_(vocoder,_beatboxing_etc)",
        ],
        "drums": [
            "cymbals",
            "drum_machine",
            "full_acoustic_drumkit",
            "hi_hat",
            "kick_drum",
            "overheads",
            "snare_drum",
            "toms",
        ],
        "bass": [
            "bass_guitar",
            "bass_synthesizer_(moog_etc)",
            "contrabass_double_bass_(bass_of_instrings)",
        ],
        "guitars": [
            "acoustic_guitar",
            "clean_electric_guitar",
            "distorted_electric_guitar",
        ],
        "piano": [
            "grand_piano",
            "electric_piano_(rhodes,_wurlitzer,_piano_sound_alike)",
        ],
        "other": [
            "fx_processed_sound,_scratches,_gun_shots,_explosions_etc",
            "organ,_electric_organ",
            "other_sounds_(hapischord,_melotron_etc)",
            "synth_lead",
            "synth_pad",
            "banjo,_mandolin,_ukulele,_harp_etc",
            "a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)",
            "pitched_percussion_(mallets,_glockenspiel,_...)",
            "cello_(solo)",
            "cello_section",
            "other_strings",
            "string_section",
            "viola_section",
            "viola_(solo)",
            "brass_(trumpet,_trombone,_french_horn,_brass_etc)",
            "flutes_(piccolo,_bamboo_flute,_panpipes,_flutes_etc)",
            "other_wind",
            "reeds_(saxophone,_clarinets,_oboe,_english_horn,_bagpipe)",
        ],
    }

    LEAF_TO_REDUCED: ClassVar = {leaf: group for group, leaves in REDUCED_TAX.items() for leaf in leaves}

    FIXED_ORDER: ClassVar = ["vocals", "drums", "bass", "guitars", "piano", "other"]


class VDBGPOWrapper(torch.utils.data.IterableDataset):
    """Wraps any dataset, converting its output to VDBGPO's 6-stem format.

    Stems are grouped and summed into (vocals, drums, bass, guitars, piano,
    other). Training wraps `__iter__` with random silent stems; validation
    and demo wrap `__getitem__` and load all stems from a song.
    
    Design
    ------
    Delegation is used where possible: `__getattr__` forwards every unknown
    attribute to the wrapped dataset, so `collate_fn`, `song_ids`, `sample_rate`,
    `available_tax`, etc. keep working through the wrapper.
    """

    # Import taxonomy from the class
    REDUCED_TAX = VDBGPOTaxonomy.REDUCED_TAX
    LEAF_TO_REDUCED = VDBGPOTaxonomy.LEAF_TO_REDUCED
    FIXED_ORDER = VDBGPOTaxonomy.FIXED_ORDER

    def __init__(
        self,
        dataset,
        sample_size: int = 10 * 44100,
        sample_rate: int = 44100,
        p_active: float = 0.9,  # Probability a stem is active (not silent)
    ):
        super().__init__()
        import hydra.utils
        from omegaconf import DictConfig

        if isinstance(dataset, (dict, DictConfig)):
            dataset = hydra.utils.instantiate(dataset)
        
        self.dataset = dataset
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.p_active = p_active

        self.n_src = 6

        self.root_seed = getattr(dataset, 'root_seed', 0)

    def __getattr__(self, name):
        # `dataset` itself is a miss during unpickling in a worker, which
        # would otherwise recurse forever.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        """Yields 6 stems in fixed order, each independently silent w.p. `1 - p_active`.

        Loads directly from the underlying dataset's `stem_paths`, not its
        `__iter__`.
        """
        fixed_order = self.FIXED_ORDER

        available_tax = getattr(self.dataset, 'available_tax', {})
        stem_paths = getattr(self.dataset, 'stem_paths', {})

        dataset_available_tax = {
            group: [leaf for leaf in leaves if stem_paths.get(leaf)]
            for group, leaves in self.REDUCED_TAX.items()
        }
        dataset_available_tax = {g: v for g, v in dataset_available_tax.items() if v}

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rng = np.random.default_rng([worker_info.id, self.root_seed])
        else:
            rng = np.random.default_rng([0, self.root_seed])
        
        while True:
            try:
                stems = torch.zeros((self.n_src, 2, self.sample_size))
                
                for i, group in enumerate(fixed_order):
                    if dataset_available_tax.get(group) and rng.random() < self.p_active:
                        leaf = rng.choice(dataset_available_tax[group], replace=False)
                        track = rng.choice(stem_paths[leaf], replace=False)
                        stems[i] = self.dataset._crop_active(track, rng)

                yield stems

            except (sf.LibsndfileError, Exception) as e:
                if isinstance(e, sf.LibsndfileError):
                    print("[warn] Soundfile error.")
                else:
                    print(f"[warn] Error in VDBGPOWrapper.__iter__: {e}")
                continue

    def __getitem__(self, index):
        """Song stems from the underlying dataset, grouped and summed into
        the 6 VDBGPO groups. Returns [id, artist, name, offset, stem_names].
        """
        item = self.dataset[index]

        if isinstance(item, tuple):
            stems, info = item
        else:
            stems = item
            info = []

        if info and len(info) >= 5:
            id, artist, name, offset, stem_names = info
        else:
            id, artist, name, offset, stem_names = "", "", "", 0, []

        fixed_order = self.FIXED_ORDER

        output_stems = torch.zeros((self.n_src, 2, self.sample_size))
        group_to_index = {group: i for i, group in enumerate(fixed_order)}

        for i, leaf_name in enumerate(stem_names):
            # stem_names use spaces; the taxonomy map uses underscores
            leaf_key = leaf_name.replace(" ", "_")
            group = self.LEAF_TO_REDUCED.get(leaf_key)
            if group is None:
                group = "other"

            idx = group_to_index[group]

            output_stems[idx] += stems[i]

        new_stem_names = fixed_order

        return output_stems, [id, artist, name, offset, new_stem_names]


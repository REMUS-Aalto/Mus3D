"""Translation of AlbumDB stem filenames into the MoisesDB taxonomy.

AlbumDB (``AlbumDB/stems_mixed/<NN Song Title>/<stem>.wav``) carries no metadata
at all: the only label is the file name.  The names are systematic
(``<family>_<qualifier>...``), so this module maps every basename observed in
the 10 songs onto a leaf of ``src.data.moisesdb.STEM_TAX``, the same target
taxonomy used by the base table (see ``label_taxonomy``).

Two AlbumDB-specific conventions:

* Many stems are split across channel positions (``_l`` / ``_r`` / ``_c``) or
  across song sections (``_intro``, ``_outro``, ``_verse2``).  These are not
  distinct instruments, and since the preparation script sums every track that
  lands on the same leaf, they merge automatically -- ``vox_back_l``,
  ``vox_back_r`` and ``vox_back_c`` all become one ``background_vocals`` track.
* The album has a single male lead singer, so lead-vocal stems map to
  ``lead_male_singer`` with no gender ambiguity (unlike MUSDB25, which needed
  the gender-agnostic ``lead_vocals`` leaf).
"""

from __future__ import annotations

# --- MoisesDB leaves, aliased so the table below stays readable ------------
_ATONAL = "a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)"
_PITCHED = "pitched_percussion_(mallets,_glockenspiel,_...)"
_ORGAN = "organ,_electric_organ"
_PLUCKED = "banjo,_mandolin,_ukulele,_harp_etc"

#: stem basename (no extension) -> MoisesDB leaf.
LABEL_TO_LEAF: dict[str, str] = {
    # ---------------- bass ----------------
    "bass": "bass_guitar",
    "bass_lo": "bass_guitar",
    "bass_melody": "bass_guitar",
    "bass_synth": "bass_synthesizer_(moog_etc)",
    # ---------------- bowed strings ----------------
    "cello": "cello_(solo)",
    "viola": "viola_(solo)",
    "violin_1": "other_strings",
    "violin_2": "other_strings",
    # ---------------- drums ----------------
    "drums_cymbal": "cymbals",
    "drums_cymbal_reverse": "cymbals",
    "drums_ride": "cymbals",
    "drums_kick": "kick_drum",
    "drums_outro_kick_rim": "kick_drum",
    "drums_rim": "snare_drum",
    "drums_snare": "snare_drum",
    "drums_snare_1": "snare_drum",
    "drums_snare_2": "snare_drum",
    "drums_snare2": "snare_drum",
    "drums_snare_rim": "snare_drum",
    "drums_toms": "toms",
    "drums_overheads": "overheads",
    "drums_outro_overheads": "overheads",
    "drums_intro": "full_acoustic_drumkit",
    "drums_outro_kit": "full_acoustic_drumkit",
    # ---------------- guitar: acoustic ----------------
    "guitar_acoustic": "acoustic_guitar",
    "guitar_acoustic_l": "acoustic_guitar",
    "guitar_acoustic_r": "acoustic_guitar",
    "guitar_acoustic_main": "acoustic_guitar",
    "guitar_acoustic_outro": "acoustic_guitar",
    "guitar_acoustic_lead_harmony": "acoustic_guitar",
    "guitar_acoustic_lead_l": "acoustic_guitar",
    "guitar_acoustic_lead_r": "acoustic_guitar",
    "guitar_acoustic_solo": "acoustic_guitar",
    "guitar_acoustic_solo_hi": "acoustic_guitar",
    "guitar_acoustic_solo_lo": "acoustic_guitar",
    "guitar_acoustic_solo_l": "acoustic_guitar",
    "guitar_acoustic_solo_r": "acoustic_guitar",
    "guitar_acoustic_strummed_l": "acoustic_guitar",
    "guitar_acoustic_strummed_r": "acoustic_guitar",
    # ---------------- guitar: electric ----------------
    "guitar_electric_background": "clean_electric_guitar",
    "guitar_electric_intro": "clean_electric_guitar",
    "guitar_electric_intro_lo": "clean_electric_guitar",
    "guitar_electric_l": "clean_electric_guitar",
    "guitar_electric_r": "clean_electric_guitar",
    "guitar_electric_lead": "clean_electric_guitar",
    "guitar_electric_lead_l": "clean_electric_guitar",
    "guitar_electric_lead_r": "clean_electric_guitar",
    "guitar_electric_main": "clean_electric_guitar",
    "guitar_electric_octaves": "clean_electric_guitar",
    "guitar_electric_octaves_2": "clean_electric_guitar",
    "guitar_electric_outro_lead": "clean_electric_guitar",
    "guitar_electric_outro_rhythm": "clean_electric_guitar",
    "guitar_electric_rhythm": "clean_electric_guitar",
    "guitar_electric_rhythm_2": "clean_electric_guitar",
    "guitar_electric_rhythm_2_l": "clean_electric_guitar",
    "guitar_electric_rhythm_2_r": "clean_electric_guitar",
    "guitar_electric_rhythm_3": "clean_electric_guitar",
    "guitar_electric_rhythm_c": "clean_electric_guitar",
    "guitar_electric_rhythm_l": "clean_electric_guitar",
    "guitar_electric_rhythm_l_2": "clean_electric_guitar",
    "guitar_electric_rhythm_r": "clean_electric_guitar",
    "guitar_electric_rhythm_r_2": "clean_electric_guitar",
    "guitar_electric_smooth_l": "clean_electric_guitar",
    "guitar_electric_smooth_r": "clean_electric_guitar",
    "guitar_electric_solo": "clean_electric_guitar",
    "guitar_electric_solo_c": "clean_electric_guitar",
    "guitar_electric_solo_harmony": "clean_electric_guitar",
    "guitar_electric_solo_intro": "clean_electric_guitar",
    "guitar_electric_solo_l": "clean_electric_guitar",
    "guitar_electric_solo_oct": "clean_electric_guitar",
    "guitar_electric_solo_r": "clean_electric_guitar",
    "guitar_electric_strum": "clean_electric_guitar",
    "guitar_electric_strummed": "clean_electric_guitar",
    "guitar_electric_strummed_l": "clean_electric_guitar",
    "guitar_electric_strummed_r": "clean_electric_guitar",
    "guitar_electric_verse2": "clean_electric_guitar",
    "guitar_electric_wah_l": "clean_electric_guitar",
    "guitar_electric_wah_r": "clean_electric_guitar",
    "guitar_electric_heavy_l": "distorted_electric_guitar",
    "guitar_electric_heavy_r": "distorted_electric_guitar",
    # ---------------- keys ----------------
    "keys_piano": "grand_piano",
    "keys_organ": _ORGAN,
    "keys_organ_outro": _ORGAN,
    "keys_synth_1": "synth_pad",
    "keys_synth_2": "synth_pad",
    "keys_synth_3": "synth_pad",
    "keys_synth_4": "synth_pad",
    "keys_synth_hi": "synth_pad",
    "keys_synth_lo": "synth_pad",
    "keys_synth_l": "synth_pad",
    "keys_synth_r": "synth_pad",
    "keys_synth_outro": "synth_pad",
    "keys_synth_solo_3": "synth_lead",
    "keys_synth_solo_hi": "synth_lead",
    "keys_synth_solo_lo": "synth_lead",
    # ---------------- percussion ----------------
    "perc_clap_1": _ATONAL,
    "perc_clap_2": _ATONAL,
    "perc_clap_3": _ATONAL,
    "perc_clap_4": _ATONAL,
    "perc_clap_5": _ATONAL,
    "perc_clap_6": _ATONAL,
    "perc_claps": _ATONAL,
    "perc_clap_sample": _ATONAL,
    "perc_clave": _ATONAL,
    "perc_clicks_1": _ATONAL,
    "perc_clicks_2": _ATONAL,
    "perc_shaker": _ATONAL,
    "perc_tambourine": _ATONAL,
    "perc_tapping": _ATONAL,
    "perc_triangle": _ATONAL,
    "perc_glockenspiel": _PITCHED,
    # ---------------- other plucked ----------------
    "ukelele": _PLUCKED,
    # ---------------- vocals ----------------
    # One male lead singer on the whole album, so lead stems carry the gender.
    "vox_lead": "lead_male_singer",
    "vox_lead_double_l": "lead_male_singer",
    "vox_lead_double_r": "lead_male_singer",
    "vox_lead_outro": "lead_male_singer",
    "vox_outro": "lead_male_singer",
    "vox_back_c": "background_vocals",
    "vox_back_l": "background_vocals",
    "vox_back_r": "background_vocals",
    "vox_backing_c": "background_vocals",
    "vox_backing_c_hi": "background_vocals",
    "vox_backing_c_hi_2": "background_vocals",
    "vox_backing_chorus_l": "background_vocals",
    "vox_backing_chorus_r": "background_vocals",
    "vox_backing_hi": "background_vocals",
    "vox_backing_hi_2_l": "background_vocals",
    "vox_backing_hi_2_r": "background_vocals",
    "vox_backing_hi_l": "background_vocals",
    "vox_backing_hi_r": "background_vocals",
    "vox_backing_l": "background_vocals",
    "vox_backing_l_2": "background_vocals",
    "vox_backing_lo_l": "background_vocals",
    "vox_backing_lo_r": "background_vocals",
    "vox_backing_low_l": "background_vocals",
    "vox_backing_low_r": "background_vocals",
    "vox_backing_mid": "background_vocals",
    "vox_backing_outro": "background_vocals",
    "vox_backing_outro_l": "background_vocals",
    "vox_backing_outro_r": "background_vocals",
    "vox_backing_r": "background_vocals",
    "vox_backing_r_2": "background_vocals",
    "vox_backing_response": "background_vocals",
    "vox_backing_solo": "background_vocals",
    "vox_backing_verse": "background_vocals",
}

#: Assignments that are a judgement call rather than an obvious match.
GUESSED: set[str] = {
    # tone unknown from the name; MUSDB25 makes the same call for bare
    # "electric guitar", and only the explicitly "heavy" stems go to distorted.
    *(k for k, v in LABEL_TO_LEAF.items() if v == "clean_electric_guitar"),
    # section stems of the kit that are not broken out by drum
    "drums_intro", "drums_outro_kit",
    # rim shots parked with the snare they are played on
    "drums_rim", "drums_snare_rim", "drums_outro_kick_rim",
    # generic synth: pad vs lead unknown, only "_solo_" names claim lead
    "keys_synth_1", "keys_synth_2", "keys_synth_3", "keys_synth_4",
    "keys_synth_hi", "keys_synth_lo", "keys_synth_l", "keys_synth_r",
    "keys_synth_outro",
    # unlabelled section vocal; the album's only lead voice is the male singer
    "vox_outro",
    # clicks/tapping have no MoisesDB home beyond a-tonal percussion
    "perc_clicks_1", "perc_clicks_2", "perc_tapping", "perc_clap_sample",
}


def leaf_for(label: str) -> str:
    """Return the MoisesDB leaf for an AlbumDB stem basename.

    Raises ``KeyError`` for an unknown name: unlike MUSDB25 there is no free-text
    label space here, the file names are a closed set, and silently bucketing a
    new one into "other" would hide a real stem.
    """
    return LABEL_TO_LEAF[label.strip().lower()]

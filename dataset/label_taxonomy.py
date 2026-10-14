"""The base instrument-label table: free-text stem labels -> MoisesDB taxonomy.

Maps every label observed across the corpora onto a leaf of
``src.data.moisesdb.STEM_TAX``.  Two label namespaces are covered and they never
conflict: MoisesDB-like leaf names (``distorted electric guitar``, ``male
singer``) and Hornbostel-Sachs-ish class paths (``percussion.drumset``).

This is CORPUS-PREPARATION ONLY -- nothing at training time imports it.  The
per-corpus tables extend it: ``medleydb_taxonomy.py`` adds the jazz/world labels
on top, ``albumdb_taxonomy.py`` keeps its own.

The mapping is hand-authored; entries listed in ``GUESSED`` are the ones where
the MoisesDB taxonomy has no clean home and a judgement call was made.
``docs/musdb25_taxonomy_coverage.md`` is the human-readable review copy (it was
compiled against MUSDB25, the corpus this table was first written for, and the
judgement calls it reviews are the same ones).
"""

from __future__ import annotations

# --- MoisesDB leaves, aliased so the table below stays readable ------------
_FX = "fx_processed_sound,_scratches,_gun_shots,_explosions_etc"
_ATONAL = "a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)"
_PITCHED = "pitched_percussion_(mallets,_glockenspiel,_...)"
_EPIANO = "electric_piano_(rhodes,_wurlitzer,_piano_sound_alike)"
_ORGAN = "organ,_electric_organ"
_OTHERKEYS = "other_sounds_(hapischord,_melotron_etc)"
_PLUCKED = "banjo,_mandolin,_ukulele,_harp_etc"
_BRASS = "brass_(trumpet,_trombone,_french_horn,_brass_etc)"
_FLUTES = "flutes_(piccolo,_bamboo_flute,_panpipes,_flutes_etc)"
_REEDS = "reeds_(saxophone,_clarinets,_oboe,_english_horn,_bagpipe)"
_VOX_OTHER = "other_(vocoder,_beatboxing_etc)"
_DBASS = "contrabass_double_bass_(bass_of_instrings)"

#: The leaves of the MoisesDB ``vocals`` level-1 family.  ``lead_vocals`` and
#: ``vocals_unspecified`` are additions for datasets whose vocal metadata is
#: coarser than MoisesDB's; both are empty on MoisesDB proper.
VOCAL_LEAVES: frozenset[str] = frozenset({
    "background_vocals", "lead_vocals", "vocals_unspecified",
    "lead_female_singer", "lead_male_singer", _VOX_OTHER,
})

# --- label -> MoisesDB leaf ------------------------------------------------
# Keys are lower-cased instrument strings, from both the stem level and the raw
# level of the source metadata (the two namespaces overlap and never conflict).
LABEL_TO_LEAF: dict[str, str] = {
    # ---------------- bass ----------------
    "bass": "bass_guitar",
    "electric bass": "bass_guitar",
    "string.plucked.lute.guitar.bass": "bass_guitar",
    "string.plucked.bass": "bass_guitar",
    "bass synthesizer": "bass_synthesizer_(moog_etc)",
    "synthesizer.bass": "bass_synthesizer_(moog_etc)",
    "double bass": _DBASS,
    "pizzicato bass": _DBASS,
    "wind.brass.bass": _BRASS,  # tuba/sousaphone acting as bass
    # ---------------- drums ----------------
    "drum set": "full_acoustic_drumkit",
    "drumset": "full_acoustic_drumkit",
    "drums": "full_acoustic_drumkit",
    "percussion.drumset": "full_acoustic_drumkit",
    "brush kit": "full_acoustic_drumkit",
    "brushes": "full_acoustic_drumkit",
    "kick drum": "kick_drum",
    "bass drum": "kick_drum",
    "snare drum": "snare_drum",
    "rim": "snare_drum",
    "sidestick": "snare_drum",
    "tom": "toms",
    "toms": "toms",
    "floor tom": "toms",
    "hi hat": "hi_hat",
    "high hat": "hi_hat",
    "cymbal": "cymbals",
    "ride cymbal": "cymbals",
    "crash cymbal": "cymbals",
    "china cymbal": "cymbals",
    "reverse cymbal": "cymbals",
    "finger cymbals": "cymbals",
    "gong roll": "cymbals",
    "overhead": "overheads",
    "drum machine": "drum_machine",
    "drum loop": "drum_machine",
    "electronic percussion": "drum_machine",
    "sampler": "drum_machine",
    "percussion.unpitched.membraphones": "full_acoustic_drumkit",
    # ---------------- percussion ----------------
    "percussion": _ATONAL,
    "auxiliary percussion": _ATONAL,
    "claps": _ATONAL,
    "shaker": _ATONAL,
    "tambourine": _ATONAL,
    "cowbell": _ATONAL,
    "djembe": _ATONAL,
    "hand drum": _ATONAL,
    "bongo": _ATONAL,
    "conga": _ATONAL,
    "cabasa": _ATONAL,
    "maracas": _ATONAL,
    "rainstick": _ATONAL,
    "guiro": _ATONAL,
    "woodblock": _ATONAL,
    "sticks": _ATONAL,
    "triangle": _ATONAL,
    "anvil": _ATONAL,
    "sleigh bells": _ATONAL,
    "taiko": _ATONAL,
    "timbale": _ATONAL,
    "wineglasses": _ATONAL,
    "percussion.unpitched.idiophones": _ATONAL,
    "percussion.unpitched.idiophones.rattles": _ATONAL,
    "percussion.unpitched.idiophones.body": _ATONAL,
    "percussion.unpitched.idiophones.metal": _ATONAL,
    "percussion.unpitched.idiophones.wood": _ATONAL,
    "vibraphone": _PITCHED,
    "glockenspiel": _PITCHED,
    "marimba": _PITCHED,
    "timpani": _PITCHED,
    "percussion.timpani": _PITCHED,
    "percussion.pitched": _PITCHED,
    "bells": _PITCHED,
    "hand bells": _PITCHED,
    "school bell": _PITCHED,
    # ---------------- guitar ----------------
    "acoustic guitar": "acoustic_guitar",
    "dobro": "acoustic_guitar",
    "clean electric guitar": "clean_electric_guitar",
    "electric guitar": "clean_electric_guitar",
    "guitar": "clean_electric_guitar",
    "rhythm guitar": "clean_electric_guitar",
    "lap steel guitar": "clean_electric_guitar",
    "slide guitar": "clean_electric_guitar",
    "string.plucked.lute.guitar": "clean_electric_guitar",
    "string.plucked.lute": "clean_electric_guitar",
    "string.plucked": "clean_electric_guitar",
    "distorted electric guitar": "distorted_electric_guitar",
    "guitar synthesizer": "synth_lead",
    "electric guitar synthesizer": "synth_lead",
    "string.plucked.lute.guitar.synthesizer": "synth_lead",
    # ---------------- other plucked ----------------
    "banjo": _PLUCKED,
    "mandolin": _PLUCKED,
    "ukulele": _PLUCKED,
    "harp": _PLUCKED,
    "sitar": _PLUCKED,
    # ---------------- piano / keys ----------------
    "piano": "grand_piano",
    "keys.piano": "grand_piano",
    "tack piano": "grand_piano",
    "electric piano": _EPIANO,
    "rhodes": _EPIANO,
    "wurlitzer": _EPIANO,
    "clavinet": _EPIANO,
    "keys.piano.electronic": _EPIANO,
    "organ": _ORGAN,
    "hammond": _ORGAN,
    "keys.organ": _ORGAN,
    "accordion": _ORGAN,
    "keys.free_reeds": _ORGAN,
    "wind.woodwind.free_reeds": _ORGAN,
    "harmonica": _ORGAN,
    "keys": _OTHERKEYS,
    "mellotron": _OTHERKEYS,
    "synthesizer": "synth_pad",
    "string synthesizer": "synth_pad",
    "string.synthesizer": "synth_pad",
    "choir synthesizer": "synth_pad",
    # ---------------- bowed strings ----------------
    "string": "other_strings",
    "string.bowed": "other_strings",
    "strings": "string_section",
    "string section": "string_section",
    "orchestra": "string_section",
    "ensemble": "string_section",
    "violin": "other_strings",
    "fiddle": "other_strings",
    "violin section": "string_section",
    "viola": "viola_(solo)",
    "viola section": "viola_section",
    "cello": "cello_(solo)",
    # ---------------- wind ----------------
    "wind.brass": _BRASS,
    "brass": _BRASS,
    "brass section": _BRASS,
    "horn section": _BRASS,
    "horns": _BRASS,
    "trumpet": _BRASS,
    "trumpet section": _BRASS,
    "cornet": _BRASS,
    "euphonium": _BRASS,
    "tuba": _BRASS,
    "french horn": _BRASS,
    "flute": _FLUTES,
    "wind.woodwind.flutes": _FLUTES,
    "whistle": _FLUTES,
    "clarinet": _REEDS,
    "wind.woodwind.single_reeds": _REEDS,
    # ---------------- vocals ----------------
    # Each vocal label goes to the leaf that asserts exactly what the label
    # asserts, and no more.  "lead vocals" says lead but never says the gender;
    # "vocals"/"vocalists" do not even say lead vs background.  Neither the
    # gender nor the role is invented.
    "lead vocals": "lead_vocals",
    "vocals": "vocals_unspecified",
    "vocalists": "vocals_unspecified",
    "backing vocals": "background_vocals",
    "gang vocals": "background_vocals",
    "alto vocals": "background_vocals",
    "male singer": "lead_male_singer",
    "male lead vocals": "lead_male_singer",
    "male rapper": "lead_male_singer",
    "female singer": "lead_female_singer",
    "female lead vocals": "lead_female_singer",
    "synth vocals": _VOX_OTHER,
    "vocals.synthesizer": _VOX_OTHER,
    "beatboxing": _VOX_OTHER,
    # ---------------- other / fx ----------------
    "fx/processed sound": _FX,
    "fx_processed": _FX,
    "fx_processed.sfx": _FX,
    "fx_processed.sample": _FX,
    "sfx": _FX,
    "sfx sample": _FX,
    "glitch sfx": _FX,
    "sample": _FX,
    "scratches": _FX,
    "sprinkles": _FX,
    "break": _FX,
    "click": _FX,
    "noise": _FX,
    "hit": _FX,
    "electronic hit": _FX,
    "sub drop": _FX,
    "misc": _FX,
    "other": _FX,
}

#: Labels whose assignment is a judgement call rather than an obvious match.
#: Reviewed by a human via ``docs/musdb25_taxonomy_coverage.md``.
GUESSED: set[str] = {
    # ambiguous electric-guitar tone (clean vs distorted unknown from label)
    "electric guitar", "guitar", "rhythm guitar", "string.plucked.lute.guitar",
    "string.plucked.lute", "string.plucked", "lap steel guitar", "slide guitar",
    "dobro",
    # rappers have no MoisesDB leaf
    "male rapper",
    # loops/samplers: acoustic-vs-machine origin unknown
    "drum loop", "sampler", "electronic percussion",
    # HS class covers whole unpitched membranophone family, mostly kit toms
    "percussion.unpitched.membraphones",
    # generic synth: pad vs lead unknown from the label alone
    "synthesizer", "string synthesizer", "string.synthesizer",
    "choir synthesizer",
    # free-reed aerophones parked with organ
    "accordion", "harmonica", "keys.free_reeds", "wind.woodwind.free_reeds",
    # bass-register brass
    "wind.brass.bass",
    # no MoisesDB home at all -> "other"
    "sfx", "sfx sample", "glitch sfx", "sample", "break", "click", "noise",
    "hit", "electronic hit", "sub drop", "misc", "other", "sprinkles",
    "school bell", "wineglasses", "anvil", "taiko",
    # orchestral catch-alls
    "orchestra", "ensemble", "string", "string.bowed",
    "keys", "mellotron",
}

#: Fallback used when a label is absent from the table (should never fire; the
#: preparation script asserts full coverage).
FALLBACK_LEAF = _FX


def leaf_for(label: str) -> str:
    """Return the MoisesDB leaf for an instrument label."""
    return LABEL_TO_LEAF.get((label or "").strip().lower(), FALLBACK_LEAF)

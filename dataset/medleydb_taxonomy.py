"""MedleyDB instrument labels -> MoisesDB taxonomy.

``label_taxonomy.py`` already covers most MedleyDB stem labels; this module
reuses that base table and adds only the labels it does not have.  Those extras
are mostly jazz and world instruments -- saxophones, trombones, tabla, oud.

``main system`` is not an instrument.  It is MedleyDB's label for the room / PA
capture on live recordings, i.e. the whole band on one track.  It has no leaf
and must never become a separation target, so it is listed in ``NON_SOURCE``
and handled by the preparation script rather than mapped.
"""

from __future__ import annotations

from label_taxonomy import LABEL_TO_LEAF as _BASE_TABLE

_ATONAL = "a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)"
_ORGAN = "organ,_electric_organ"
_PLUCKED = "banjo,_mandolin,_ukulele,_harp_etc"
_BRASS = "brass_(trumpet,_trombone,_french_horn,_brass_etc)"
_REEDS = "reeds_(saxophone,_clarinets,_oboe,_english_horn,_bagpipe)"

#: Labels that are not instruments and must not become separation targets.
#: A "main system" track is the full band captured on the room/PA mics; using
#: it as a target would teach the model to answer any query with the mixture.
NON_SOURCE: frozenset[str] = frozenset({"main system"})

#: MedleyDB labels absent from the base table.
EXTRA_LABELS: dict[str, str] = {
    # ---------------- wind: reeds ----------------
    "tenor saxophone": _REEDS,
    "alto saxophone": _REEDS,
    "soprano saxophone": _REEDS,
    "baritone saxophone": _REEDS,
    "bass clarinet": _REEDS,
    "oboe": _REEDS,
    "bassoon": _REEDS,
    # ---------------- wind: brass ----------------
    "trombone": _BRASS,
    "trombone section": _BRASS,
    # ---------------- keys ----------------
    "electronic organ": _ORGAN,
    # free-reed aerophone, parked with organ like harmonica/accordion already are
    "melodica": _ORGAN,
    # ---------------- percussion ----------------
    "tabla": _ATONAL,
    "darbuka": _ATONAL,
    "doumbek": _ATONAL,
    "castanet": _ATONAL,
    # ---------------- other plucked ----------------
    "oud": _PLUCKED,
}

LABEL_TO_LEAF: dict[str, str] = {**_BASE_TABLE, **EXTRA_LABELS}

#: Assignments in EXTRA_LABELS that are a judgement call.
GUESSED: set[str] = {
    # hand drums: MoisesDB's only home is the a-tonal bucket, though tabla and
    # darbuka are pitched enough that "pitched percussion" is arguable
    "tabla", "darbuka", "doumbek",
    # free-reed aerophone with no MoisesDB leaf
    "melodica",
    # fretless lute; the plucked leaf names banjo/mandolin/ukulele/harp only
    "oud",
}


def leaf_for(label: str) -> str | None:
    """MedleyDB stem label -> MoisesDB leaf.

    Returns ``None`` for a ``NON_SOURCE`` label (the caller must drop the
    track), and raises ``KeyError`` for an unknown one -- MedleyDB's label set
    is closed and reviewed, so an unseen label means the table needs extending,
    not a silent fallback to "other".
    """
    key = (label or "").strip().lower()
    if key in NON_SOURCE:
        return None
    return LABEL_TO_LEAF[key]

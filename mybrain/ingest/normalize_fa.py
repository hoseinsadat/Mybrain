"""Persian/Arabic-script normalization.

Two forms:
- normalize_persian(): canonical form for storage and display (unifies yeh/kaf/alef variants, digits,
  strips tatweel and diacritics, keeps ZWNJ so words render correctly).
- lexical_form(): what BM25 / tsvector / alias matching see (adds ZWNJ->space, casefold, punctuation->space).
"""

from __future__ import annotations

import re
import unicodedata

_CHAR_MAP = {
    "ي": "ی",  # Arabic yeh -> Persian yeh
    "ى": "ی",  # alef maksura -> Persian yeh
    "ك": "ک",  # Arabic kaf -> Persian kaf
    "أ": "ا",  # alef with hamza above -> alef
    "إ": "ا",  # alef with hamza below -> alef
    "ٱ": "ا",  # alef wasla -> alef
    "ة": "ه",  # teh marbuta -> heh
    "ۀ": "ه",  # heh with yeh above -> heh
    "ـ": "",  # tatweel
    "‍": "",  # ZWJ
    "‎": "",  # LRM
    "‏": "",  # RLM
    "﻿": "",  # BOM
    "­": "",  # soft hyphen
}
_DIGITS = {**{chr(0x660 + i): str(i) for i in range(10)}, **{chr(0x6F0 + i): str(i) for i in range(10)}}
_TABLE = str.maketrans({**_CHAR_MAP, **_DIGITS})
_DIACRITICS = re.compile(r"[ً-ٰٕۖ-ۭ]")
_ZWNJ = "‌"
_PERSIAN_RANGE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]")
_HONORIFICS = re.compile(
    r"^(?:(?:آقای|خانم|دکتر|مهندس|جناب|سرکار|حاج|حاجی|سید|mr|mrs|ms|dr|prof|sir|eng)\.?\s+)+", re.IGNORECASE
)


def normalize_persian(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TABLE)
    text = _DIACRITICS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # ZWNJ padded by spaces is noise from copy/paste
    text = re.sub(r"\s*‌\s*", _ZWNJ, text)
    return text


def lexical_form(text: str) -> str:
    text = normalize_persian(text).replace(_ZWNJ, " ").casefold()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_name(text: str) -> str:
    """Alias key for entity resolution: lexical form without honorifics or corporate suffix noise."""
    t = lexical_form(text)
    t = _HONORIFICS.sub("", t)
    t = re.sub(r"\b(?:llc|ltd|inc|corp|co|plc|gmbh|sa|ag|شرکت|سهامی خاص|سهامی عام)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def has_persian(text: str) -> bool:
    return bool(_PERSIAN_RANGE.search(text or ""))


def detect_lang(text: str) -> str:
    fa = len(_PERSIAN_RANGE.findall(text or ""))
    en = len(_LATIN.findall(text or ""))
    if fa == 0 and en == 0:
        return "unknown"
    if fa > 3 * en:
        return "fa"
    if en > 3 * fa:
        return "en"
    return "mixed"

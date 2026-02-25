from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Pattern, Tuple


_RE_BLOCK: dict[str, Pattern[str]] = {
    "meta": re.compile(r"\[\[META\]\](.*?)\[\[/META\]\]", re.DOTALL),
    "public": re.compile(r"\[\[PUBLIC_REPLY\]\](.*?)\[\[/PUBLIC_REPLY\]\]", re.DOTALL),
    "private": re.compile(r"\[\[PRIVATE_REPLY\]\](.*?)\[\[/PRIVATE_REPLY\]\]", re.DOTALL),
}


_MALFORMED_OPEN_CLOSE_PAIRS = (
    ("[[META]]", "[[/META]]"),
    ("[[PUBLIC_REPLY]]", "[[/PUBLIC_REPLY]]"),
    ("[[PRIVATE_REPLY]]", "[[/PRIVATE_REPLY]]"),
)


@dataclass(frozen=True)
class Envelope:
    meta: Dict[str, str]
    public: str
    private: str
    status: str


def _last_block(pattern: Pattern[str], text: str) -> Tuple[Optional[str], int]:
    matches = list(pattern.finditer(text))
    if not matches:
        return None, 0
    return matches[-1].group(1), len(matches)


def _parse_meta(meta_text: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in meta_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k:
            continue
        meta[k] = v.strip()
    return meta


def _has_malformed_tags(raw: str) -> bool:
    for open_tag, close_tag in _MALFORMED_OPEN_CLOSE_PAIRS:
        if (open_tag in raw) ^ (close_tag in raw):
            return True
    return False


def _salvage_public_before_meta(raw: str) -> str:
    """
    Some web UIs return malformed wrappers like:
    [[PUBLIC_REPLY]] ... [[META]] ... [[/META]]
    without closing [[/PUBLIC_REPLY]]. Recover the public payload before META.
    """
    t = raw or ""
    if "[[PUBLIC_REPLY]]" not in t:
        return ""
    if "[[/PUBLIC_REPLY]]" in t:
        return ""
    m = re.search(r"\[\[PUBLIC_REPLY\]\](.*?)(?:\[\[META\]\]|$)", t, re.DOTALL)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def parse_envelope(raw_text: str) -> Envelope:
    raw = raw_text or ""
    status_flags: list[str] = []

    meta_block, meta_n = _last_block(_RE_BLOCK["meta"], raw)
    public_block, pub_n = _last_block(_RE_BLOCK["public"], raw)
    private_block, priv_n = _last_block(_RE_BLOCK["private"], raw)

    if meta_n > 1:
        status_flags.append("MULTI_META")
    if pub_n > 1:
        status_flags.append("MULTI_PUBLIC")
    if priv_n > 1:
        status_flags.append("MULTI_PRIVATE")
    if _has_malformed_tags(raw):
        status_flags.append("MALFORMED")

    meta = _parse_meta(meta_block) if meta_block is not None else {}
    public = (public_block or "").strip()
    if not public:
        public = _salvage_public_before_meta(raw)
    private = (private_block or "").strip()

    if public == "" and (meta_block is None and private_block is None):
        status = "NO_TAGS"
    elif public == "":
        status = "NO_PUBLIC"
    else:
        status = "OK"

    if status_flags:
        status = status + "|" + "|".join(status_flags)

    return Envelope(meta=meta, public=public, private=private, status=status)

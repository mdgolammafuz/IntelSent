
from __future__ import annotations

import re
from typing import Iterable, List, Optional

def _normalize(s: str) -> str:
    s = s.strip().lower()
    # trim trailing punctuation
    s = re.sub(r'[\s,;:.]+$', '', s)
    # compress whitespace
    s = re.sub(r'\s+', ' ', s)
    return s

def extract_driver(contexts: Iterable[str], patterns: List[str]) -> Optional[str]:
    """
    Scan contexts with provided regex patterns and return a short, normalized phrase.
    Returns None if nothing matches.
    """
    if not contexts:
        return None

    # compile all patterns once
    regs = [re.compile(p, flags=re.IGNORECASE) for p in patterns]

    # search in order, first hit wins
    for ctx in contexts:
        if not ctx:
            continue
        for rx in regs:
            m = rx.search(ctx)
            if m:
                # first capturing group should be the driver phrase
                phrase = m.group(1)
                return _normalize(phrase)

    return None

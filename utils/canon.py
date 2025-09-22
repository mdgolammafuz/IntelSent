import re

_SUBSTS = [
    (re.compile(r"\bsearch and news advertising\b", re.I), "search"),
    (re.compile(r"(?<!xbox )game pass\b", re.I), "xbox game pass"),
    (re.compile(r"\boffice\s*365\s*commercial\b", re.I), "office 365"),
    (re.compile(r"\bmicrosoft\s*365\b", re.I), "office 365"),
    (re.compile(r"\boffice\s*commercial\b", re.I), "office 365"),
    (re.compile(r"\bwindows\s*oem\b", re.I), "windows"),
    (re.compile(r"\bwindows\s*commercial\b", re.I), "windows"),
    (re.compile(r"\bazure and other cloud services\b", re.I), "azure"),
    (re.compile(r"\bserver products\b", re.I), "server products"),
    (re.compile(r"\bcloud services\b", re.I), "cloud services"),
    (re.compile(r"\biphone(?: net sales)?\b", re.I), "iphone"),
    (re.compile(r"\bhigher revenue per search\b", re.I), "search"),
    (re.compile(r"\bgrowth in xbox game pass subscriptions\b", re.I), "xbox game pass"),
]
_PREFIXES = [
    re.compile(r"^growth in\s+", re.I),
    re.compile(r"^higher (revenue|net sales) of\s+", re.I),
    re.compile(r"^(increase[s]?|decrease[s]?|improvement[s]?) in\s+", re.I),
]

def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s).strip(" ,.;:-")
    return s

def canonicalize(s: str) -> str:
    s = normalize(s)
    for rx, repl in _SUBSTS:
        s = rx.sub(repl, s)
    for rx in _PREFIXES:
        s = rx.sub("", s)
    s = re.sub(r"\b(\w+)\s+\1\b", r"\1", s)
    return s

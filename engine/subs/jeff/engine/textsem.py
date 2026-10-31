"""What a text says, read from its words and style: the JSON carries the
literal string, font, weight, size, colour and tracking of every text, and
geometry alone throws most of that away.

kinds(el) returns a set of tags:
  date, time, price, offer, url, email, phone, address  - details a reader looks up
  attribution   "- Martin Luther", "Steve Jobs" under a quote, a script signature
  quote         a sentence set as a quotation (quote marks, or a long line ending in . ! ?)
  label         a one-word caption over a value ("FRIDAY", "MARCH", "WHEN")
  cta           a short imperative ("Shop now", "Book your table")
  kicker        a short tracked or all-caps line that leads ("NO BOYS ALLOWED")
  script        set in a handwriting face
  runon         display copy far longer than display copy usually is
protected_spans(text) gives the character ranges a line break must not split
(a date, a time, a price, a phone number, a name after a dash, "No. 5").
"""

from __future__ import annotations

import re

MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DAYS = r"(?:mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:rs(?:day)?)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)"

DATE = re.compile(
    rf"\b(?:\d{{1,2}}[./-]\d{{1,2}}(?:[./-]\d{{2,4}})?|{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTHS}(?:\s+\d{{4}})?|(?:19|20)\d{{2}}|{DAYS}|{MONTHS})\b", re.I)
TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)|\b\d{1,2}:\d{2}\b", re.I)
PRICE = re.compile(r"[$€£¥₹]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:usd|eur|gbp|dollars?|euros?)\b", re.I)
OFFER = re.compile(r"\b\d{1,3}\s?%|\b(?:sale|off|free|discount|deal|save|bogo)\b", re.I)
URL = re.compile(r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|io|co|us|uk|de|shop|store|info|biz)\b(?:/\S*)?", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE = re.compile(r"(?:\+\d[\d\s().-]{7,}\d|\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b|\b\d{2,4}[\s-]\d{3,4}[\s-]\d{3,4}\b)")
ADDRESS = re.compile(r"\b\d{1,5}\s+(?:[A-Z][\w']*\s){1,4}(?:st|street|ave|avenue|rd|road|blvd|dr|drive|ln|lane|way|pl|place|sq|square)\b\.?", re.I)
CTA = re.compile(
    r"^\W*(?:shop|buy|order|learn|get|sign|try|download|join|book|start|discover|explore|subscribe|register|"
    r"reserve|claim|grab|see|watch|read|call|visit|apply|donate|contact|find|request|enroll|rsvp)\b", re.I)
SCRIPT_FONTS = re.compile(r"script|hand|brush|pacifico|vibes|satisfy|sacramento|allura|parisienne|kaushan|"
                          r"lobster|caveat|amatic|signature|marker|cursive|calligraph|tangerine|yellowtail", re.I)
DASH_NAME = re.compile(r"^\s*[-–—~]\s*\S")


def _letters(s):
    return re.sub(r"[^A-Za-z]", "", s)


def kinds(el: dict) -> set:
    t = str(el.get("content", "")).strip()
    if not t:
        return set()
    out = set()
    flat = " ".join(t.split())
    words = flat.split()
    n = len(words)
    for tag, rx in (("date", DATE), ("time", TIME), ("price", PRICE), ("url", URL),
                    ("email", EMAIL), ("phone", PHONE), ("address", ADDRESS)):
        if rx.search(flat):
            out.add(tag)
    if OFFER.search(flat) and n <= 6:
        out.add("offer")
    fam = el.get("font_family", "") or ""
    if SCRIPT_FONTS.search(fam):
        out.add("script")
    letters = _letters(flat)
    caps = bool(letters) and letters.upper() == letters
    if DASH_NAME.match(flat) and n <= 6:
        out.add("attribution")
    elif 1 <= n <= 4 and not out & {"date", "time", "price", "url", "email", "phone"} and \
            all(w[:1].isupper() for w in words if w[:1].isalpha()) and not caps and not flat.endswith((".", "!", "?", ":")) \
            and not CTA.match(flat):
        out.add("name")           # Title Case, a few words: a person, a place, a brand
    if re.match(r'^\s*["“‘«]', flat) or (n >= 6 and flat.endswith((".", "!", "?", '"', "”"))):
        out.add("quote")
    if n == 1 and len(letters) >= 3 and (caps or flat.endswith(":")) and not out & {"date", "time", "price"}:
        out.add("label")
    if n == 1 and caps and re.fullmatch(rf"{DAYS}|{MONTHS}", flat, re.I):
        out.add("label")
    if CTA.match(flat) and n <= 5:
        out.add("cta")
    ls = float(el.get("letter_spacing", 0) or 0)
    fs = float(el.get("font_size", 0) or 0)
    if n <= 6 and (caps or (fs and ls >= 0.08 * fs)) and len(flat) <= 40:
        out.add("kicker")
    if len(flat) > 90 and fs and fs >= 28:
        out.add("runon")
    return out


DETAIL = {"date", "time", "price", "url", "email", "phone", "address"}


def is_detail(el: dict) -> bool:
    """Information a reader looks up (when, where, how much, how to reach),
    with little else on the line."""
    k = kinds(el)
    if not k & DETAIL:
        return False
    flat = " ".join(str(el.get("content", "")).split())
    rest = flat
    for rx in (DATE, TIME, PRICE, URL, EMAIL, PHONE, ADDRESS):
        rest = rx.sub(" ", rest)
    return len(_letters(rest)) <= max(12, 0.5 * len(_letters(flat)))


_SPANS = [DATE, TIME, PRICE, PHONE, EMAIL, URL, ADDRESS,
          re.compile(r"[-–—]\s*\S+(?:\s+[A-Z]\S*){0,2}"),          # "- Martin Luther"
          re.compile(r"\b(?:No|Nr|St|Dr|Mr|Mrs|Ms|Mt)\.\s+\S+"),   # "No. 5", "St. Paul"
          re.compile(r"\b\d+\s+\S+")]                               # "5 PM", "24 hours", "3 ways"


def protected_spans(text: str) -> list:
    """Character ranges that must stay on one line."""
    spans = []
    for rx in _SPANS:
        for m in rx.finditer(text):
            a, b = m.start(), m.end()
            if " " in text[a:b].strip():
                spans.append((a, b))
    return spans


def break_cost(text: str, wrapped: str) -> float:
    """How badly a wrap reads: a protected span split across lines, or a lone
    word left on the last line of a multi-line text."""
    if "\n" not in wrapped:
        return 0.0
    cost = 0.0
    brk = [i for i, c in enumerate(wrapped) if c == "\n" and i < len(text) and text[i] in " \n"]
    for a, b in protected_spans(text):
        if any(a < i < b for i in brk):
            cost += 1.0
    lines = [ln for ln in wrapped.split("\n") if ln.strip()]
    if len(lines) >= 2 and len(lines[-1].split()) == 1 and len(lines[-2].split()) >= 2:
        cost += 0.5
    return cost

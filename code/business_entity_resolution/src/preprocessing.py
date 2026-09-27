import re
import unicodedata
from typing import Tuple, Optional, Set

# Legal suffix normalization map (US, India, UK, France, Germany)
LEGAL_SUFFIX_MAP = {
    r'\b(pvt|pvtltd|private limited|pvt ltd)\b': 'private limited',
    r'\b(ltd|limited)\b': 'limited',
    r'\b(inc|incorporated)\b': 'incorporated',
    r'\b(corp|corporation)\b': 'corporation',
    r'\b(llc|l\.l\.c\.)\b': 'llc',
    r'\b(llp|l\.l\.p\.)\b': 'llp',
    r'\b(co|company)\b': 'company',
    r'\b(gmbh)\b': 'gmbh',
    r'\b(sa|s\.a\.)\b': 'sa',
    r'\b(sarl|s\.a\.r\.l\.)\b': 'sarl',
    r'\b(sas|s\.a\.s\.|sasu)\b': 'sas',
    r'\b(eurl|e\.u\.r\.l\.)\b': 'eurl',
    r'\b(sci|s\.c\.i\.)\b': 'sci',
    r'\b(snc|s\.n\.c\.)\b': 'snc',
    r'\b(gie|g\.i\.e\.)\b': 'gie',
    r'\b(selarl|earl|scop|sem)\b': 'sarl',
    r'\b(societe|entreprise|etablissement|ets)\b': 'company'
}

BASE_NAME_STRIP_RE = re.compile(
    r'\b(private limited|limited|incorporated|corporation|llc|llp|company|gmbh|sa|sarl|sas|sasu|eurl|sci|snc|gie|pvt|ltd|inc|corp|co|societe|entreprise)\b',
    re.IGNORECASE
)

# High-speed word substitution dictionary for address abbreviations and US/Indian/French streets
ADDR_AND_STATE_WORD_MAP = {
    # Road abbreviations (English & French)
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard', 'bd': 'boulevard',
    'ln': 'lane', 'dr': 'drive', 'ct': 'court', 'pl': 'place', 'sq': 'square',
    'hwy': 'highway', 'flr': 'floor', 'fl': 'floor', 'apt': 'apartment', 'ste': 'suite',
    'no': 'number', 'ph': 'phase', 'sec': 'sector', 'off': 'office', 'near': 'nr',
    'rue': 'street', 'chemin': 'chemin', 'ch': 'chemin', 'impasse': 'impasse', 'imp': 'impasse',
    'allee': 'allee', 'all': 'allee', 'quai': 'quai', 'cours': 'cours', 'route': 'route', 'rte': 'route',
    # US states
    'mo': 'missouri', 'oh': 'ohio', 'va': 'virginia', 'ny': 'new york', 'ca': 'california',
    'tx': 'texas', 'fl': 'florida', 'il': 'illinois', 'pa': 'pennsylvania', 'nc': 'north carolina',
    'mi': 'michigan', 'ga': 'georgia', 'nj': 'new jersey', 'wa': 'washington', 'az': 'arizona',
    'ma': 'massachusetts', 'tn': 'tennessee', 'in': 'indiana', 'md': 'maryland', 'wi': 'wisconsin',
    'co': 'colorado', 'mn': 'minnesota', 'sc': 'south carolina', 'al': 'alabama', 'la': 'louisiana',
    'ky': 'kentucky', 'or': 'oregon', 'ok': 'oklahoma', 'ct': 'connecticut', 'ia': 'iowa',
    'ut': 'utah', 'ar': 'arkansas', 'nv': 'nevada', 'ms': 'mississippi', 'ks': 'kansas',
    'nm': 'new mexico', 'ne': 'nebraska', 'id': 'idaho', 'wv': 'west virginia', 'hi': 'hawaii',
    'nh': 'new hampshire', 'me': 'maine', 'ri': 'rhode island', 'mt': 'montana', 'de': 'delaware',
    'sd': 'south dakota', 'nd': 'north dakota', 'ak': 'alaska', 'vt': 'vermont', 'wy': 'wyoming',
    # India states
    'mh': 'maharashtra', 'ka': 'karnataka', 'dl': 'delhi', 'up': 'uttar pradesh',
    'wb': 'west bengal', 'gj': 'gujarat', 'rj': 'rajasthan', 'ts': 'telangana',
    'ap': 'andhra pradesh', 'hr': 'haryana', 'mp': 'madhya pradesh', 'pb': 'punjab', 'kl': 'kerala'
}

# Regexes
DOMAIN_RE = re.compile(r'(\.com|\.in|\.org|\.net|\.co|\.io|\.biz|\.info|\.gov|\.edu)\b', re.IGNORECASE)
POSTAL_CODE_RE = re.compile(r'\b([1-9][0-9]{5}|[0-9]{5})\b')
HOUSE_NUM_RE = re.compile(r'\b([0-9]{1,6})\b')
ALL_NUMS_RE = re.compile(r'\b([0-9]{1,6})\b')
PUNCT_RE = re.compile(r'[^a-z0-9\s]')
LEET_TRANS = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a', '$': 's'})

def clean_leetspeak(word: str) -> str:
    """Repairs words where digits or symbols are accidentally substituted for letters."""
    if any(c.isdigit() or c in '@$' for c in word) and any(c.isalpha() for c in word):
        return word.translate(LEET_TRANS)
    return word

def strip_accents(text: str) -> str:
    """Normalizes unicode characters (NFKD) and removes diacritics / accents (crucial for France)."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def normalize_country(country_str: Optional[str]) -> str:
    """Normalizes country string while preserving all country labels (including France)."""
    if not country_str or not isinstance(country_str, str):
        return "UNKNOWN"
    return country_str.strip().upper()

def normalize_name(text: Optional[str]) -> str:
    """
    Advanced Business Name Normalizer:
    1. Unicode NFKD accent decomposition (handles France accents seamlessly).
    2. Strips URL domain suffixes (.com, .org, .net, etc.) before punctuation removal.
    3. Handles '&' -> 'and'.
    4. Canonicalizes legal suffixes.
    5. Strips non-alphanumeric punctuation.
    6. Normalizes leetspeak token corruptions.
    """
    if not text or not isinstance(text, str):
        return ""
    
    clean = strip_accents(text).lower()
    clean = DOMAIN_RE.sub('', clean)
    clean = clean.replace("&", " and ")
    
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        clean = re.sub(pattern, replacement, clean)
        
    clean = PUNCT_RE.sub(' ', clean)
    tokens = [clean_leetspeak(t) for t in clean.split() if t]
    return " ".join(tokens)

def get_base_name(norm_name: str) -> str:
    """Extracts the core business name without legal suffixes."""
    if not norm_name:
        return ""
    base = BASE_NAME_STRIP_RE.sub('', norm_name)
    base = re.sub(r'\s+', ' ', base).strip()
    return base if len(base) >= 3 else norm_name

def extract_postal_code(text: Optional[str]) -> str:
    """Extracts 5-digit (US/France) or 6-digit (India) postal codes from address text."""
    if not text or not isinstance(text, str):
        return ""
    matches = POSTAL_CODE_RE.findall(text)
    return matches[-1] if matches else ""

def normalize_address(text: Optional[str]) -> Tuple[str, str, str, str]:
    """
    Normalizes an address string:
    - Extracts postal code
    - Extracts primary building number and set of distinct address numbers
    - Standardizes road abbreviations and US/Indian states
    - Returns tuple: (norm_addr, postal_code, house_num, address_numbers_str)
    """
    if not text or not isinstance(text, str):
        return "", "", "", ""
    
    clean_raw = strip_accents(text)
    matches_pin = POSTAL_CODE_RE.findall(clean_raw)
    postal_code = matches_pin[-1] if matches_pin else ""
    
    all_nums = ALL_NUMS_RE.findall(clean_raw)
    house_num = all_nums[0] if all_nums else ""
    nums_str = ",".join(sorted(set(all_nums)))
    
    clean = PUNCT_RE.sub(' ', clean_raw.lower().replace("&", " and "))
    words = [ADDR_AND_STATE_WORD_MAP.get(w, w) for w in clean.split()]
    norm_addr = " ".join(words)
    
    return norm_addr, postal_code, house_num, nums_str

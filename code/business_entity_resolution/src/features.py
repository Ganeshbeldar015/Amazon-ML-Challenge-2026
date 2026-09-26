import re
import math
from typing import List, Set, Optional
from rapidfuzz import fuzz
from preprocessing import get_base_name

def token_jaccard(t1: List[str], t2: List[str]) -> float:
    s1, s2 = set(t1), set(t2)
    if not s1 or not s2:
        return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0

def token_overlap(t1: List[str], t2: List[str]) -> float:
    s1, s2 = set(t1), set(t2)
    if not s1 or not s2:
        return 0.0
    d = min(len(s1), len(s2))
    return len(s1 & s2) / d if d > 0 else 0.0

def char_3grams(text: str) -> Set[str]:
    s = text.replace(" ", "")
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s) - 2)}

def char_qgram_jaccard(text1: str, text2: str) -> float:
    g1 = char_3grams(text1)
    g2 = char_3grams(text2)
    if not g1 or not g2:
        return 0.0
    u = len(g1 | g2)
    return len(g1 & g2) / u if u > 0 else 0.0

def simple_soundex(token: str) -> str:
    """Fast Soundex phonetic algorithm."""
    if not token or not token.isalpha():
        return ""
    token = token.upper()
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    first_letter = token[0]
    encoded = [first_letter]
    prev = mapping.get(first_letter, '')
    for char in token[1:]:
        code = mapping.get(char, '')
        if code and code != prev:
            encoded.append(code)
            prev = code
        elif not code:
            prev = ''
    soundex_code = "".join(encoded).replace('0', '')
    return (soundex_code + "000")[:4]

def check_acronym(t1: List[str], t2: List[str]) -> float:
    """Checks if one entity is an acronym of the other (e.g. TCS <-> Tata Consultancy Services)."""
    if len(t1) == 1 and len(t2) >= 2:
        acro = "".join(w[0] for w in t2 if w and w[0].isalpha())
        if t1[0] == acro:
            return 1.0
    elif len(t2) == 1 and len(t1) >= 2:
        acro = "".join(w[0] for w in t1 if w and w[0].isalpha())
        if t2[0] == acro:
            return 1.0
    return 0.0

def longest_common_prefix_len(s1: str, s2: str) -> int:
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

NUM_RE = re.compile(r'\b[0-9]{1,6}\b')

class FeatureExtractor:
    """
    45-dimension state-of-the-art feature extractor for Business Entity Resolution:
    - Multi-angle name similarity (exact, base exact, unspaced, prefix-2, acronym, fuzz, sort, set, partial, q-gram, length ratio)
    - Phonetic Soundex matching for names and street titles
    - Longest common prefix and shared token counts
    - Address tail (City/State) overlap and street number comparisons
    - Geometric mean and non-linear interaction features
    - Multi-tier geographic and postal hierarchy
    """
    FEATURE_NAMES = [
        # Name features (20)
        "name_exact",
        "name_base_exact",
        "name_unspaced_match",
        "name_base_unspaced",
        "name_first_token_match",
        "name_prefix2_match",
        "is_acronym_match",
        "name_soundex_match",
        "name_lcp_ratio",
        "name_shared_tokens_count",
        "name_base_jaccard",
        "name_fuzz_ratio",
        "name_token_sort",
        "name_token_set",
        "name_partial_ratio",
        "name_qgram_jaccard",
        "name_jaccard",
        "name_len_diff",
        "name_token_count_diff",
        "name_num_conflict",
        # Address & Geographic features (17)
        "has_addr_both",
        "house_num_match",
        "house_num_diff",
        "addr_has_shared_num",
        "addr_nums_jaccard",
        "addr_nums_conflict",
        "addr_exact",
        "addr_first_token_match",
        "addr_soundex_match",
        "addr_tail_overlap",
        "addr_shared_tokens_count",
        "addr_fuzz_ratio",
        "addr_partial_ratio",
        "addr_token_sort",
        "addr_token_set",
        "addr_qgram_jaccard",
        "addr_overlap",
        # Interaction features (2)
        "name_x_addr_sim",
        "geom_mean_sim",
        # Postal hierarchy (5)
        "both_have_postal",
        "postal_match",
        "postal_prefix2",
        "postal_prefix3",
        "postal_conflict",
        # Source flag (1)
        "is_source2"
    ]

    @classmethod
    def extract_pair_features(
        cls,
        name1: str,
        addr1: str,
        country1: str,
        postal1: str,
        house1: str,
        name2: str,
        addr2: str,
        country2: str,
        postal2: str,
        house2: str,
        target_id: str = ""
    ) -> List[float]:
        t_name1 = name1.split()
        t_name2 = name2.split()
        t_addr1 = addr1.split()
        t_addr2 = addr2.split()

        # ---------------------------------------------------------------------
        # 1. Business Name Features (20)
        # ---------------------------------------------------------------------
        name_exact = 1.0 if name1 and name1 == name2 else 0.0
        
        base1 = get_base_name(name1)
        base2 = get_base_name(name2)
        name_base_exact = 1.0 if base1 and base1 == base2 else 0.0
        
        u1 = name1.replace(" ", "")
        u2 = name2.replace(" ", "")
        name_unspaced_match = 1.0 if u1 and u1 == u2 else 0.0
        
        bu1 = base1.replace(" ", "")
        bu2 = base2.replace(" ", "")
        name_base_unspaced = 1.0 if bu1 and bu1 == bu2 else 0.0
        
        first_token_match = 1.0 if t_name1 and t_name2 and t_name1[0] == t_name2[0] else 0.0
        
        if len(t_name1) >= 2 and len(t_name2) >= 2:
            prefix2_match = 1.0 if (t_name1[0] == t_name2[0] and t_name1[1] == t_name2[1]) else 0.0
        else:
            prefix2_match = first_token_match
            
        is_acronym = check_acronym(t_name1, t_name2)
        
        # Phonetic Soundex match on primary token
        if t_name1 and t_name2 and len(t_name1[0]) >= 3 and len(t_name2[0]) >= 3:
            sx1 = simple_soundex(t_name1[0])
            sx2 = simple_soundex(t_name2[0])
            name_soundex = 1.0 if sx1 and sx1 == sx2 else 0.0
        else:
            name_soundex = first_token_match
            
        # Longest common prefix ratio
        lcp_len = longest_common_prefix_len(name1, name2)
        max_l = max(len(name1), len(name2), 1)
        name_lcp_ratio = lcp_len / max_l
        
        # Shared tokens count (normalized up to 5)
        s_tokens1 = set(t_name1)
        s_tokens2 = set(t_name2)
        name_shared_cnt = min(len(s_tokens1 & s_tokens2), 5) / 5.0
        
        # Base name token Jaccard
        t_base1 = base1.split()
        t_base2 = base2.split()
        name_base_jacc = token_jaccard(t_base1, t_base2)
        
        name_fuzz = fuzz.ratio(name1, name2) / 100.0
        name_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
        name_token_set = fuzz.token_set_ratio(name1, name2) / 100.0
        name_partial = fuzz.partial_ratio(name1, name2) / 100.0
        name_qgram = char_qgram_jaccard(name1, name2)
        name_jacc = token_jaccard(t_name1, t_name2)
        
        name_len_diff = abs(len(name1) - len(name2)) / max_l
        max_tokens = max(len(t_name1), len(t_name2), 1)
        name_token_count_diff = abs(len(t_name1) - len(t_name2)) / max_tokens

        nums1 = set(NUM_RE.findall(name1))
        nums2 = set(NUM_RE.findall(name2))
        name_num_conflict = 1.0 if (nums1 and nums2 and nums1 != nums2) else 0.0

        # ---------------------------------------------------------------------
        # 2. Address & Geographic Features (17)
        # ---------------------------------------------------------------------
        has_addr_both = 1.0 if addr1 and addr2 else 0.0

        if has_addr_both > 0.0:
            house_num_match = 1.0 if house1 and house2 and house1 == house2 else (0.5 if not house1 or not house2 else 0.0)
            
            # Numeric difference in house numbers
            if house1.isdigit() and house2.isdigit():
                h1, h2 = int(house1), int(house2)
                h_diff = min(abs(h1 - h2), 100) / 100.0
            else:
                h_diff = 0.0 if house_num_match == 1.0 else 0.5
            
            a_nums1 = set(NUM_RE.findall(addr1))
            a_nums2 = set(NUM_RE.findall(addr2))
            if a_nums1 and a_nums2:
                u_nums = len(a_nums1 | a_nums2)
                addr_nums_jacc = len(a_nums1 & a_nums2) / u_nums if u_nums > 0 else 0.0
                addr_nums_conflict = 1.0 if len(a_nums1 & a_nums2) == 0 else 0.0
                addr_has_shared_num = 1.0 if (a_nums1 & a_nums2) else 0.0
            else:
                addr_nums_jacc = 0.5
                addr_nums_conflict = 0.0
                addr_has_shared_num = 0.5

            addr_exact = 1.0 if addr1 == addr2 else 0.0
            addr_first_tok_match = 1.0 if (t_addr1 and t_addr2 and t_addr1[0] == t_addr2[0]) else 0.0
            
            # Street name soundex
            st1 = [w for w in t_addr1 if w.isalpha() and len(w) >= 3]
            st2 = [w for w in t_addr2 if w.isalpha() and len(w) >= 3]
            if st1 and st2:
                addr_sx = 1.0 if simple_soundex(st1[0]) == simple_soundex(st2[0]) else 0.0
            else:
                addr_sx = 0.5
                
            # Tail tokens overlap (City & State in tail)
            tail1 = t_addr1[-2:] if len(t_addr1) >= 2 else t_addr1
            tail2 = t_addr2[-2:] if len(t_addr2) >= 2 else t_addr2
            addr_tail_ov = token_overlap(tail1, tail2)
            
            addr_shared_cnt = min(len(set(t_addr1) & set(t_addr2)), 8) / 8.0
            
            addr_fuzz = fuzz.ratio(addr1, addr2) / 100.0
            addr_partial = fuzz.partial_ratio(addr1, addr2) / 100.0
            addr_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
            addr_set = fuzz.token_set_ratio(addr1, addr2) / 100.0
            addr_qgram = char_qgram_jaccard(addr1, addr2)
            addr_ov = token_overlap(t_addr1, t_addr2)
            
            name_x_addr = name_sort * addr_sort
            geom_mean = math.sqrt(name_sort * addr_sort)
        else:
            house_num_match = 0.5
            h_diff = 0.5
            addr_has_shared_num = 0.5
            addr_nums_jacc = 0.5
            addr_nums_conflict = 0.0
            addr_exact = 0.0
            addr_first_tok_match = 0.0
            addr_sx = 0.5
            addr_tail_ov = 0.0
            addr_shared_cnt = 0.0
            addr_fuzz = 0.0
            addr_partial = 0.0
            addr_sort = 0.0
            addr_set = 0.0
            addr_qgram = 0.0
            addr_ov = 0.0
            name_x_addr = name_sort * 0.5
            geom_mean = math.sqrt(name_sort * 0.5)

        # ---------------------------------------------------------------------
        # 3. Postal Hierarchy (5)
        # ---------------------------------------------------------------------
        both_have_postal = 1.0 if postal1 and postal2 else 0.0
        if both_have_postal > 0.0:
            postal_match = 1.0 if postal1 == postal2 else 0.0
            postal_prefix2 = 1.0 if (len(postal1) >= 2 and len(postal2) >= 2 and postal1[:2] == postal2[:2]) else 0.0
            postal_prefix3 = 1.0 if (len(postal1) >= 3 and len(postal2) >= 3 and postal1[:3] == postal2[:3]) else 0.0
            postal_conflict = 1.0 if postal1 != postal2 else 0.0
        else:
            postal_match = 0.5
            postal_prefix2 = 0.5
            postal_prefix3 = 0.5
            postal_conflict = 0.0

        # ---------------------------------------------------------------------
        # 4. Source Indicator (1)
        # ---------------------------------------------------------------------
        is_source2 = 1.0 if target_id.startswith("S2-") else 0.0

        return [
            # Name features (20)
            name_exact,
            name_base_exact,
            name_unspaced_match,
            name_base_unspaced,
            first_token_match,
            prefix2_match,
            is_acronym,
            name_soundex,
            name_lcp_ratio,
            name_shared_cnt,
            name_base_jacc,
            name_fuzz,
            name_sort,
            name_token_set,
            name_partial,
            name_qgram,
            name_jacc,
            name_len_diff,
            name_token_count_diff,
            name_num_conflict,
            # Address & Geographic features (17)
            has_addr_both,
            house_num_match,
            h_diff,
            addr_has_shared_num,
            addr_nums_jacc,
            addr_nums_conflict,
            addr_exact,
            addr_first_tok_match,
            addr_sx,
            addr_tail_ov,
            addr_shared_cnt,
            addr_fuzz,
            addr_partial,
            addr_sort,
            addr_set,
            addr_qgram,
            addr_ov,
            # Interaction features (2)
            name_x_addr,
            geom_mean,
            # Postal hierarchy (5)
            both_have_postal,
            postal_match,
            postal_prefix2,
            postal_prefix3,
            postal_conflict,
            # Source flag (1)
            is_source2
        ]

import json
import pandas as pd

import re
from typing import List, Dict, Union

def lookup_original_ee(original_index, original_ee_df):
    pass

def sop_to_lists(sop_path):
    with open(sop_path, "r") as f:
        sop_json = json.load(f)
    
    titles = []
    descriptions = []
    for evidence_category in sop_json["EvidenceCategories"]:
        titles.append(evidence_category["title"])
        descriptions.append(evidence_category["description"])
    
    return titles, descriptions

EVIDENCE_RE   = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL | re.IGNORECASE)
TAG_PATTERNS  = {
    "category":              re.compile(r"<category>(.*?)</category>", re.DOTALL | re.IGNORECASE),
    "explanation":           re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL | re.IGNORECASE),
    "score":                 re.compile(r"<score>(.*?)</score>", re.DOTALL | re.IGNORECASE),
    "score_adjustment_reason": re.compile(r"<score_adjustment_reason>(.*?)</score_adjustment_reason>", 
                                          re.DOTALL | re.IGNORECASE),
}

def _clean(txt: str) -> str:
    """Trim leading/trailing whitespace and collapse internal runs of whitespace."""
    return re.sub(r"\s+", " ", txt.strip())

def _parse_block(block: str) -> Union[Dict[str, Union[str, float]], int]:
    """
    Parse a single <evidence>…</evidence> block.
    Returns a dict with the four required keys, or -1 if tags are missing.
    The 'score' field is returned as float; -1.0 signals a failed conversion.
    """
    out: Dict[str, Union[str, float]] = {}
    for tag, pattern in TAG_PATTERNS.items():
        match = pattern.search(block)
        if not match:           # required tag missing
            return -1
        content = _clean(match.group(1))

        if tag == "score":
            try:
                out[tag] = float(content)
            except ValueError:  # cannot parse → sentinel value
                out[tag] = -1.0
        else:
            out[tag] = content
    return out


def parse_evidence_text(raw_text: str) -> Union[List[Dict[str, Union[str, float]]], int]:
    """
    Extract every <evidence> … </evidence> section from *raw_text* and
    return a list of dictionaries.  If no evidence blocks are found or a
    block is missing any required tag, return -1.
    """
    evidence_blocks = EVIDENCE_RE.findall(raw_text)
    if not evidence_blocks:
        return -1

    parsed: List[Dict[str, Union[str, float]]] = []
    for block in evidence_blocks:
        result = _parse_block(block)
        if result == -1:
            return -1          # propagate fatal format error
        parsed.append(result)
    return parsed

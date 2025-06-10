from .templates import TASK_AWARE_JUDGE_PROMPT_TEMPLATE, EVIDENCE_AWARE_JUDGE_PROMPT_TEMPLATE

# Evidence sufficiency:
JUDGE_PROMPT_VCI_ESCORE_TASK_AWARE_TASK_DESC =\
"""The task for which the explanations were generated is to take a scientific article and determine which evidence code applies to the evidence presented in the paper.
Given for this problem is 1) a genetic variant, 2) a disease in which that genetic variant might be pathogenic or benign, 3) a mode of inheritance, and 4) text of a scientific article that is to be analyzed.
Here's some information on the evidence codes:
Each pathogenic criterion is weighted as very strong (PVS1), strong (PS1-4); moderate (PM1-6), or supporting (PP1-5) and each benign criterion is weighted as stand-alone
 (BA1), strong (BS1-4) or supporting (BP1-6). The numbering within each category does not convey any differences of weight and are merely labeled to help in referring to 
 the different criteria. For a given variant the user selects the criteria based on the evidence observed for the variant."""

EVIDENCE_PROMPT_TASK_AWARE = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
{evidence_code}
"""

EVIDENCE_PROMPT_PAPER = \
"""PubMed ID: {pmid}
PubMed Abstract: 
{abstract}
PubMed Full Text:
{full_text}
"""

JUDGE_PROMPT_MAP_VCI_ESCORE = {
    "task_aware": TASK_AWARE_JUDGE_PROMPT_TEMPLATE.format(
        task_description = JUDGE_PROMPT_VCI_ESCORE_TASK_AWARE_TASK_DESC,
        input_description_template = "{input_description_template}",
        pubmed_info = "{pubmed_info}",
    ),
    "evidence_aware": EVIDENCE_AWARE_JUDGE_PROMPT_TEMPLATE.format(
        task_description = JUDGE_PROMPT_VCI_ESCORE_TASK_AWARE_TASK_DESC,
        input_description_template = "{input_description_template}",
        pubmed_info = "{pubmed_info}",
    ),
}
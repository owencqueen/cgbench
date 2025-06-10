from .templates import TASK_AWARE_JUDGE_PROMPT_TEMPLATE, EVIDENCE_AWARE_JUDGE_PROMPT_TEMPLATE

# Evidence sufficiency:
JUDGE_PROMPT_VCI_TASK_AWARE_TASK_DESC =\
"""The task for which the explanations were generated is to take a scientific article and determine if a given evidence code, which is accompanied by a description, is met or not met from the evidence within the paper for a given disease and genetic variant.
Given for this problem is 1) a genetic variant, 2) a disease in which that genetic variant might be pathogenic or benign, 3) a mode of inheritance, and 4) text of a scientific article that is to be analyzed.
The code is "met" if the evidence in the paper meets specified rules for the given evidence code.
The code is "not met" if, upon evaluation, the evidence in the paper does not meet the criteria specified by the code."""

EVIDENCE_PROMPT_TASK_AWARE = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
{evidence_code}
Code status: {prediction}
"""

EVIDENCE_PROMPT_PAPER = \
"""PubMed ID: {pmid}
PubMed Abstract: 
{abstract}
PubMed Full Text:
{full_text}
"""

JUDGE_PROMPT_MAP_VCI_EVER = {
    "task_aware": TASK_AWARE_JUDGE_PROMPT_TEMPLATE.format(
        task_description = JUDGE_PROMPT_VCI_TASK_AWARE_TASK_DESC,
        input_description_template = "{input_description_template}",
        pubmed_info = "{pubmed_info}",
    ),
    "evidence_aware": EVIDENCE_AWARE_JUDGE_PROMPT_TEMPLATE.format(
        task_description = JUDGE_PROMPT_VCI_TASK_AWARE_TASK_DESC,
        input_description_template = "{input_description_template}",
        pubmed_info = "{pubmed_info}",
    ),
}
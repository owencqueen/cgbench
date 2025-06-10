EVIDENCE_CODE_TEMPLATE = \
"""Evidence code: {ecode}
Description: {desc}
"""

# EVIDENCE_PROMPT_WITH_LABEL = \
# """Variant: {variant}
# Disease: {disease}
# Mode of inheritance: {inheritance}
# PubMed ID: {pmid}
# PubMed Abstract: 
# {abstract}
# PubMed Full Text:
# {full_text}
# Evidence code: {evidence_code}
# """

EVIDENCE_PROMPT_WITH_LABEL = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
PubMed Abstract: 
{abstract}
PubMed Full Text:
{full_text}
{evidence_code}
"""

EVIDENCE_PROMPT_WITH_LABEL_EXPLANATION = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
PubMed Abstract: 
{abstract}
PubMed Full Text:
{full_text}
{evidence_code}
Prediction: {prediction}
Explanation: {explanation}
"""

EVIDENCE_PROMPT_WITH_LABEL_ABSTRACT = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
PubMed Abstract: 
{abstract}
Evidence code: {evidence_code}
Prediction: {prediction}
"""

EVIDENCE_PROMPT_WITH_LABEL_ABSTRACT_EXPLANATION = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
PubMed Abstract: 
{abstract}
Evidence code: {evidence_code}
Prediction: {prediction}
Explanation: {explanation}
"""

# No PM information at all : -------------------------------------------
EVIDENCE_PROMPT_NOPM_WITH_LABEL = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
Evidence code: {evidence_code}
Prediction: {prediction}
"""

EVIDENCE_PROMPT_NOPM_WITH_LABEL_EXPLANATION = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
Evidence code: {evidence_code}
Prediction: {prediction}
Explanation: {explanation}
"""


INSTRUCTION_SYSTEM_PROMPT = \
"""You are an evidence critic that is highly skilled in clinical genetics, especially in clinical classification of variants. 
Your job is to take a scientific article and determine if a given evidence code, which is accompanied by a description, is met or not met from the evidence within the paper for a given disease and genetic variant.
The code is "met" if the evidence in the paper meets specified rules for the given evidence code.
The code is "not met" if, upon evaluation, the evidence in the paper does not meet the criteria specified by the code.

You will be given a variant, disease, mode of inheritance, PubMed paper, and evidence code to accomplish this task.
Your determination should be made dependent on the variant, disease, and mode of inheritance given.
Make your judgement based on the evidence presented in the paper, and use logical reasoning to determine your answer.
Think step-by-step, and use your best knowledge of clinical genetics and literature curation for clinical classifications.
Along with your prediction, produce an explanation for which parts of the paper meet the evidence code provided.

Provide your answer in the following format:
Prediction: "met" or "not met"
Explanation: <Your explanation for why you believe this code is met or not met by the evidence in this paper>
"""

INSTRUCTION_SYSTEM_PROMPT_WITH_ICL = \
"""You are an evidence critic that is highly skilled in clinical genetics, especially in clinical classification of variants. 
Your job is to take a scientific article and determine if a given evidence code, which is accompanied by a description, is met or not met from the evidence within the paper for a given disease and genetic variant.
The code is "met" if the evidence in the paper meets specified rules for the given evidence code.
The code is "not met" if, upon evaluation, the evidence in the paper does not meet the criteria specified by the code.

You will be given a variant, disease, mode of inheritance, PubMed paper, and evidence code to accomplish this task.
Your determination should be made dependent on the variant, disease, and mode of inheritance given.
Make your judgement based on the evidence presented in the paper, and use logical reasoning to determine your answer.
Think step-by-step, and use your best knowledge of clinical genetics and literature curation for clinical classifications.
Along with your prediction, produce an explanation for which parts of the paper meet the evidence code provided.

Provide your answer in the following format:
Prediction: "met" or "not met"
Explanation: <Your explanation for why you believe this code is met or not met by the evidence in this paper>

Here are some examples of correct answers to this question for other papers:
{icl_string}
"""
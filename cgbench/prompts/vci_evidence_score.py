EVIDENCE_PROMPT_NO_LABEL = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
PubMed Abstract: 
{abstract}
PubMed Full Text:
{full_text}
"""

EVIDENCE_PROMPT_NO_LABEL_ABSTRACT = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
PubMed Abstract: 
{abstract}
"""

EVIDENCE_PROMPT_NO_LABEL_PMID = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
"""

# Full text: -------------------------------------------
EVIDENCE_PROMPT_WITH_LABEL = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
PubMed Abstract: 
{abstract}
PubMed Full Text:
{full_text}

Evidence code: {evidence_code}
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

Evidence code: {evidence_code}
Explanation: {explanation}
"""

# Abstract: -------------------------------------------
EVIDENCE_PROMPT_WITH_LABEL_ABSTRACT = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
PubMed Abstract: 
{abstract}

Evidence code: {evidence_code}
"""

EVIDENCE_PROMPT_WITH_LABEL_ABSTRACT_EXPLANATION = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}
PubMed ID: {pmid}
PubMed Abstract: 
{abstract}

Evidence code: {evidence_code}
Explanation: {explanation}
"""

# No PM information at all : -------------------------------------------
EVIDENCE_PROMPT_NOPM_WITH_LABEL = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}

Evidence code: {evidence_code}
"""

EVIDENCE_PROMPT_NOPM_WITH_LABEL_EXPLANATION = \
"""Variant: {variant}
Disease: {disease}
Mode of inheritance: {inheritance}

Evidence code: {evidence_code}
Explanation: {explanation}
"""

EVIDENCE_CODE_TEMPLATE = \
"""Evidence code: {ecode}
{desc}
"""

INSTRUCTION_SYSTEM_PROMPT = \
"""You are an evidence critic that is highly skilled in clinical genetics, especially in clinical classification of variants. 
Your job is to take a PubMed article, which is a scientific, peer reviewed paper, and classify it into one of the "evidence codes" given to you.
You will be given a name of a variant of a gene, and you must determine what level of evidence, if any, is provided in the paper.
Make your judgement based on the evidence presented in the paper, and use logical reasoning to determine your answer.
Think step-by-step, and use your best knowledge of clinical genetics and literature curation for clinical classifications.

Here's some information on the evidence codes:
Each pathogenic criterion is weighted as very strong (PVS1), strong (PS1-4); moderate (PM1-6), or supporting (PP1-5) and each benign criterion is weighted as stand-alone
 (BA1), strong (BS1-4) or supporting (BP1-6). The numbering within each category does not convey any differences of weight and are merely labeled to help in referring to 
 the different criteria. For a given variant the user selects the criteria based on the evidence observed for the variant.

Some codes have may modifiers such as BP1_Strong or PM1_Supporting. These modifiers indicate another granularity of strength of evidence from the core code such as BP1 for BP1_Strong, etc.
In the below code specifications, the core code is described with a general description, and finer-grained codes are described with a more detailed description.
Consider the finer-grained codes, such as BP1_Strong, as also meeting the criteria of the core code, BP1, unless the more detailed description directly contradicts the core code description.
In this case, the finer-grained code description takes precedence. 

Here are the evidence codes you must use to classify paper:
{evidence_code_str}

Provide your answer in the following format:
Evidence code: <predicted evidence code, such as BP5 or PP4>
Explanation: <Your explanation for why you classified this evidence as this code>

"""

INSTRUCTION_SYSTEM_PROMPT_STACK = \
"""You are an evidence critic that is highly skilled in clinical genetics, especially in clinical classification of variants. 
Your job is to take a PubMed article, which is a scientific, peer reviewed paper, and classify it into one of the "evidence codes" given to you.
You will be given a name of a variant of a gene, and you must determine what level of evidence, if any, is provided in the paper.
Make your judgement based on the evidence presented in the paper, and use logical reasoning to determine your answer.
Think step-by-step, and use your best knowledge of clinical genetics and literature curation for clinical classifications.

Here's some information on the evidence codes:
Each pathogenic criterion is weighted as very strong (PVS1), strong (PS1-4); moderate (PM1-6), or supporting (PP1-5) and each benign criterion is weighted as stand-alone
 (BA1), strong (BS1-4) or supporting (BP1-6). The numbering within each category does not convey any differences of weight and are merely labeled to help in referring to 
 the different criteria. For a given variant the user selects the criteria based on the evidence observed for the variant.
 
Some codes have may modifiers such as BP1_Strong or PM1_Supporting. These modifiers indicate another granularity of strength of evidence from the core code such as BP1 for BP1_Strong, etc.
In the below code specifications, the core code is described with a general description, and finer-grained codes are described with a more detailed description.
Consider the finer-grained codes, such as BP1_Strong, as also meeting the criteria of the core code, BP1, unless the more detailed description directly contradicts the core code description.
In this case, the finer-grained code description takes precedence. 

Here are the evidence codes you must use to classify paper:
{evidence_code_str}

Provide your answer in the following format:
Evidence code: <predicted evidence code, such as BP5 or PP4>
Explanation: <Your explanation for why you classified this evidence as this code>

"""

INSTRUCTION_SYSTEM_PROMPT_STACK_CONCISE_EXPLANATION = \
"""You are an evidence critic that is highly skilled in clinical genetics, especially in clinical classification of variants. 
Your job is to take a PubMed article, which is a scientific, peer reviewed paper, and classify it into one of the "evidence codes" given to you.
You will be given a name of a variant of a gene, and you must determine what level of evidence, if any, is provided in the paper.
Make your judgement based on the evidence presented in the paper, and use logical reasoning to determine your answer.
Think step-by-step, and use your best knowledge of clinical genetics and literature curation for clinical classifications.

Here's some information on the evidence codes:
Each pathogenic criterion is weighted as very strong (PVS1), strong (PS1-4); moderate (PM1-6), or supporting (PP1-5) and each benign criterion is weighted as stand-alone
 (BA1), strong (BS1-4) or supporting (BP1-6). The numbering within each category does not convey any differences of weight and are merely labeled to help in referring to 
 the different criteria. For a given variant the user selects the criteria based on the evidence observed for the variant.
 
Some codes have may modifiers such as BP1_Strong or PM1_Supporting. These modifiers indicate another granularity of strength of evidence from the core code such as BP1 for BP1_Strong, etc.
In the below code specifications, the core code is described with a general description, and finer-grained codes are described with a more detailed description.
Consider the finer-grained codes, such as BP1_Strong, as also meeting the criteria of the core code, BP1, unless the more detailed description directly contradicts the core code description.
In this case, the finer-grained code description takes precedence. 

Here are the evidence codes you must use to classify paper:
{evidence_code_str}

Provide your answer in the following format:
Evidence code: <predicted evidence code, such as BP5 or PP4>
Explanation: <Your explanation for why you classified this evidence as this code>

Please provide your explanation as a concise, scientific description of the evidence presented in the paper and why you chose this code.
When possible, provide reasons why you chose this code over other codes presented, especially codes that are similar.
For example, if you choose PS1_Supporting, explain why you did not choose PS1_Strong or PS1_Moderate.
Your explanation will be read by clinical geneticists, so please use appropriate and precise scientific language.
Cite specific evidence and statistics (if applicable) from the paper when possible.
Please be concise and only include necessary information.

"""

INSTRUCTION_SYSTEM_PROMPT_STACK_SUMMARIZED = \
"""You are an evidence critic that is highly skilled in clinical genetics, especially in clinical classification of variants. 
Your job is to take a PubMed article, which is a scientific, peer reviewed paper, and classify it into one of the "evidence codes" given to you.
You will be given a name of a variant of a gene, and you must determine what level of evidence, if any, is provided in the paper.
Make your judgement based on the evidence presented in the paper, and use logical reasoning to determine your answer.
You are given a summary of the paper, which is generated by a large language model to summarize the relevant parts of it for the variant and disease in question.
Think step-by-step, and use your best knowledge of clinical genetics and literature curation for clinical classifications.

Here's some information on the evidence codes:
Each pathogenic criterion is weighted as very strong (PVS1), strong (PS1-4); moderate (PM1-6), or supporting (PP1-5) and each benign criterion is weighted as stand-alone
 (BA1), strong (BS1-4) or supporting (BP1-6). The numbering within each category does not convey any differences of weight and are merely labeled to help in referring to 
 the different criteria. For a given variant the user selects the criteria based on the evidence observed for the variant.
 
Some codes have may modifiers such as BP1_Strong or PM1_Supporting. These modifiers indicate another granularity of strength of evidence from the core code such as BP1 for BP1_Strong, etc.
In the below code specifications, the core code is described with a general description, and finer-grained codes are described with a more detailed description.
Consider the finer-grained codes, such as BP1_Strong, as also meeting the criteria of the core code, BP1, unless the more detailed description directly contradicts the core code description.
In this case, the finer-grained code description takes precedence. 

Here are the evidence codes you must use to classify paper:
{evidence_code_str}

Provide your answer in the following format:
Evidence code: <predicted evidence code, such as BP5 or PP4>
Explanation: <Your explanation for why you classified this evidence as this code>

"""

FORMAT_INSTRUCTION_PROMPT = \
"""
Provide your answer in the following format:
Evidence code: <predicted evidence code, such as BP5 or PP4>
Explanation: <Your explanation for why you classified this evidence as this code>
"""
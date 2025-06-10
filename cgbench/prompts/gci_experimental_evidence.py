from pydantic import BaseModel, Field
from typing import Optional

import sys
sys.path.append("/home/users/oqueen/clingen_benchmark/cgbench")
from util_agents import MODEL_API_MAP

EVIDENCE_EXTRACTION_SYSTEM_PROMPT =\
"""You are an expert in molecular biology and clinical genetics, especially determining the meaning of experimental evidence and how it contributes the interpretation of a gene-disease relationship.
Your job is to extract and score the quality of experimental evidence from a scientific publication and determine it's contribution to the interpretation of a gene-disease relationship.
You will be given a gene, disease, mode of inheritance, and paper from PubMed, and you will be asked to provide structured pieces of evidence from this paper that contribute to how the gene may or may not be implicated in the molecular etiology of the disease given.
"""

EVIDENCE_EXTRACTION_INPUT_TEMPLATE = \
"""
Gene: {gene}
Disease: {disease}
Mode of inheritance: {mode_of_inheritance}
PubMed ID: {pmid}
PubMed Abstract:
{abstract}
PubMed Full Text:
{full_text}
"""

# Label templates -----------------------------------------
EVIDENCE_EXTRACTION_LABEL_TEMPLATE = \
"""
Gene: {gene}
Disease: {disease}
Mode of inheritance: {mode_of_inheritance}
PubMed ID: {pmid}
PubMed Abstract:
{abstract}
PubMed Full Text:
{full_text}

Extracted Evidence:
{extracted_evidence}
"""

EVIDENCE_EXTRACTION_LABEL_TEMPLATE_OMIT_PAPER = \
"""
Gene: {gene}
Disease: {disease}
Mode of inheritance: {mode_of_inheritance}

Extracted Evidence:
{extracted_evidence}
"""

EVIDENCE_EXTRACTION_LABEL_TEMPLATE_OMIT_FULLTEXT = \
"""
Gene: {gene}
Disease: {disease}
Mode of inheritance: {mode_of_inheritance}
PubMed ID: {pmid}
PubMed Abstract:
{abstract}

Extracted Evidence:
{extracted_evidence}
"""
# ---------------------------------------------------------

EXPERIMENTAL_CATEGORY_PROMPT_BASE = \
"""The experimental category classifies this extracted piece of evidence as provided by the guidelines.
This category describes what type of experimental evidence is described in this case; descriptions of these categories are provided in the list below. 
Your prediction should be only one of the experimental categories given in the following list:
{experimental_category_list}
"""

EXPERIMENTAL_CATEGORY_CODE_TEMPLATE =\
"""Category: {category}
Description: {description}
"""

EXPERIMENTAL_SCORE_TEMPLATE = """Category: {category}
Default Score: {default_score}
Score range: {lower_bound}-{upper_bound}
"""

EXPERIMENTAL_SCORE_TEMPLATE_NO_CATEGORY = \
"""Default Score: {default_score}
Score range: {lower_bound}-{upper_bound}
"""

EXPLANATION_PROMPT_BASE = "Detailed explanation of the findings for this piece of evidence and how it contributes to the molecular etiology of the disease."

SCORE_PROMPT_BASE = """Numerical score indicating the strength of the evidence. 
A higher score denotes stronger evidence while lower denotes weaker evidence. Provide a score within the range provided in increments of 0.25. 
Your score should start at the default score if the evidence meets the criteria, then you can add or subtract points based on the strength, e.g., you should deduct points if you believe the evidence is weaker than the default guidelines and add points if it is stronger. 
The default score and ranges for each category are given below:
{experimental_score_list_str}"""

SCORE_ADJUSTMENT_PROMPT_BASE = "Detailed explanation of why you chose to either deduct or add points to the default score for this evidence. Please provide detailed comments on the strength of the evidence and how this contributes to your assessment."

STRUCTURE_PROMPT = \
"""
You must choose from the following list of experimental evidence categories, with defined score ranges and default scores. More instruction on these score ranges and default scores are provided below. Here are the categories:
{experimental_category_list_str}

You will output your response in between tags denoting different portions of the extracted evidence:

<evidence>
<category>
The experimental category classifies this extracted piece of evidence as provided by the guidelines.
This category describes what type of experimental evidence is described in this case; descriptions of these categories are provided in the list below. 
Categories must only be chosen from the list above, and the categories you output must exactly match the name of the categories given above.
</category>
<explanation>
Detailed explanation of the findings for this piece of evidence and how it contributes to the molecular etiology of the disease.
</explanation>
<score>
Numerical score indicating the strength of the evidence. A higher score denotes stronger evidence while lower denotes weaker evidence. Provide a score within the range provided in increments of 0.25. 
Your score should start at the default score if the evidence meets the criteria, then you can add or subtract points based on the strength, e.g., you should deduct points if you believe the evidence is weaker than the default guidelines and add points if it is stronger. 
Scores are defined as above for the given categories. 
</score>
<score_adjustment_reason>
Detailed explanation of why you chose to either deduct or add points to the default score for this evidence. Please provide detailed comments on the strength of the evidence and how this contributes to your assessment.
</score_adjustment_reason>
</evidence>

The <evidence></evidence> tags wrap the entire response. You may output multiple groups of evidence tags in your response if the presented paper contains multiple pieces of evidence that meet the criteria.

If you do not use this format, your outputs are invalid. Adhere to the format strictly.
"""

ONE_STRUCTURED_OUTPUT_PROMPT = \
"""
<category>
{category}
</category>
<explanation>
{explanation}
</explanation>
<score>
{score}
</score>
<score_adjustment_reason>
{score_adjustment_reason}
</score_adjustment_reason>
"""

SOP_SCORE_RANGE = {
    "Biochemical Function A": (0.5, 0, 2),
    "Biochemical Function B": (0.5, 0, 2),
    "Protein Interaction": (0.5, 0, 2),
    "Expression A": (0.5, 0, 2),
    "Expression B": (0.5, 0, 2),
    "Functional Alteration Patient cells": (1.0, 0, 2),
    "Functional Alteration Non-patient cells": (0.5, 0, 1),
    "Model System Non-human model organism": (2, 0, 4),
    "Model Systems Cell culture model": (1, 0, 2),
    "Rescue Human": (2, 0, 4),
    "Rescue Patient Cells": (1, 0, 2),
    "Rescue Non-human model organism": (2, 0, 4),
    "Rescue Cell culture model": (1, 0, 2),
}

class ExperimentalEvidenceOutput_DEFAULT(BaseModel):
    """
    Recommended not to use this one, just for placeholder
        - Need to give options for experimental category
    """

    experimental_category: str = Field(
        description=EXPERIMENTAL_CATEGORY_PROMPT_BASE
    )
    explanation: str = Field(
        description=EXPLANATION_PROMPT_BASE
    )
    score: float = Field(
        description=SCORE_PROMPT_BASE,
        ge=0.0,
        le=1.0
    )
    score_adjustment_reason: Optional[str] = Field(
        default=None,
        description=SCORE_ADJUSTMENT_PROMPT_BASE
    )

def ee_string_prompt_with_category_options(
    experimental_category_list: list[str],
    experimental_category_list_descriptions: list[str],
):

    ecode_template_list = [] 
    for c, d in zip(experimental_category_list, experimental_category_list_descriptions):
        ecat_str = EXPERIMENTAL_CATEGORY_CODE_TEMPLATE.format(
            category=c,
            description=d,
        )
        
        escore_str = EXPERIMENTAL_SCORE_TEMPLATE_NO_CATEGORY.format(
            default_score=SOP_SCORE_RANGE[c][0],
            lower_bound=SOP_SCORE_RANGE[c][1],
            upper_bound=SOP_SCORE_RANGE[c][2],
        )

        ecode_template_list.append(ecat_str + escore_str)

    experimental_category_list_str = "\n".join(ecode_template_list)

    structure_prompt = STRUCTURE_PROMPT.format(
        experimental_category_list_str=experimental_category_list_str,
    )

    return structure_prompt

def ee_structured_output_with_category_options(
    experimental_category_list: list[str],
    experimental_category_list_descriptions: list[str],
    model_name: str = "gpt-4o-mini",
):

    ecode_template_list = []
    ecode_score_list = []
    for c, d in zip(experimental_category_list, experimental_category_list_descriptions):
        ecode_template_list.append(
            EXPERIMENTAL_CATEGORY_CODE_TEMPLATE.format(
                category=c,
                description=d,
            )
        )
        ecode_score_list.append(
            EXPERIMENTAL_SCORE_TEMPLATE.format(
                category=c,
                default_score=SOP_SCORE_RANGE[c][0],
                lower_bound=SOP_SCORE_RANGE[c][1],
                upper_bound=SOP_SCORE_RANGE[c][2],
            )
        )
    experimental_category_list_str = "\n".join(ecode_template_list)
    experimental_score_list_str = "\n".join(ecode_score_list)

    if MODEL_API_MAP[model_name] == "OPENAI":
        class ExperimentalEvidenceOutput(BaseModel):
            experimental_category: str = Field(
                description=EXPERIMENTAL_CATEGORY_PROMPT_BASE.format(experimental_category_list=experimental_category_list_str)
            )
            explanation: str = Field(
                description=EXPLANATION_PROMPT_BASE
            )
            score: float = Field(
                description=SCORE_PROMPT_BASE.format(experimental_score_list_str = experimental_score_list_str),
            )
            score_adjustment_reason: Optional[str] = Field(
                default=None,
                description=SCORE_ADJUSTMENT_PROMPT_BASE
            )

        class EEOutputList(BaseModel):
            evidence_list: list[ExperimentalEvidenceOutput] = Field(
                description="List of experimental evidence findings. See individual ExperimentalEvidenceOutput descriptions for a more precise definition of these units."
            )

    elif MODEL_API_MAP[model_name] == "ANTHROPIC":
        EEOutputList = {
        "type": "object",
        "properties": {
            "evidence_list": {
            "type": "array",
            "description": "List of experimental evidence findings.",
            "items": {
                "type": "object",
                "properties": {
                "experimental_category": {
                    "type": "string",
                    "description": EXPERIMENTAL_CATEGORY_PROMPT_BASE.format(experimental_category_list=experimental_category_list_str)
                },
                "explanation": {
                    "type": "string",
                    "description": EXPLANATION_PROMPT_BASE
                },
                "score": {
                    "type": "number",
                    "description": SCORE_PROMPT_BASE.format(experimental_score_list_str = experimental_score_list_str)
                },
                "score_adjustment_reason": {
                    "type": ["string", "null"],
                    "description": SCORE_ADJUSTMENT_PROMPT_BASE
                }
                },
                "required": ["experimental_category", "explanation", "score"]
            }
            }
        },
        "required": ["evidence_list"]
        }

    
    return EEOutputList
import os, copy, re
from openai import OpenAI
import numpy as np

from .prompts.judge import *
from .util_agents import OnePassAgent

# Prompt map:
# PROMPT_MAP = {
#     "vci_evidence_score": JUDGE_PROMPT_VCI_EVIDENCE_SCORE,
#     "vci_evidence_sufficiency": JUDGE_PROMPT_VCI_EVIDENCE_SUFFICIENCY,
#     "gci_evidence_extraction": JUDGE_PROMPT_GCI_EVIDENCE_EXTRACTION,
# }

def extract_score(out):
    # Look for 'Yes' or 'No' after 'Decision:' using regex, capturing everything up to the next newline
    match = re.search(r'Decision:\s*[\'"]?(Yes|No)[\'"]?[^\n]*', out, re.IGNORECASE)
    if match:
        # Extract the full match and check if it starts with Yes or No
        full_match = match.group(0).strip()
        if full_match.lower().startswith(('yes', 'no')):
            return int(full_match.lower().startswith('yes'))
        # If the full match doesn't start with Yes/No, check if it contains Yes/No
        yes_no_match = re.search(r'[\'"]?(Yes|No)[\'"]?', full_match, re.IGNORECASE)
        if yes_no_match:
            return int(yes_no_match.group(1).lower() == 'yes')
    
    # If no match found with 'Decision:', look for Yes/No at start of line
    match = re.search(r'^[\'"]?(Yes|No)[\'"]?[^\n]*', out, re.IGNORECASE)
    if match:
        full_match = match.group(0).strip()
        return int(full_match.lower().startswith('yes'))
    
    # If still no match, return None to indicate no valid answer found
    return None


class LLMExplanationJudge(object):

    input_prompt = "Explanation 1: \n{}\n\nExplanation 2: \n{}\n\nNow determine whether these two explanations are consistent with one another."

    def __init__(self,
            system_prompt = None, 
            model_name = "gpt-4o-mini",
            temperature = 0.2,
            frequency_penalty = 0.0,
            double_input = True,
            tiebreak = True,
            best_of_k = 1,
        ):

        self.system_prompt = system_prompt
        self.model_name = model_name
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.double_input = double_input
        self.tiebreak = tiebreak

        self.best_of_k = best_of_k
        
        self.llm = OnePassAgent(
            system_prompt = self.system_prompt,
            model_name = model_name,
            temperature = temperature,
            frequency_penalty = frequency_penalty,
        )

    def judge_forward(
            self,
            llm_explanation,
            human_explanation,
            max_tokens = 200,
            prefix_prompt = None,
        ):

        if prefix_prompt is not None:
            inp_prompt = prefix_prompt + "\n" + self.input_prompt.format(llm_explanation, human_explanation)
        else:
            inp_prompt = self.input_prompt.format(llm_explanation, human_explanation)

        llm_out, llm_cost = self.llm(
            input = inp_prompt,
            max_tokens = max_tokens,
            get_cost = True,
        )

        return llm_out, llm_cost

    def majority_vote(self, llm_outputs):
        return (np.mean(llm_outputs) > 0.5).item()
            
    def __call__(self, 
            llm_explanation,
            human_explanation,
            prefix_prompt = None,
        ):

        sample_k = max(self.best_of_k // 2, 1) if self.double_input else self.best_of_k
        llm_cost_total = 0.0

        llm_out_list = []

        for _ in range(sample_k):
            llm_out_1, llm_cost_1 = self.judge_forward(
                llm_explanation,
                human_explanation,
                prefix_prompt = prefix_prompt,
            )

            llm_out_list.append(llm_out_1)
            llm_cost_total += llm_cost_1

        if self.double_input:
            for _ in range(sample_k):
                llm_out_2, llm_cost_2 = self.judge_forward(
                    human_explanation,
                    llm_explanation,
                    prefix_prompt = prefix_prompt,
                )

                llm_out_list.append(llm_out_2)
                llm_cost_total += llm_cost_2
            
        # Extract scores:
        extracted_scores = [extract_score(out) for out in llm_out_list if extract_score(out) is not None] # Ignore None values

        majority_vote_result = self.majority_vote(extracted_scores)

        return majority_vote_result, llm_out_list, llm_cost_total

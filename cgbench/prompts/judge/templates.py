TASK_AWARE_JUDGE_PROMPT_TEMPLATE =\
"""Here is the task given in this example:
{task_description}

Here is the input given for this example:
{input_description_template}
"""

EVIDENCE_AWARE_JUDGE_PROMPT_TEMPLATE =\
"""Here is the task given in this example:
{task_description}

Here is the input given for this example:
{input_description_template}

And the paper referred to in the explanations:
{pubmed_info}
"""
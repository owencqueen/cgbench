import os, copy
from openai import OpenAI
from openai import BadRequestError
from openai.types.chat import ChatCompletionMessage
import anthropic

from together import Together

MODEL_API_MAP = {
    "gpt-4o": "OPENAI",
    "gpt-4o-2024-11-20": "OPENAI",
    "gpt-4o-mini": "OPENAI",
    "o1": "OPENAI",
    "o1-mini": "OPENAI",
    "o3": "OPENAI",
    "o3-mini": "OPENAI",
    "o4-mini": "OPENAI",
    "o3-preview": "OPENAI",
    "claude-3-5-sonnet": "ANTHROPIC",
    "claude-3-haiku": "ANTHROPIC",
    # "llama-4-maverick": "TOGETHER",
    # "llama-3.3-70b": "TOGETHER",
    # "deepseek-r1": "TOGETHER",
    "claude-3-7-sonnet-20250219": "ANTHROPIC",
    "llama-4-maverick": "TOGETHER",
    "qwen2.5-72b":      "TOGETHER",
    "deepseek-r1":      "TOGETHER",
    "google/gemini-2.5-flash-preview": "OPENROUTER",
    "google/gemini-2.5-pro-preview-03-25": "OPENROUTER",
}

def openai_api_calculate_cost(response,model="gpt-4o"):
    # Pricing as of early March 2025
    pricing = {
        'o1': {
            'prompt': 15.0,
            'completion': 60.0,
        },
        'o3': {
            'prompt': 10.0,
            'completion': 40.0,
        },
        'gpt-4o': {
            'prompt': 2.5,
            'completion': 10.0,
        },
        'gpt-4o-2024-11-20': {
            'prompt': 2.5,
            'completion': 10.0,
        },
        'o3-mini': {
            'prompt': 1.10,
            'completion': 4.40,
        },
        'gpt-4o-mini': {
            'prompt': 0.30,
            'completion': 1.20,
        },
        "o4-mini": {
            "prompt": 1.10,
            "completion": 4.40,
        }
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = response.usage.prompt_tokens * model_pricing['prompt'] / 1000000
    completion_cost = response.usage.completion_tokens * model_pricing['completion'] / 1000000

    total_cost = prompt_cost + completion_cost
    # print(f"\nTokens used:  {usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion = {usage['total_tokens']:,} tokens")
    # print(f"Total cost for {model}: ${total_cost:.4f}\n")

    return total_cost

def anthropic_api_calculate_cost(response, model="claude-3-5-sonnet"):
    # Pricing as of early 2024
    pricing = {
        "claude-3-7-sonnet-20250219": {
            'prompt': 3.0,
            'completion': 15.0,
        },
        "claude-3-5-sonnet": {
            'prompt': 3.0,
            'completion': 15.0,
        },
        "claude-3-haiku": {
            'prompt': 0.25,
            'completion': 1.25,
        },
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = response.usage.input_tokens * model_pricing['prompt'] / 1000000
    completion_cost = response.usage.output_tokens * model_pricing['completion'] / 1000000

    total_cost = prompt_cost + completion_cost
    # print(f"\nTokens used: {response.usage.prompt_tokens:,} prompt + {response.usage.completion_tokens:,} completion = {response.usage.total_tokens:,} tokens")
    # print(f"Total cost for {model}: ${total_cost:.4f}\n")

    return total_cost

TOGETHER_MODEL_STRINGS = {
    "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "qwen2.5-72b":      "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "deepseek-r1":      "deepseek-ai/DeepSeek-R1",
}

TOGETHER_PRICING = {
    "llama-4-maverick": {"prompt": 0.27, "completion": 0.85},
    "qwen2.5-72b":      {"prompt": 0.60, "completion": 0.60},
    "deepseek-r1":      {"prompt": 3.00, "completion": 7.00},
}

def together_api_calculate_cost(response, model="llama-4-maverick"):
    try:
        price = TOGETHER_PRICING[model]
    except KeyError:
        raise ValueError("Invalid Together model for cost calc")

    prompt_cost = response.usage.prompt_tokens     * price["prompt"]     / 1_000_000
    completion_cost = response.usage.completion_tokens * price["completion"] / 1_000_000
    return prompt_cost + completion_cost

OPENROUTER_PRICING = {
    "google/gemini-2.5-flash-preview": {"prompt": 0.15, "completion": 0.60},
    "google/gemini-2.5-flash-preview:thinking": {"prompt": 0.15, "completion": 3.50},
    "google/gemini-2.5-pro-preview-03-25": {"prompt": 1.25, "output": 1.00},
}
def openrouter_api_calculate_cost(resp, model):
    price = OPENROUTER_PRICING[model]
    return (resp.usage.prompt_tokens  * price["prompt"] +
            resp.usage.completion_tokens * price["completion"]) / 1_000_000

class OnePassAgent(object):
    def __init__(
            self,  
            system_prompt: str = None,
            model_name: str = "gpt-4o",
            temperature: float = 0.2,
            frequency_penalty: float = 0.0,
        ):

        self.model_name = model_name
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty

        self.api_framework = MODEL_API_MAP[self.model_name]
        
        #if ("o1" in self.model_name) or ("o3" in self.model_name):
            #self.system_prompt = {"role": "user", "content": system_prompt}
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)
        # Means an error will be thrown if system prompt is not set

        
        # Initialize API clients based on the framework
        if self.api_framework == "OPENAI":
            self.api_system = "openai"
            self.client = OpenAI(api_key=os.getenv('ZOULAB_OPENAI_API_KEY'))
        elif self.api_framework == "ANTHROPIC":
            self.api_system = "anthropic"
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        elif self.api_framework == "TOGETHER":
            self.api_system = "together"
            self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
            # Swap the human-friendly alias for the real model string
            self.together_model = TOGETHER_MODEL_STRINGS[self.model_name]
        elif self.api_framework == "OPENROUTER":
            self.api_system = "openrouter"
            self.client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                # optional – makes your calls show up in OR rankings
                default_headers={
                    "HTTP-Referer": "https://your-site.com/",
                    "X-Title":      "OnePassAgent Demo"
                }
            )
        else:
            raise NotImplementedError(f"{model_name} not supported")

    def set_system_prompt(self, system_prompt: str):
        # ("o4" in self.model_name)
        if self.api_framework == "OPENAI" and (not (("o1" in self.model_name) or ("o3" in self.model_name) or ("o4" in self.model_name))) or (self.api_framework == "TOGETHER"):
            self.system_prompt = {"role": "system", "content": system_prompt}
        else:
            self.system_prompt = system_prompt

    def forward(self, 
            input: str, 
            max_tokens: int = 2000, 
            response_format = None, 
            get_cost: bool = False
        ):
        if self.api_framework == "OPENAI":
            if ("o1" in self.model_name) or ("o3" in self.model_name) or ("o4" in self.model_name):
                message_input = [{"role": "user", "content": self.system_prompt + "\n" + input}]
                #message_input["content"] += "\n" + input
                #message_input = [message_input]

                if response_format is None:
                    response = self.client.chat.completions.create(
                            model = self.model_name,
                            messages = message_input,
                        )
                    model_out = response.choices[0].message.content
                else:
                    response = self.client.beta.chat.completions.parse(
                            model = self.model_name,
                            messages = message_input,
                            response_format = response_format
                        )
                    model_out = response.choices[0].message.parsed
            else:
                message_input = [self.system_prompt, {"role": "user", "content": input}]

                if response_format is None:

                    response = self.client.chat.completions.create(
                            model = self.model_name,
                            messages = message_input,
                            temperature = self.temperature,
                            max_tokens = max_tokens,
                            frequency_penalty = self.frequency_penalty,
                    )
                    model_out = response.choices[0].message.content
                else:

                    response = self.client.beta.chat.completions.parse(
                            model = self.model_name,
                            messages = message_input,
                            temperature = self.temperature,
                            max_tokens = max_tokens,
                            frequency_penalty = self.frequency_penalty,
                            response_format = response_format,
                        )
                
                    model_out = response.choices[0].message.parsed
        elif self.api_framework == "ANTHROPIC":
            # For Anthropic models, we need to handle the system prompt differently
            # if isinstance(self.system_prompt, dict):
            #     system_content = self.system_prompt["content"]
            # else:
            system_content = self.system_prompt
            
            # Handle JSON structured input for Anthropic models
            if response_format is not None:
                # For Anthropic models, we need to use the tools parameter for JSON structured output
                # Extract the schema from the response_format
                # if hasattr(response_format, 'schema'):
                #     schema = response_format.schema
                # else:
                #     # If schema is not directly available, try to extract it from the response_format
                #     schema = response_format
                
                # Create the message for Anthropic API with tools parameter
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system_content,
                    messages=[
                        {"role": "user", "content": input}
                    ],
                    tools=[{
                        "name": "analyze_experimental_evidence",
                        "description": "Analyze experimental evidence and provide structured findings",
                        "input_schema": response_format
                    }],
                    tool_choice={"type": "tool", "name": "analyze_experimental_evidence"}
                )

                # Extract the tool call result
                if response.content and len(response.content) > 0:
                    if response.content[0].type == "tool_use":
                        tool_call = response.content[0]
                        # Check if the tool call has input data
                        if hasattr(tool_call, 'input') and tool_call.input:
                            model_out = tool_call.input
                        else:
                            # If input is empty, try to get text content or provide a default
                            if hasattr(response, 'text') and response.text:
                                model_out = response.text
                            else:
                                # If no valid output is found, return a default or error message
                                model_out = {"error": "No valid output generated by the model"}
                                import ipdb; ipdb.set_trace()
                                raise ValueError("No valid output generated by the model")
                    else:
                        # Handle other content types
                        model_out = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
                else:
                    # Fallback if no content is available
                    model_out = {"error": "No content in the response"}
                    raise ValueError("No content in the response")

            else:
                # Standard text output for Anthropic models
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system_content,
                    messages=[
                        {"role": "user", "content": input}
                    ]
                )
                
                model_out = response.content[0].text
        elif self.api_framework == "TOGETHER":
            # ensure the system prompt is a dict
            if isinstance(self.system_prompt, str):
                sys_msg = {"role": "system", "content": self.system_prompt}
            else:
                sys_msg = self.system_prompt

            message_input = [sys_msg, {"role": "user", "content": input}]
            if "deepseek" in self.model_name:
                max_tokens = None
                response = self.client.chat.completions.create(
                    model=self.together_model,
                    messages=message_input,
                    temperature=self.temperature,
                    frequency_penalty=self.frequency_penalty,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.together_model,
                    messages=message_input,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=self.frequency_penalty,
                )
            model_out = response.choices[0].message.content

        elif self.api_framework == "OPENROUTER":
            message_input = [self.system_prompt, {"role": "user", "content": input}]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_input,
                temperature=self.temperature,
                max_tokens=max_tokens,
                frequency_penalty=self.frequency_penalty,
            )
            model_out = response.choices[0].message.content

        if get_cost:
            if self.api_framework == "OPENAI":
                return model_out, openai_api_calculate_cost(response, model=self.model_name)
            elif self.api_framework == "ANTHROPIC":
                return model_out, anthropic_api_calculate_cost(response, model=self.model_name)
            elif self.api_framework == "TOGETHER":
                return model_out, together_api_calculate_cost(response, model=self.model_name)
            elif self.api_framework == "OPENROUTER":
                return model_out, openrouter_api_calculate_cost(response, model=self.model_name)
        else:
            return model_out
    
    def __call__(self, input, max_tokens = 2000, get_cost = False):
        return self.forward(input, max_tokens = max_tokens, get_cost = get_cost)

class BestofKAgent(object):
    """
    General agent that runs best-of-k sampling. Doesn't perform aggregation, need another module for that.
    """
    def __init__(
            self,  
            system_prompt: str,
            model_name: str,
            temperature: float = 0.2,
            frequency_penalty: float = 0.0,
            best_of_k: int = 1,
        ):

        self.model_name = model_name
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.best_of_k = best_of_k

        self.llm = OnePassAgent(
            system_prompt = system_prompt,
            model_name = model_name,
            temperature = temperature,
            frequency_penalty = frequency_penalty,
        )

    def forward(self, input, max_tokens = 2000, response_format = None):
        llm_out_list = []
        llm_cost_total = 0.0

        for _ in range(self.best_of_k):
            llm_out, llm_cost = self.llm(input, max_tokens = max_tokens, response_format = response_format, get_cost = True)
            llm_out_list.append(llm_out)
            llm_cost_total += llm_cost

        return llm_out_list, llm_cost_total
    
    


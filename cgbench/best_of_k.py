from .util_agents import OnePassAgent

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

    def set_system_prompt(self, system_prompt: str):
        self.llm.set_system_prompt(system_prompt)

    def forward(self, input, max_tokens = 2000, response_format = None, get_cost = False):
        llm_out_list = []
        llm_cost_total = 0.0

        for _ in range(self.best_of_k):
            llm_out, llm_cost = self.llm.forward(input, max_tokens = max_tokens, response_format = response_format, get_cost = get_cost)
            llm_out_list.append(llm_out)
            llm_cost_total += llm_cost

        if get_cost:
            return llm_out_list, llm_cost_total
        else:
            return llm_out_list
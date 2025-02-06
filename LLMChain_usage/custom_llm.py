from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun

class DualLLM(LLM):
    system_prompt: str = "You are an AI assistant."
    use_alternate_model: bool = False
    verbose: bool = True  # Add verbose flag
    
    @property
    def _llm_type(self) -> str:
        return "dual_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        from hellper import get_llm_response, get_llm_response_mini
        if self.verbose:
            print(f"\n=== LLM Call ===")
            print(f"Using Model: {'O-1' if self.use_alternate_model else 'Mini'}")
            # print(f"Prompt:\n{prompt}")
        response = get_llm_response(prompt) if self.use_alternate_model else get_llm_response_mini(prompt, self.system_prompt)

        if self.verbose:
            print(f"\n=== LLM Response ===")
            print(response)
        
        
        return response
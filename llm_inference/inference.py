from vllm import LLM
from vllm.sampling_params import BeamSearchParams


def inference_with_beam_search(prompt: str, llm: LLM, temperature: float = .0, beam_width: int = 8, max_tokens: int = 512):
    beam_search_params = BeamSearchParams(temperature=temperature, beam_width=beam_width, max_tokens=max_tokens)
    results = llm.generate(prompt, beam_search_params)[0]
    return [res.text.strip() for res in results.outputs]

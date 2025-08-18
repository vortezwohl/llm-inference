from vllm import LLM
from vllm.sampling_params import BeamSearchParams


def inference_with_beam_search(prompt: str, temperature: float, beam_width: int, max_tokens: int, llm: LLM):
    beam_search_params = BeamSearchParams(temperature=temperature, beam_width=beam_width, max_tokens=max_tokens)
    results = llm.generate(prompt, beam_search_params)[0]
    return [res.text.strip() for res in results.outputs]

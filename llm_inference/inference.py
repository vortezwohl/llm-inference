import logging

from vllm import LLM, TextPrompt
from vllm.sampling_params import BeamSearchParams

logger = logging.getLogger('llm_inference')


def inference_with_beam_search(prompt: str, llm: LLM, temperature: float = .0, beam_width: int = 4, max_tokens: int = 4096) -> list[tuple[str, float]]:
    beam_search_params = BeamSearchParams(temperature=temperature, beam_width=beam_width, max_tokens=max_tokens, include_stop_str_in_output=False)
    results = llm.beam_search(prompts=[TextPrompt(prompt=prompt)], params=beam_search_params)[0].sequences
    results.sort(key=lambda x: x.cum_logprob, reverse=True)
    logger.debug(f'inference_with_beam_search: {results}')
    return [(res.text, res.cum_logprob) for res in results]

import logging

from vllm import LLM

from llm_inference.inference import inference_with_beam_search

logger = logging.getLogger('llm_inference')
seed_x_lm = LLM(model='ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8', trust_remote_code=True, gpu_memory_utilization=.8, swap_space=16, cpu_offload_gb=4, max_seq_len_to_capture=2048)


def translate(sentence: str, target_lang: str = 'en', **kwargs) -> str:
    prompt = f'Translate the sentence into <{target_lang.lower().replace("<", "").replace(">", "")}> AFTER explanation in detail: <sentence>{sentence}</sentence> [COT]'
    logger.debug(f'PROMPT: {prompt.replace("\n", " ")}')
    best_ans = sorted(inference_with_beam_search(prompt=prompt, llm=seed_x_lm, **kwargs), key=lambda x: len(x[0]), reverse=True)
    logger.debug(f'ANS: {best_ans}')
    best_ans = best_ans[0]
    prompt = f'{best_ans}<{target_lang.lower().replace("<", "").replace(">", "")}>'
    logger.debug(f'REPROMPT WITH BEST ANS AFTER BEAM SEARCH: {prompt.replace("\n", " ")}')
    best_ans = sorted(inference_with_beam_search(prompt=prompt, llm=seed_x_lm, **kwargs), key=lambda x: len(x[0]), reverse=True)
    logger.debug(f'FIN ANS: {best_ans}')
    best_ans = best_ans[0]
    return best_ans[0]

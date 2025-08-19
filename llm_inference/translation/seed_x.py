import logging

from vllm import LLM

from llm_inference.inference import inference

logger = logging.getLogger('llm_inference')
seed_x_lm = LLM(model='ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8', trust_remote_code=True, gpu_memory_utilization=.975,
                swap_space=2, cpu_offload_gb=1, max_seq_len_to_capture=6144)


def translate(sentence: str, target_lang: str = 'en', resample: int = 1, **kwargs) -> str:
    pre_think = '[COT]'
    post_think = '[COT]'
    pre_stop_seq = '<sentence>'
    stop_seq = '</sentence>'
    regex = rf'.+{stop_seq}'
    kwargs['stop'] = post_think
    prompt = (f'Translate sentence "{sentence}" into "{target_lang.lower()}" and explain in detail:'
              f'<{target_lang.lower().replace("<", "").replace(">", "")}>{pre_think}')
    logger.debug(f'PROMPT: {prompt.replace("\n", " ")}')
    best_ans = sorted(inference(prompt=[prompt] * resample, llm=seed_x_lm, **kwargs),
                      key=lambda x: len(x[0]), reverse=True)
    logger.debug(f'ANS: {best_ans}')
    best_ans = best_ans[0]
    kwargs['stop'] = None
    prompt = (f'Translate sentence "{sentence}" into "{target_lang.lower()}" and explain in detail:'
              f'{pre_think}{best_ans.replace(pre_think, "").replace(post_think, "")}{post_think}'
              + inference(['After thinking, the best translation is '], llm=seed_x_lm, **kwargs)[0] +
              f'<{target_lang.lower().replace("<", "").replace(">", "")}>'
              f'{pre_stop_seq}')
    logger.debug(f'REPROMPT WITH BEST ANS: {prompt.replace("\n", " ")}')
    kwargs['max_tokens'] = int(len(sentence) * 2.5)
    kwargs['stop'] = stop_seq
    best_ans = sorted(inference(prompt=[prompt] * resample, llm=seed_x_lm, regex=regex, **kwargs),
                      key=lambda x: len(x[0]), reverse=True)[0].strip()
    logger.debug(f'TRANSLATION: {best_ans}')
    return best_ans

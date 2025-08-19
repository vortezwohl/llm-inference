import logging

from vllm import LLM

from llm_inference.inference import inference

logger = logging.getLogger('llm_inference')
seed_x_lm = LLM(model='ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8', trust_remote_code=True, gpu_memory_utilization=.975,
                swap_space=2, cpu_offload_gb=1, max_seq_len_to_capture=6144)


def translate(sentence: str, target_lang: str = 'en', resample: int = 1, **kwargs) -> str:
    pre_think = '[COT]'
    post_think = '[COT]'
    stop_seq = '[END]'
    lang_seq = f'<{target_lang.lower().replace("<", "").replace(">", "")}>'
    regex = rf'.+"{stop_seq}'
    kwargs['stop'] = post_think
    prompt = (f'Translate sentence "{sentence}" into "{target_lang.lower()}" and explain in detail:'
              f'{lang_seq}{pre_think}')
    logger.debug(f'PROMPT: {prompt.replace("\n", " ")}')
    best_ans = sorted(inference(prompt=[prompt] * resample, llm=seed_x_lm, **kwargs),
                      key=lambda x: len(x[0]), reverse=True)[0]
    logger.debug(f'ANS: {best_ans}')
    kwargs['stop'] = None
    prompt = (f'Translate sentence "{sentence}" into "{target_lang.lower()}":'
              f'{lang_seq}{pre_think}{best_ans}{post_think}'
              + (inference([f'Translate sentence "After all thinking above, the best {target_lang} translation is:" into "{target_lang}": {lang_seq}'],
                           llm=seed_x_lm, **kwargs)[0] if 'en' not in target_lang
                 else 'After all thinking above, the best english translation is:') + '"')
    logger.debug(f'REPROMPT WITH BEST ANS: {prompt.replace("\n", " ")}')
    kwargs['max_tokens'] = int(len(sentence) * 2.5)
    kwargs['stop'] = stop_seq
    best_ans = sorted(inference(prompt=[prompt] * resample, llm=seed_x_lm, regex=regex, **kwargs),
                      key=lambda x: len(x[0]), reverse=True)[0][1:-1].strip()
    logger.debug(f'TRANSLATION: {best_ans}')
    return best_ans

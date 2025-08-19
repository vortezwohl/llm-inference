import logging

from vllm import LLM

from llm_inference.inference import inference

logger = logging.getLogger('llm_inference')
seed_x_lm = LLM(model='ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8', trust_remote_code=True, gpu_memory_utilization=.92,
                swap_space=2, cpu_offload_gb=1, max_seq_len_to_capture=6144)


def translate_cot(sentence: str, target_lang: str = 'en', resample: int = 1, **kwargs) -> str:
    pre_think = '[COT]'
    post_think = '[COT]'
    regex = rf'.+"{post_think}'
    lang_seq = f'<{target_lang.lower().replace("<", "").replace(">", "")}>'
    kwargs['stop'] = post_think
    prompt = (f'Translate sentence "{sentence}" into "{target_lang.lower()}" and explain in detail:'
              f'{lang_seq}{pre_think}')
    logger.debug(f'PROMPT: {prompt.replace("\n", " ")}')
    cot = sorted(inference(prompt=[prompt] * resample, llm=seed_x_lm, regex=regex, **kwargs),
                      key=lambda x: len(x[0]), reverse=True)[0]
    logger.debug(f'COT: {cot}')
    return cot


def translate(sentence: str, target_lang: str = 'en', **kwargs) -> str:
    lang_seq = f'<{target_lang.lower().replace("<", "").replace(">", "")}>'
    stop_seq = '[END]'
    regex = rf'.+"{stop_seq}'
    prompt = inference([f'Translate sentence [$PLACEHOLDER$] into "{target_lang}":'],
                       llm=seed_x_lm, **kwargs)[0].replace('[$PLACEHOLDER$]', f'"{sentence}"') + lang_seq
    kwargs['max_tokens'] = int(len(seed_x_lm.get_tokenizer().encode(sentence)) * 2.75)
    kwargs['stop'] = stop_seq
    translation = inference(prompt=[prompt], llm=seed_x_lm, regex=regex, **kwargs)[0].strip()
    logger.debug(f'TRANSLATION: {translation}')
    return translation

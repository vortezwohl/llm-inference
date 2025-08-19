import logging

from vllm import LLM

from llm_inference.inference import inference

logger = logging.getLogger('llm_inference')
seed_x_lm = LLM(model='ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8', trust_remote_code=True, gpu_memory_utilization=.975,
                swap_space=2, cpu_offload_gb=1, max_seq_len_to_capture=4096)


def translate_cot(sentence: str, target_lang: str = 'en', resample: int = 1, **kwargs) -> str:
    pre_think = '[COT]'
    post_think = '[COT]'
    lang_seq = f'<{target_lang.lower().replace("<", "").replace(">", "")}>'
    kwargs['stop'] = post_think
    prompt = f'Translate the sentence and explain in detail.<sentence>{sentence}</sentence>{lang_seq}{pre_think}'
    logger.debug(f'PROMPT: {prompt.replace("\n", " ")}')
    cot = sorted(inference(prompt=[prompt] * resample, llm=seed_x_lm, **kwargs),
                 key=lambda x: len(x[0]), reverse=True)[0]
    logger.debug(f'COT: {cot}')
    return cot

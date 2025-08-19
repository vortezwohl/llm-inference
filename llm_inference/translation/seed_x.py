import logging

from vllm import LLM

from llm_inference.inference import inference

logger = logging.getLogger('llm_inference')
seed_x_lm = LLM(model='ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8', trust_remote_code=True, gpu_memory_utilization=.975, swap_space=2, cpu_offload_gb=2, max_seq_len_to_capture=4096)


def translate(sentence: str, target_lang: str = 'en', resample: int = 1, **kwargs) -> str:
    prompt = (f'Translate the sentence into "{target_lang.lower()}" and explain in detail:'
              f'<sentence>{sentence}</sentence><{target_lang.lower().replace("<", "").replace(">", "")}>[COT]')
    logger.debug(f'PROMPT: {prompt.replace("\n", " ")}')
    best_ans = sorted(inference(prompt=[prompt] * resample, llm=seed_x_lm, **kwargs), key=lambda x: len(x[0]), reverse=True)
    logger.debug(f'ANS: {best_ans}')
    best_ans = best_ans[0]
    prompt = (f'[COT]{best_ans}[COT]Translate the sentence into "{target_lang.lower()}":'
              f'<sentence>{sentence}</sentence><{target_lang.lower().replace("<", "").replace(">", "")}><sentence>')
    logger.debug(f'REPROMPT WITH BEST ANS: {prompt.replace("\n", " ")}')
    best_ans = sorted(inference(prompt=[prompt] * resample, llm=seed_x_lm, **kwargs), key=lambda x: len(x[0]), reverse=True)
    logger.debug(f'FIN ANS: {best_ans[0]}')
    best_ans = best_ans[0]
    return best_ans[0]

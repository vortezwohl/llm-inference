from vllm import LLM

from llm_inference.inference import inference_with_beam_search

seed_x_lm = LLM(model='ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8', trust_remote_code=True, gpu_memory_utilization=.999, swap_space=8, cpu_offload_gb=4)


def translate(sentence: str, target_lang: str = 'en', **kwargs):
    prompt = f'Translate the following sentence into "{target_lang}" and explain it in detail: {sentence} <{target_lang.lower().replace("<", "").replace(">", "")}>'
    return inference_with_beam_search(prompt=prompt, llm=seed_x_lm, **kwargs)

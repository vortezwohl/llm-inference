from vllm import LLM

from llm_inference.inference import inference_with_beam_search

seed_x_lm = LLM(model='ByteDance-Seed/Seed-X-PPO-7B-GPTQ-Int8', trust_remote_code=True, gpu_memory_utilization=.8, swap_space=8, cpu_offload_gb=16, max_seq_len_to_capture=2048)


def translate(sentence: str, target_lang: str = 'en', **kwargs) -> list[tuple[str, float]]:
    prompt = f'Translate the sentence into <{target_lang.lower().replace("<", "").replace(">", "")}> AFTER explanation in detail: <sentence>{sentence}</sentence> [COT]'
    best_ans = sorted(inference_with_beam_search(prompt=prompt, llm=seed_x_lm, **kwargs), key=lambda x: len(x[0]), reverse=True)
    prompt = f'{best_ans}<{target_lang.lower().replace("<", "").replace(">", "")}>'
    return sorted(inference_with_beam_search(prompt=prompt, llm=seed_x_lm, **kwargs), key=lambda x: len(x[0]), reverse=True)

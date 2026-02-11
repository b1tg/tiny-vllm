import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
  # Try multiple model locations
  paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Qwen3-0.6B"),
    os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"),
  ]

  path = None
  for p in paths:
    if os.path.exists(p):
      path = p
      break

  if path is None:
    # Try downloading via huggingface_hub
    from huggingface_hub import snapshot_download
    path = snapshot_download("Qwen/Qwen3-0.6B")

  tokenizer = AutoTokenizer.from_pretrained(path)
  llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

  sampling_params = SamplingParams(temperature=0.6, max_tokens=10)
  prompts = ["Hello"]
  prompts = [
    tokenizer.apply_chat_template(
      [{"role": "user", "content": prompt}],
      tokenize=False,
      add_generation_prompt=True,
    )
    for prompt in prompts
  ]
  outputs = llm.generate(prompts, sampling_params)
  result = ""
  for prompt, output in zip(prompts, outputs):
    print(f"\nPrompt: {prompt!r}")
    print(f"Completion: {output['text']!r}")
    result += output['text']
  assert "Okay," in result or "okay" in result.lower() or "hello" in result.lower() or "user" in result.lower(), result


if __name__ == "__main__":
  main()
